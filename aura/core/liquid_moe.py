"""
Liquid-MoE Spike Router for AURA
Continuous-time dynamics with Top-K sparse routing and local learning rules
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np

# ---------------------- utils ----------------------

def softplus(z: np.ndarray) -> np.ndarray:
    """Numerically stable softplus activation"""
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)

def tanh(z: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation"""
    return np.tanh(z)

def softmax(z: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature"""
    z = z / max(1e-8, temp)
    z = z - np.max(z)
    ez = np.exp(z)
    s = ez.sum()
    return ez / (s + 1e-12)

# ---------------------- energy meter ----------------------

@dataclass
class EnergyMeter:
    """Tracks MAC-level energy consumption for Planck-style accounting"""
    e_mac_j: float = 3e-12       # ~3 pJ per MAC (tunable per device)
    total_j: float = 0.0

    def add_macs(self, nmacs: int) -> None:
        """Add MAC operations to energy total"""
        self.total_j += self.e_mac_j * float(nmacs)

    def reset(self) -> None:
        """Reset energy counter"""
        self.total_j = 0.0

    def get_energy_per_query(self) -> float:
        """Get energy per query in Joules"""
        return self.total_j

# ---------------------- liquid cell (continuous-time) ----------------------

@dataclass
class LiquidCell:
    """Continuous-time liquid state machine with input-dependent time constants"""
    in_dim: int
    hidden_dim: int
    dt: float = 0.02
    tau_min: float = 0.02
    tau_max: float = 2.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(1337))

    # parameters
    W: np.ndarray = field(init=False)   # hidden->hidden
    U: np.ndarray = field(init=False)   # input->hidden
    b: np.ndarray = field(init=False)
    V: np.ndarray = field(init=False)   # input->tau
    c: np.ndarray = field(init=False)
    h: np.ndarray = field(init=False)

    def __post_init__(self):
        """Initialize liquid cell parameters"""
        # Xavier-ish initialization
        self.W = self.rng.normal(0, np.sqrt(2.0/(self.hidden_dim+self.hidden_dim)), 
                                (self.hidden_dim, self.hidden_dim))
        self.U = self.rng.normal(0, np.sqrt(2.0/(self.in_dim+self.hidden_dim)), 
                                (self.hidden_dim, self.in_dim))
        self.b = np.zeros((self.hidden_dim,), dtype=np.float64)
        self.V = self.rng.normal(0, np.sqrt(2.0/(self.in_dim+self.hidden_dim)), 
                                (self.hidden_dim, self.in_dim))
        self.c = self.rng.normal(0, 0.1, (self.hidden_dim,))
        self.h = np.zeros((self.hidden_dim,), dtype=np.float64)

    def reset(self) -> None:
        """Reset liquid state"""
        self.h[:] = 0.0

    def step(self, x: np.ndarray, energy: Optional[EnergyMeter] = None) -> np.ndarray:
        """Single step of liquid dynamics"""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        
        # tau(x) = tau_min + softplus(V x + c) clipped to tau_max
        vx = self.V @ x + self.c
        tau = self.tau_min + softplus(vx)
        tau = np.minimum(tau, self.tau_max)

        # dh/dt = -h/tau + tanh(W h + U x + b)
        Wh = self.W @ self.h
        Ux = self.U @ x
        a = tanh(Wh + Ux + self.b)
        dh = - self.h / np.maximum(tau, 1e-6) + a

        # Euler integrate
        self.h = self.h + self.dt * dh

        if energy is not None:
            # crude MAC count
            nmacs = (self.hidden_dim*self.hidden_dim) + (self.hidden_dim*self.in_dim)
            energy.add_macs(nmacs)
        
        return self.h.copy()

# ---------------------- liquid gating over experts ----------------------

@dataclass
class LiquidGatingNetwork:
    """Liquid gating network for Top-K expert selection"""
    in_dim: int
    hidden_dim: int
    n_experts: int
    top_k: int = 2
    temperature: float = 1.0
    usage_smoothing: float = 0.99     # moving average for usage
    bias_lr: float = 0.01             # tiny bias nudge for load balance
    usage_beta: float = 0.5           # exploration strength for usage bias

    cell: LiquidCell = field(init=False)
    Wg: np.ndarray = field(init=False)  # hidden->experts logits
    bg: np.ndarray = field(init=False)
    usage_ma: np.ndarray = field(init=False)

    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(2025))
    energy: EnergyMeter = field(default_factory=EnergyMeter)

    def __post_init__(self):
        """Initialize gating network"""
        self.cell = LiquidCell(self.in_dim, self.hidden_dim, rng=self.rng)
        self.Wg = self.rng.normal(0, np.sqrt(2.0/(self.hidden_dim+self.n_experts)), 
                                 (self.n_experts, self.hidden_dim))
        self.bg = np.zeros((self.n_experts,), dtype=np.float64)
        self.usage_ma = np.zeros((self.n_experts,), dtype=np.float64)

    def forward(self, x: np.ndarray, attn_gain: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (gates_sparse, topk_idx, full_probs)."""
        h = self.cell.step(x, self.energy)
        logits = (self.Wg @ h) + self.bg
        
        # Apply usage bias to prevent expert collapse
        logits = self._apply_usage_bias(logits)
        
        # attention can sharpen/soften via effective temperature
        temp = max(0.2, self.temperature / max(1e-6, attn_gain))
        probs = softmax(logits, temp=temp)

        k = max(1, min(self.top_k, self.n_experts))
        topk_idx = np.argpartition(probs, -k)[-k:]
        topk_probs = probs[topk_idx]
        
        if topk_probs.sum() <= 1e-12:
            gates = np.ones_like(topk_probs) / len(topk_probs)
        else:
            gates = topk_probs / topk_probs.sum()
        
        out = np.zeros_like(probs)
        out[topk_idx] = gates

        # update usage moving average (for load balance nudging)
        self.usage_ma = self.usage_smoothing*self.usage_ma + (1.0-self.usage_smoothing)*out
        return out, topk_idx, probs

    def _apply_usage_bias(self, logits: np.ndarray) -> np.ndarray:
        """Apply usage-based bias to prevent expert collapse"""
        eps = 1e-6
        target = 1.0 / self.n_experts
        inv_usage = target / (self.usage_ma + eps)
        biased = logits + self.usage_beta * np.log(inv_usage)  # log-space bias
        return biased

    def nudge_for_load_balance(self) -> None:
        """Local bias update pulling usage toward uniform without backprop."""
        if self.n_experts <= 0:
            return
        target = 1.0 / float(self.n_experts)
        delta = target - self.usage_ma
        self.bg += self.bias_lr * delta

    def reset(self) -> None:
        """Reset gating network state"""
        self.cell.reset()
        self.usage_ma[:] = 0.0
        self.energy.reset()

# ---------------------- expert adapter for your NLMS specialists ----------------------

class NLMSExpertAdapter:
    """Wraps a Neuron (with NLMSHead) as an MoE expert."""
    def __init__(self, neuron):
        self.neuron = neuron

    def predict(self, x: np.ndarray) -> float:
        """Get prediction from expert"""
        return float(self.neuron.get_readout(x))

    async def update(self, x: np.ndarray, y_true: float) -> float:
        """Update expert with new data"""
        return float(await self.neuron.update_nlms(x, y_true))

# ---------------------- Liquid-MoE router ----------------------

@dataclass
class LiquidMoERouter:
    """Liquid-MoE router with continuous-time dynamics and Top-K sparse routing"""
    experts: Dict[str, NLMSExpertAdapter]
    in_dim: int
    hidden_dim: int
    top_k: int = 2
    temperature: float = 1.0
    routing_temperature_min: float = 0.2

    gating: LiquidGatingNetwork = field(init=False)
    names: List[str] = field(init=False)
    energy: EnergyMeter = field(default_factory=EnergyMeter)

    def __post_init__(self):
        """Initialize Liquid-MoE router"""
        self.names = list(self.experts.keys())
        self.gating = LiquidGatingNetwork(
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            n_experts=len(self.names),
            top_k=self.top_k,
            temperature=self.temperature,
        )

    def route(self, x: np.ndarray, attn_gain: float = 1.0) -> Dict[str, any]:
        """Route input through selected experts"""
        gates_sparse, topk_idx, probs = self.gating.forward(x, attn_gain=attn_gain)
        chosen = [(self.names[i], float(gates_sparse[i])) for i in topk_idx]
        
        # energy for mixing + K expert readouts (approximate)
        self.energy.add_macs(len(topk_idx) * self.in_dim)  # readout MAC proxy
        
        y = 0.0
        per_expert = {}
        for i in topk_idx:
            name = self.names[int(i)]
            gate = float(gates_sparse[i])
            pred = self.experts[name].predict(x)
            per_expert[name] = {'gate': gate, 'pred': float(pred)}
            y += gate * pred
        
        return {
            'y': float(y),
            'topk': chosen,
            'probs': probs,
            'per_expert': per_expert,
            'energy_j': self.energy.total_j + self.gating.energy.total_j,
            'attn_gain': attn_gain,
            'topk_indices': topk_idx.tolist()
        }

    async def learn(self, x: np.ndarray, y_true: float, attn_gain: float = 1.0) -> Dict[str, any]:
        """Update only the routed experts (streaming, no backprop)."""
        out = self.route(x, attn_gain=attn_gain)
        
        # supervised residual-style targets for selected experts
        for name, info in out['per_expert'].items():
            gate = info['gate']
            # gate-weighted target (simple local rule)
            target = float(y_true)
            await self.experts[name].update(x, target)
        
        # tiny load-balance nudge
        self.gating.nudge_for_load_balance()
        return out

    def reset(self) -> None:
        """Reset router state"""
        self.gating.reset()
        self.energy.reset()

    def get_usage_stats(self) -> Dict[str, float]:
        """Get expert usage statistics"""
        return {
            'usage_ma': self.gating.usage_ma.tolist(),
            'target_usage': 1.0 / len(self.names),
            'usage_std': float(np.std(self.gating.usage_ma)),
            'usage_entropy': float(-np.sum(self.gating.usage_ma * np.log(self.gating.usage_ma + 1e-12)))
        }

    def get_energy_stats(self) -> Dict[str, float]:
        """Get energy consumption statistics"""
        return {
            'total_energy_j': self.energy.total_j + self.gating.energy.total_j,
            'gating_energy_j': self.gating.energy.total_j,
            'expert_energy_j': self.energy.total_j,
            'energy_per_mac_j': self.energy.e_mac_j
        }

# ---------------------- utility functions ----------------------

def create_liquid_moe_from_router(router, features: int = 384, hidden_dim: int = 64, 
                                 top_k: int = 2, temperature: float = 1.0) -> LiquidMoERouter:
    """Create Liquid-MoE router from existing ThalamicConversationRouter"""
    experts = {}
    
    # Convert routing neuron groups to experts
    for group_name, neurons in router.routing_neurons.items():
        if neurons:  # Use first neuron as representative expert
            experts[group_name] = NLMSExpertAdapter(neurons[0])
    
    return LiquidMoERouter(
        experts=experts,
        in_dim=features,
        hidden_dim=hidden_dim,
        top_k=top_k,
        temperature=temperature
    )

def attention_gain_from_text(text: str, attn_system) -> float:
    """Extract attention gain from text using existing attention system"""
    if attn_system is None or not text:
        return 1.0
    
    tokens = text.split()
    if not tokens:
        return 1.0
    
    from .attention import prosody_channels_from_text
    amp, pitch, boundary = prosody_channels_from_text(tokens)
    token_ids = [hash(t.lower()) % 50000 for t in tokens]
    ar = attn_system.compute(token_ids, amp, pitch, boundary)
    
    # turn salience into a gain [0.8..2.0]
    sal = np.asarray(ar["salience"], dtype=np.float64)
    g = 1.0 + 0.5 * float(sal.mean() if sal.size else 0.0)
    return float(np.clip(g, 0.8, 2.0))
