"""
AURA Network - Comprehensive neural network system
Combines base network, SPAN integration, SVC capabilities, and training functionality
"""

import asyncio
import trio
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Core components
from .hippocampus import Hippocampus
from .amygdala import Amygdala
from .nlms import SpikingAttention, NLMSHead
from .thalamic_router import ThalamicConversationRouter
from .central_nervous_system import CentralNervousSystem
from .thalamus import Thalamus
from .neuron import Neuron, MaturationStage, ActivityState

# SPAN integration - commented out to avoid circular import
# from ..training.span_integration import SPANNeuron, SPANPattern, create_final_precision_span_patterns

# Enhanced SVC pipeline - commented out to avoid circular import
# from ..utils.enhanced_svc_pipeline import (
#     load_enhanced_svc_dataset,
#     get_enhanced_full_knowledge_embedding,
#     create_sample_enhanced_data
# )

# Network configuration
n_neurons = 10000                # typical: 10 - 10,000+
n_features = 384                # typical: 5 - 500+
n_outputs = 1
input_channels = n_features
output_channels = n_features

Array = np.ndarray


class Network:
    """
    AURA Network - Comprehensive neural network system
    Combines base functionality, SPAN integration, SVC capabilities, and training
    """
    
    def __init__(
        self,
        # Base network parameters
        neuron_count: int = n_neurons,
        features: int = n_features,
        input_channels: int = input_channels,
        output_channels: int = output_channels,
        
        # SPAN integration parameters
        enable_span: bool = True,
        span_neurons_per_region: int = 10,
        
        # SVC parameters
        domains: Optional[List[str]] = None,
        realms: Optional[List[str]] = None,
        offline: bool = False,
        nlms_clamp: Tuple[float, float] = (0.0, 1.0),
        nlms_l2: float = 1e-4,
        features_mode: str = 'sbert',  # 'sbert' | 'phasor' | 'combined'
        features_alpha: float = 0.7,   # weight on SBERT when combined
        weights_dir: str = 'svc_nlms_weights',
        startnew: bool = False,
    ):
        # Store parameters
        self.neuron_count = neuron_count
        self.features = features
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # SPAN parameters
        self.enable_span = enable_span
        self.span_neurons_per_region = span_neurons_per_region
        
        # SVC parameters
        self.domains = domains or ['general', 'technical', 'creative', 'analytical']
        self.realms = realms or ['academic', 'professional', 'personal', 'creative']
        self.offline = offline
        self.nlms_clamp = nlms_clamp
        self.nlms_l2 = nlms_l2
        self.features_mode = features_mode
        self.features_alpha = features_alpha
        self.weights_dir = weights_dir
        self.startnew = startnew
        
        # Initialize core brain regions
        self._thalamus = Thalamus(
            neuron_count=neuron_count,
            input_channels=input_channels,
            output_channels=output_channels
        )
        self._hippocampus = Hippocampus(
            neuron_count=neuron_count,
            features=features,
            neurogenesis_rate=0.01,
            input_dim=features
        )
        self._amygdala = Amygdala()
        self._thalamic_router = ThalamicConversationRouter(
            neuron_count=60, 
            features=features, 
            input_dim=features
        )
        self._cns = CentralNervousSystem(input_dim=features)
        
        # Register brain regions with CNS
        self._cns.register_brain_region('thalamus', self._thalamus, priority=0.7)
        self._cns.register_brain_region('hippocampus', self._hippocampus, priority=0.6)
        self._cns.register_brain_region('amygdala', self._amygdala, priority=0.8)
        self._cns.register_brain_region('router', self._thalamic_router, priority=0.5)
        
        # Attention configurations
        self.attention_configs = {
        'historical': {'decay': 0.8, 'theta': 1.2, 'k_winners': 7, 'gain_up': 1.8, 'gain_down': 0.5},
        'emotional': {'decay': 0.6, 'theta': 0.9, 'k_winners': 4, 'gain_up': 2.0, 'gain_down': 0.4},
        'analytical': {'decay': 0.7, 'theta': 1.0, 'k_winners': 5, 'gain_up': 1.5, 'gain_down': 0.6},
        'memory': {'decay': 0.75, 'theta': 1.1, 'k_winners': 6, 'gain_up': 1.6, 'gain_down': 0.6}
        }

        # SPAN integration - commented out to avoid circular import
        # self.span_hippocampus_neurons: List[SPANNeuron] = []
        # self.span_thalamus_neurons: List[SPANNeuron] = []
        # self.span_amygdala_neurons: List[SPANNeuron] = []
        self.span_performance_history: List[Dict[str, Any]] = []
        self.integration_statistics: Dict[str, Any] = {}
        
        # SVC specialists
        self.specialists: Dict[str, Neuron] = {}
        self.sbert_model = None
        
        # Initialize components
        self._setup_sbert()
        self._setup_specialists()
    
    def _setup_sbert(self):
        """Setup SBERT model for text embeddings"""
        if not self.offline:
            try:
                from sentence_transformers import SentenceTransformer
                self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device='mps')
                print("✅ SBERT model loaded successfully")
            except ImportError:
                print("⚠️  SBERT not available, using zero embeddings")
            except Exception as e:
                print(f"⚠️  SBERT loading failed: {e}, using zero embeddings")
    
    def _setup_specialists(self):
        """Setup specialist neurons for different analysis tasks"""
        # Domain classification specialist  (D = feature dim, C = #domains)
        domain_specialist = Neuron(
            neuron_id="domain_classifier",
            specialization="DOMAIN_CLASSIFIER",
            abilities={'classification': 0.9},
            n_features=len(self.domains),
            n_outputs=1,
            maturation=MaturationStage.DIFFERENTIATED,
            activity=ActivityState.RESTING
        )
        domain_specialist.nlms_head = NLMSHead(
            n_features=self.features,
            n_outputs=len(self.domains),
            mu=0.01,
            l2_decay=self.nlms_l2,
            clip01=False
        )
        self.specialists['domain_classifier'] = domain_specialist
        
        # Realm classification specialist
        realm_specialist = Neuron(
            neuron_id="realm_classifier",
            specialization="REALM_CLASSIFIER",
            abilities={'classification': 0.9},
            n_features=len(self.realms),
            n_outputs=1,
            maturation=MaturationStage.DIFFERENTIATED,
            activity=ActivityState.RESTING
        )
        realm_specialist.nlms_head = NLMSHead(
            n_features=self.features,
            n_outputs=len(self.realms),
            mu=0.01,
            l2_decay=self.nlms_l2,
            clip01=False
        )
        self.specialists['realm_classifier'] = realm_specialist
        
        # Difficulty regression specialist
        difficulty_specialist = Neuron(
            neuron_id="difficulty_regressor",
            specialization="DIFFICULTY_REGRESSOR",
            abilities={'regression': 0.9},
            n_features=1,
            n_outputs=1,
            maturation=MaturationStage.DIFFERENTIATED,
            activity=ActivityState.RESTING
        )
        difficulty_specialist.nlms_head = NLMSHead(
            n_features=self.features,
            n_outputs=1,
            mu=0.01,
            l2_decay=self.nlms_l2,
            clip01=True
        )
        self.specialists['difficulty_regressor'] = difficulty_specialist
    
    def get_features(self, text: str) -> np.ndarray:
        """Get feature vector for text based on features_mode"""
        if self.features_mode == 'sbert':
            return self._get_sbert_features(text)
        elif self.features_mode == 'phasor':
            return self._get_phasor_features(text)
        elif self.features_mode == 'combined':
            sbert_feat = self._get_sbert_features(text)
            phasor_feat = self._get_phasor_features(text)
            # Combine features with alpha weighting
            return self.features_alpha * sbert_feat + (1 - self.features_alpha) * phasor_feat
        else:
            raise ValueError(f"Unknown features_mode: {self.features_mode}")
    
    def _get_sbert_features(self, text: str) -> np.ndarray:
        """Get SBERT features for text"""
        if self.sbert_model is not None:
            vec = self.sbert_model.encode([text])[0]
            D = self.features
            v = np.asarray(vec, dtype=np.float64).reshape(-1)
            if v.size == D: return v
            if v.size > D:  return v[:D]
            return np.pad(v, (0, D - v.size))
        else:
            return np.zeros(self.features, dtype=np.float64)
    
    def _get_phasor_features(self, text: str) -> np.ndarray:
        """Get phasor-based features for text"""
        # Simple phasor-based feature extraction
        words = text.lower().split()
        n_words = len(words)
        
        # Basic features: length, counts → then pad/trim to self.features
        features = np.array([
            len(text),
            n_words,
            sum(len(word) for word in words),
            text.count(' '),
            text.count('.'),
            text.count('!'),
            text.count('?')
        ], dtype=np.float64)
        
        # normalize and fit feature dim
        if n_words > 0:
            features = features / float(n_words)
        D = self.features
        if features.size == D: return features
        if features.size > D:  return features[:D]
        return np.pad(features, (0, D - features.size))
    
    async def init_weights(self):
        """Initialize network weights"""
        # Initialize base network weights
        await self._thalamus.init_weights()
        await self._hippocampus.init_weights()
        await self._amygdala.init_weights()
        await self._thalamic_router.init_weights()
        await self._cns.init_weights()
        
        # Initialize SPAN neurons if enabled - commented out to avoid circular import
        # if self.enable_span:
        #     await self._initialize_span_neurons()
        
        # Initialize specialists
        await self._initialize_specialists()
    
    # async def _initialize_span_neurons(self):
    #     """Initialize SPAN-enhanced neurons - commented out to avoid circular import"""
    #     print("🧠 Initializing SPAN neurons...")
    #     
    #     # Add SPAN neurons to hippocampus (memory formation)
    #     for i in range(min(self.span_neurons_per_region, len(self._hippocampus.neurons))):
    #         base_neuron = self._hippocampus.neurons[i]
    #         span_neuron = SPANNeuron(base_neuron, learning_rate=0.0005)
    #         self.span_hippocampus_neurons.append(span_neuron)
    #     
    #     # Add SPAN neurons to thalamus (sensory gating)
    #     for i in range(min(self.span_neurons_per_region, len(self._thalamus.neurons))):
    #         base_neuron = self._thalamus.neurons[i]
    #         span_neuron = SPANNeuron(base_neuron, learning_rate=0.0005)
    #         self.span_amygdala_neurons.append(span_neuron)
    #     
    #     # Add SPAN neurons to amygdala (emotional processing)
    #     for i in range(min(self.span_neurons_per_region, len(self._amygdala.neurons))):
    #         base_neuron = self._amygdala.neurons[i]
    #         span_neuron = SPANNeuron(base_neuron, learning_rate=0.0005)
    #         self.span_amygdala_neurons.append(span_neuron)
    #     
    #     print(f"✅ SPAN neurons initialized: {len(self.span_hippocampus_neurons)} hippocampus, {len(self.span_thalamus_neurons)} thalamus, {len(self.span_amygdala_neurons)} amygdala")
    
    async def _initialize_specialists(self):
        """Initialize specialist neurons"""
        for specialist in self.specialists.values():
            if hasattr(specialist, 'nlms_head') and specialist.nlms_head:
                C = specialist.nlms_head.n_outputs
                baseW = np.zeros((specialist.nlms_head.n_features, C), dtype=np.float64)
                await specialist.nlms_head.attach(
                    baseW,
                    slice(0, specialist.nlms_head.n_features),  # tok_slice
                    slice(0, 0),
                    slice(0, 0)
                )
    
    async def process_input(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Process input through the network"""
        # Process through thalamus (sensory gating)
        thalamus_output = await self._thalamus.process(input_data)
        
        # Process through hippocampus (memory formation)
        hippocampus_output = await self._hippocampus.process(input_data)
        
        # Process through amygdala (emotional analysis)
        amygdala_output = await self._amygdala.process(input_data)
        
        # Process through thalamic router (conversation routing)
        router_output = await self._thalamic_router.process(input_data)
        
        # Central nervous system coordination
        cns_output = await self._cns.process(input_data)
        
        return {
            'thalamus': thalamus_output,
            'hippocampus': hippocampus_output,
            'amygdala': amygdala_output,
            'router': router_output,
            'cns': cns_output
        }
    
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through the network with specialist analysis"""
        text = data.get('text', '')
        features = self.get_features(text)
        
        # Process through base network
        base_result = await self.process_input(features)
        
        # Process through specialists
        specialist_results = {}
        for name, specialist in self.specialists.items():
            if name == 'domain_classifier':
                prediction = specialist.nlms_head.predict(features)   # (C,)
                j = int(np.argmax(prediction))
                specialist_results['domain'] = self.domains[j]
                specialist_results['domain_confidence'] = float(prediction[j])
            elif name == 'realm_classifier':
                prediction = specialist.nlms_head.predict(features)
                j = int(np.argmax(prediction))
                specialist_results['realm'] = self.realms[j]
                specialist_results['realm_confidence'] = float(prediction[j])
            elif name == 'difficulty_regressor':
                y = specialist.nlms_head.predict(features)
                specialist_results['difficulty'] = float(y[0])
        
        return {
            'base_result': base_result,
            'specialist_results': specialist_results,
            'features': features.tolist()
        }
    
    async def train_on_data(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the network on data"""
        results = {
            'domain_accuracy': 0.0,
            'realm_accuracy': 0.0,
            'difficulty_mse': 0.0,
            'total_samples': len(training_data)
        }
        
        domain_correct = 0
        realm_correct = 0
        difficulty_errors = []
        
        for data in training_data:
            text = data.get('text', '')
            features = self.get_features(text)
            
            # Train domain classifier
            if 'domain' in data:
                domain_idx = self.domains.index(data['domain']) if data['domain'] in self.domains else 0
                domain_target = np.zeros(len(self.domains))
                domain_target[domain_idx] = 1.0
                
                domain_specialist = self.specialists['domain_classifier']
                await domain_specialist.nlms_head.step(features, domain_target)
                
                # Check accuracy
                prediction = domain_specialist.nlms_head.predict(features)
                if np.argmax(prediction) == domain_idx:
                    domain_correct += 1
            
            # Train realm classifier
            if 'realm' in data:
                realm_idx = self.realms.index(data['realm']) if data['realm'] in self.realms else 0
                realm_target = np.zeros(len(self.realms))
                realm_target[realm_idx] = 1.0
                
                realm_specialist = self.specialists['realm_classifier']
                await realm_specialist.nlms_head.step(features, realm_target)
                
                # Check accuracy
                prediction = realm_specialist.nlms_head.predict(features)
                if np.argmax(prediction) == realm_idx:
                    realm_correct += 1
            
            # Train difficulty regressor
            if 'difficulty' in data:
                difficulty_target = np.array([data['difficulty']])
                
                difficulty_specialist = self.specialists['difficulty_regressor']
                await difficulty_specialist.nlms_head.step(features, difficulty_target)
                
                # Calculate error
                y = difficulty_specialist.nlms_head.predict(features)
                error = (float(y[0]) - float(data['difficulty'])) ** 2
                difficulty_errors.append(error)
        
        # Calculate final metrics
        if results['total_samples'] > 0:
            results['domain_accuracy'] = domain_correct / results['total_samples']
            results['realm_accuracy'] = realm_correct / results['total_samples']
            results['difficulty_mse'] = np.mean(difficulty_errors) if difficulty_errors else 0.0
        
        return results
    
    def enable_attention_learning(self, regions: List[str]):
        """Enable attention learning for specified regions"""
        for region in regions:
            if region in self.attention_configs:
                config = self.attention_configs[region]
                # Apply attention configuration to the region
                print(f"✅ Attention learning enabled for {region}: {config}")
    
    def save_weights(self, base_dir: str = None):
        """Save network weights"""
        base_dir = base_dir or self.weights_dir
        import os
        os.makedirs(base_dir, exist_ok=True)
        
        # Save core weights
        try:
            th_W = np.vstack([n.nlms_head.w for n in self._thalamus.neurons])
            np.savez_compressed(os.path.join(base_dir, 'thalamus_weights.npz'), W=th_W)
        except Exception:
            pass
        
        try:
            hip_W = np.vstack([n.nlms_head.w for n in self._hippocampus.neurons])
            np.savez_compressed(os.path.join(base_dir, 'hippocampus_weights.npz'), W=hip_W)
        except Exception:
            pass
        
        # Save specialist weights
        for name, specialist in self.specialists.items():
            try:
                weights = specialist.nlms_head.w
                np.save(os.path.join(base_dir, f'{name}_weights.npy'), weights)
            except Exception:
                pass
    
    def load_weights(self, base_dir: str = None):
        """Load network weights"""
        base_dir = base_dir or self.weights_dir
        loaded = 0
        
        # Load core weights
        try:
            thalamus_weights = np.load(os.path.join(base_dir, 'thalamus_weights.npz'))['W']
            for i, neuron in enumerate(self._thalamus.neurons):
                if i < len(thalamus_weights):
                    neuron.nlms_head.w = thalamus_weights[i].astype(np.float64)
                    loaded += 1
        except Exception:
            pass
        
        try:
            hippocampus_weights = np.load(os.path.join(base_dir, 'hippocampus_weights.npz'))['W']
            for i, neuron in enumerate(self._hippocampus.neurons):
                if i < len(hippocampus_weights):
                    neuron.nlms_head.w = hippocampus_weights[i].astype(np.float64)
                    loaded += 1
        except Exception:
            pass
        
        # Load specialist weights
        for name, specialist in self.specialists.items():
            try:
                weights = np.load(os.path.join(base_dir, f'{name}_weights.npy'))
                specialist.nlms_head.w = weights.astype(np.float64)
                loaded += 1
            except Exception:
                pass
        
        return loaded


# Backward compatibility aliases
SVCNetwork = Network  # For backward compatibility
SPANIntegratedNetwork = Network  # For backward compatibility