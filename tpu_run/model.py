# JAX skeleton that mirrors Network’s forward (no CUDA deps)
import jax, jax.numpy as jnp
from jax import lax
import optax
from dataclasses import dataclass

def modrelu(z_re, z_im, b):
    r = jnp.sqrt(z_re*z_re + z_im*z_im + 1e-8)
    scale = jnp.maximum(0.0, r + b) / (r + 1e-8)
    return z_re * scale, z_im * scale

@dataclass
class Params:
    w_in: jnp.ndarray     # [d_in, d_h]
    b_mod: jnp.ndarray    # [d_h]
    w_out: jnp.ndarray    # [d_h, 1]

def forward(params: Params, x_re, x_im):
    # x_*: [B, d_in], bf16 okay
    h_re = x_re @ params.w_in     # [B, d_h]
    h_im = x_im @ params.w_in
    h_re, h_im = modrelu(h_re, h_im, params.b_mod)
    # energy-aware magnitude readout (example)
    mag = jnp.sqrt(h_re*h_re + h_im*h_im + 1e-8)
    y = (mag @ params.w_out).squeeze(-1)   # [B]
    return y  # logits

def init_params(key, d_in=384, d_h=512):
    k1, k2, k3 = jax.random.split(key, 3)
    return Params(
        w_in=jax.random.normal(k1, (d_in, d_h), jnp.bfloat16) * 0.02,
        b_mod=jnp.zeros((d_h,), jnp.bfloat16),
        w_out=jax.random.normal(k2, (d_h, 1), jnp.bfloat16) * 0.02,
    )

def pairwise_loss(yA, yB, labelA_prefers):  # label in {0,1}, 1 means A wins
    # logistic Bradley–Terry
    logits = yA - yB
    return optax.sigmoid_binary_cross_entropy(logits, labelA_prefers).mean()
