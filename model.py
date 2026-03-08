from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from config import ModelConfig


F32_HALF = np.float32(0.5)
F32_ONE = np.float32(1.0)
F32_GELU_COEF = np.float32(0.044715)
F32_GELU_SCALE = np.float32(np.sqrt(np.float32(2.0 / np.pi)))
F32_NEG_LARGE = np.float32(-1e9)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    x_sq = x * x
    x_cu = x_sq * x
    return F32_HALF * x * (F32_ONE + np.tanh(F32_GELU_SCALE * (x + F32_GELU_COEF * x_cu)))


def gelu_backward(x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    x_sq = x * x
    x_cu = x_sq * x
    tanh_term = np.tanh(F32_GELU_SCALE * (x + F32_GELU_COEF * x_cu))
    sech2 = F32_ONE - tanh_term * tanh_term
    left = F32_HALF * (F32_ONE + tanh_term)
    right = F32_HALF * x * sech2 * F32_GELU_SCALE * (F32_ONE + np.float32(3.0) * F32_GELU_COEF * x_sq)
    return grad_output * (left + right)


def linear_forward(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weight + bias


@dataclass(slots=True)
class LayerNormCache:
    x: np.ndarray
    x_hat: np.ndarray
    std_inv: np.ndarray
    gamma: np.ndarray


def layer_norm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> tuple[np.ndarray, LayerNormCache]:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    std_inv = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean) * std_inv
    out = gamma * x_hat + beta
    cache = LayerNormCache(x=x, x_hat=x_hat, std_inv=std_inv, gamma=gamma)
    return out, cache


def layer_norm_backward(grad_out: np.ndarray, cache: LayerNormCache) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_hat = cache.x_hat
    gamma = cache.gamma
    x = cache.x
    std_inv = cache.std_inv
    n = x.shape[-1]

    grad_gamma = np.sum(grad_out * x_hat, axis=tuple(range(grad_out.ndim - 1)))
    grad_beta = np.sum(grad_out, axis=tuple(range(grad_out.ndim - 1)))

    grad_xhat = grad_out * gamma
    grad_x = (1.0 / n) * std_inv * (
        n * grad_xhat
        - np.sum(grad_xhat, axis=-1, keepdims=True)
        - x_hat * np.sum(grad_xhat * x_hat, axis=-1, keepdims=True)
    )
    return grad_x, grad_gamma, grad_beta


def cross_entropy_with_grad(logits: np.ndarray, targets: np.ndarray, ignore_index: int) -> tuple[float, np.ndarray]:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    valid_mask = flat_targets != ignore_index
    valid_count = int(np.sum(valid_mask))
    if valid_count == 0:
        return 0.0, np.zeros_like(logits)

    shifted = flat_logits - np.max(flat_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    valid_indices = np.nonzero(valid_mask)[0]
    chosen = probs[valid_indices, flat_targets[valid_mask]]
    loss_value = float(-np.mean(np.log(np.clip(chosen, 1e-12, None))))

    grad = probs
    grad[valid_indices, flat_targets[valid_mask]] -= 1.0
    grad[~valid_mask] = 0.0
    grad = (grad / valid_count).reshape(logits.shape)
    return loss_value, grad


def causal_mask(size: int) -> np.ndarray:
    mask = np.triu(np.ones((size, size), dtype=bool), k=1)
    return mask[None, None, :, :]


@dataclass(slots=True)
class TransformerBlockCache:
    ln1_cache: LayerNormCache
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    attn_probs: np.ndarray
    attn_input: np.ndarray
    qkv_input: np.ndarray
    ln2_out: np.ndarray
    ln2_cache: LayerNormCache
    ffn_hidden_pre: np.ndarray
    ffn_hidden: np.ndarray


@dataclass(slots=True)
class AttentionKVCache:
    k: np.ndarray
    v: np.ndarray


@dataclass(slots=True)
class LayerWeights:
    ln1_gamma: np.ndarray
    ln1_beta: np.ndarray
    attn_qkv_weight: np.ndarray
    attn_qkv_bias: np.ndarray
    attn_out_weight: np.ndarray
    attn_out_bias: np.ndarray
    ln2_gamma: np.ndarray
    ln2_beta: np.ndarray
    ffn_in_weight: np.ndarray
    ffn_in_bias: np.ndarray
    ffn_out_weight: np.ndarray
    ffn_out_bias: np.ndarray


@dataclass(slots=True)
class LayerGrads:
    ln1_gamma: np.ndarray
    ln1_beta: np.ndarray
    attn_qkv_weight: np.ndarray
    attn_qkv_bias: np.ndarray
    attn_out_weight: np.ndarray
    attn_out_bias: np.ndarray
    ln2_gamma: np.ndarray
    ln2_beta: np.ndarray
    ffn_in_weight: np.ndarray
    ffn_in_bias: np.ndarray
    ffn_out_weight: np.ndarray
    ffn_out_bias: np.ndarray


class NanoTransformer:
    def __init__(self, config: ModelConfig, rng: np.random.Generator | None = None) -> None:
        if config.hidden_size % config.n_heads != 0:
            raise ValueError("hidden_size 必须能被 n_heads 整除")
        self.config = config
        self.rng = rng or np.random.default_rng(42)
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}
        self.cache: dict[str, Any] = {}
        self.mask_cache: dict[int, np.ndarray] = {}
        self.layer_weights: list[LayerWeights] = []
        self.layer_grads: list[LayerGrads] = []
        self.ln_f_gamma: np.ndarray | None = None
        self.ln_f_beta: np.ndarray | None = None
        self.lm_head_weight: np.ndarray | None = None
        self.lm_head_bias: np.ndarray | None = None
        self.grad_token_embedding: np.ndarray | None = None
        self.grad_position_embedding: np.ndarray | None = None
        self.grad_ln_f_gamma: np.ndarray | None = None
        self.grad_ln_f_beta: np.ndarray | None = None
        self.grad_lm_head_weight: np.ndarray | None = None
        self.grad_lm_head_bias: np.ndarray | None = None
        self._init_parameters()

    def _scaled_init(self, shape: tuple[int, ...], scale: float) -> np.ndarray:
        return (self.rng.standard_normal(shape).astype(np.float32) * scale).astype(np.float32)

    def _init_parameters(self) -> None:
        cfg = self.config
        scale = 0.02
        self.params["token_embedding"] = self._scaled_init((cfg.vocab_size, cfg.hidden_size), scale)
        self.params["position_embedding"] = self._scaled_init((cfg.max_seq_len, cfg.hidden_size), scale)
        for layer in range(cfg.n_layers):
            prefix = f"layers.{layer}"
            self.params[f"{prefix}.ln1.gamma"] = np.ones((cfg.hidden_size,), dtype=np.float32)
            self.params[f"{prefix}.ln1.beta"] = np.zeros((cfg.hidden_size,), dtype=np.float32)
            self.params[f"{prefix}.attn.qkv.weight"] = self._scaled_init((cfg.hidden_size, cfg.hidden_size * 3), scale / math.sqrt(cfg.hidden_size))
            self.params[f"{prefix}.attn.qkv.bias"] = np.zeros((cfg.hidden_size * 3,), dtype=np.float32)
            self.params[f"{prefix}.attn.out.weight"] = self._scaled_init((cfg.hidden_size, cfg.hidden_size), scale / math.sqrt(cfg.hidden_size))
            self.params[f"{prefix}.attn.out.bias"] = np.zeros((cfg.hidden_size,), dtype=np.float32)
            self.params[f"{prefix}.ln2.gamma"] = np.ones((cfg.hidden_size,), dtype=np.float32)
            self.params[f"{prefix}.ln2.beta"] = np.zeros((cfg.hidden_size,), dtype=np.float32)
            self.params[f"{prefix}.ffn.in.weight"] = self._scaled_init((cfg.hidden_size, cfg.ffn_hidden_size), scale / math.sqrt(cfg.hidden_size))
            self.params[f"{prefix}.ffn.in.bias"] = np.zeros((cfg.ffn_hidden_size,), dtype=np.float32)
            self.params[f"{prefix}.ffn.out.weight"] = self._scaled_init((cfg.ffn_hidden_size, cfg.hidden_size), scale / math.sqrt(cfg.ffn_hidden_size))
            self.params[f"{prefix}.ffn.out.bias"] = np.zeros((cfg.hidden_size,), dtype=np.float32)
        self.params["ln_f.gamma"] = np.ones((cfg.hidden_size,), dtype=np.float32)
        self.params["ln_f.beta"] = np.zeros((cfg.hidden_size,), dtype=np.float32)
        self.params["lm_head.weight"] = self._scaled_init((cfg.hidden_size, cfg.vocab_size), scale / math.sqrt(cfg.hidden_size))
        self.params["lm_head.bias"] = np.zeros((cfg.vocab_size,), dtype=np.float32)
        self._refresh_parameter_views()
        self.zero_grad()

    def _refresh_parameter_views(self) -> None:
        self.layer_weights = []
        for layer in range(self.config.n_layers):
            prefix = self._layer_prefix(layer)
            self.layer_weights.append(
                LayerWeights(
                    ln1_gamma=self.params[f"{prefix}.ln1.gamma"],
                    ln1_beta=self.params[f"{prefix}.ln1.beta"],
                    attn_qkv_weight=self.params[f"{prefix}.attn.qkv.weight"],
                    attn_qkv_bias=self.params[f"{prefix}.attn.qkv.bias"],
                    attn_out_weight=self.params[f"{prefix}.attn.out.weight"],
                    attn_out_bias=self.params[f"{prefix}.attn.out.bias"],
                    ln2_gamma=self.params[f"{prefix}.ln2.gamma"],
                    ln2_beta=self.params[f"{prefix}.ln2.beta"],
                    ffn_in_weight=self.params[f"{prefix}.ffn.in.weight"],
                    ffn_in_bias=self.params[f"{prefix}.ffn.in.bias"],
                    ffn_out_weight=self.params[f"{prefix}.ffn.out.weight"],
                    ffn_out_bias=self.params[f"{prefix}.ffn.out.bias"],
                )
            )
        self.ln_f_gamma = self.params["ln_f.gamma"]
        self.ln_f_beta = self.params["ln_f.beta"]
        self.lm_head_weight = self.params["lm_head.weight"]
        self.lm_head_bias = self.params["lm_head.bias"]

    def _refresh_gradient_views(self) -> None:
        self.layer_grads = []
        for layer in range(self.config.n_layers):
            prefix = self._layer_prefix(layer)
            self.layer_grads.append(
                LayerGrads(
                    ln1_gamma=self.grads[f"{prefix}.ln1.gamma"],
                    ln1_beta=self.grads[f"{prefix}.ln1.beta"],
                    attn_qkv_weight=self.grads[f"{prefix}.attn.qkv.weight"],
                    attn_qkv_bias=self.grads[f"{prefix}.attn.qkv.bias"],
                    attn_out_weight=self.grads[f"{prefix}.attn.out.weight"],
                    attn_out_bias=self.grads[f"{prefix}.attn.out.bias"],
                    ln2_gamma=self.grads[f"{prefix}.ln2.gamma"],
                    ln2_beta=self.grads[f"{prefix}.ln2.beta"],
                    ffn_in_weight=self.grads[f"{prefix}.ffn.in.weight"],
                    ffn_in_bias=self.grads[f"{prefix}.ffn.in.bias"],
                    ffn_out_weight=self.grads[f"{prefix}.ffn.out.weight"],
                    ffn_out_bias=self.grads[f"{prefix}.ffn.out.bias"],
                )
            )
        self.grad_token_embedding = self.grads["token_embedding"]
        self.grad_position_embedding = self.grads["position_embedding"]
        self.grad_ln_f_gamma = self.grads["ln_f.gamma"]
        self.grad_ln_f_beta = self.grads["ln_f.beta"]
        self.grad_lm_head_weight = self.grads["lm_head.weight"]
        self.grad_lm_head_bias = self.grads["lm_head.bias"]

    def zero_grad(self) -> None:
        if not self.grads:
            self.grads = {name: np.zeros_like(param) for name, param in self.params.items()}
            self._refresh_gradient_views()
            return
        for grad in self.grads.values():
            grad.fill(0.0)

    def get_causal_mask(self, seq_len: int) -> np.ndarray:
        mask = self.mask_cache.get(seq_len)
        if mask is None:
            mask = causal_mask(seq_len)
            self.mask_cache[seq_len] = mask
        return mask

    def init_kv_cache(self) -> list[AttentionKVCache | None]:
        return [None for _ in range(self.config.n_layers)]

    def _layer_prefix(self, layer: int) -> str:
        return f"layers.{layer}"

    def forward_with_kv_cache(
        self,
        input_ids: np.ndarray,
        kv_cache: list[AttentionKVCache | None] | None = None,
    ) -> tuple[np.ndarray, list[AttentionKVCache | None]]:
        cfg = self.config
        batch_size, seq_len = input_ids.shape
        if batch_size != 1:
            raise ValueError("KV cache 推理当前只支持 batch_size=1")
        if seq_len != 1:
            raise ValueError("KV cache 推理当前只支持单 token 增量前向")

        if kv_cache is None:
            kv_cache = self.init_kv_cache()
        past_len = 0
        if kv_cache and kv_cache[0] is not None:
            past_len = kv_cache[0].k.shape[2]
        if past_len >= cfg.max_seq_len:
            raise ValueError("KV cache 已达到最大上下文长度")

        position_embeddings = self.params["position_embedding"][past_len : past_len + seq_len][None, :, :]
        hidden = self.params["token_embedding"][input_ids] + position_embeddings
        next_cache: list[AttentionKVCache | None] = []

        for layer in range(cfg.n_layers):
            weights = self.layer_weights[layer]
            residual = hidden
            ln1_out, _ = layer_norm_forward(
                hidden,
                weights.ln1_gamma,
                weights.ln1_beta,
            )
            qkv = linear_forward(
                ln1_out,
                weights.attn_qkv_weight,
                weights.attn_qkv_bias,
            )
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(batch_size, seq_len, cfg.n_heads, cfg.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(batch_size, seq_len, cfg.n_heads, cfg.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(batch_size, seq_len, cfg.n_heads, cfg.head_dim).transpose(0, 2, 1, 3)

            cached = kv_cache[layer]
            if cached is not None:
                full_k = np.concatenate([cached.k, k], axis=2)
                full_v = np.concatenate([cached.v, v], axis=2)
            else:
                full_k = k
                full_v = v

            scores = (q @ full_k.transpose(0, 1, 3, 2)) / math.sqrt(cfg.head_dim)
            attn_probs = softmax(scores, axis=-1).astype(np.float32)
            attn_context = attn_probs @ full_v
            attn_merged = attn_context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, cfg.hidden_size)
            attn_out = linear_forward(
                attn_merged,
                weights.attn_out_weight,
                weights.attn_out_bias,
            )
            hidden = residual + attn_out

            ln2_out, _ = layer_norm_forward(
                hidden,
                weights.ln2_gamma,
                weights.ln2_beta,
            )
            ffn_hidden_pre = linear_forward(
                ln2_out,
                weights.ffn_in_weight,
                weights.ffn_in_bias,
            )
            ffn_hidden = gelu(ffn_hidden_pre)
            ffn_output = linear_forward(
                ffn_hidden,
                weights.ffn_out_weight,
                weights.ffn_out_bias,
            )
            hidden = hidden + ffn_output
            next_cache.append(AttentionKVCache(k=full_k, v=full_v))

        ln_f_out, _ = layer_norm_forward(hidden, self.ln_f_gamma, self.ln_f_beta)
        logits = linear_forward(ln_f_out, self.lm_head_weight, self.lm_head_bias)
        return logits, next_cache

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        cfg = self.config
        batch_size, seq_len = input_ids.shape
        if seq_len > cfg.max_seq_len:
            raise ValueError(f"输入长度 {seq_len} 超过最大长度 {cfg.max_seq_len}")
        token_embeddings = self.params["token_embedding"][input_ids]
        position_embeddings = self.params["position_embedding"][np.arange(seq_len)][None, :, :]
        hidden = token_embeddings + position_embeddings
        mask = self.get_causal_mask(seq_len)
        block_caches: list[TransformerBlockCache] = []

        for layer in range(cfg.n_layers):
            weights = self.layer_weights[layer]
            residual = hidden
            ln1_out, ln1_cache = layer_norm_forward(
                hidden,
                weights.ln1_gamma,
                weights.ln1_beta,
            )
            qkv = linear_forward(
                ln1_out,
                weights.attn_qkv_weight,
                weights.attn_qkv_bias,
            )
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(batch_size, seq_len, cfg.n_heads, cfg.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(batch_size, seq_len, cfg.n_heads, cfg.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(batch_size, seq_len, cfg.n_heads, cfg.head_dim).transpose(0, 2, 1, 3)
            scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(cfg.head_dim)
            scores = scores + mask.astype(scores.dtype) * F32_NEG_LARGE
            attn_probs = softmax(scores, axis=-1).astype(np.float32)
            attn_context = attn_probs @ v
            attn_merged = attn_context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, cfg.hidden_size)
            attn_out = linear_forward(
                attn_merged,
                weights.attn_out_weight,
                weights.attn_out_bias,
            )
            hidden = residual + attn_out

            ffn_input = hidden
            ln2_out, ln2_cache = layer_norm_forward(
                hidden,
                weights.ln2_gamma,
                weights.ln2_beta,
            )
            ffn_hidden_pre = linear_forward(
                ln2_out,
                weights.ffn_in_weight,
                weights.ffn_in_bias,
            )
            ffn_hidden = gelu(ffn_hidden_pre)
            ffn_output = linear_forward(
                ffn_hidden,
                weights.ffn_out_weight,
                weights.ffn_out_bias,
            )
            hidden = ffn_input + ffn_output

            block_caches.append(
                TransformerBlockCache(
                    ln1_cache=ln1_cache,
                    q=q,
                    k=k,
                    v=v,
                    attn_probs=attn_probs,
                    attn_input=attn_merged,
                    qkv_input=ln1_out,
                    ln2_out=ln2_out,
                    ln2_cache=ln2_cache,
                    ffn_hidden_pre=ffn_hidden_pre,
                    ffn_hidden=ffn_hidden,
                )
            )

        ln_f_out, ln_f_cache = layer_norm_forward(hidden, self.ln_f_gamma, self.ln_f_beta)
        logits = linear_forward(ln_f_out, self.lm_head_weight, self.lm_head_bias)
        self.cache = {
            "input_ids": input_ids,
            "token_embeddings": token_embeddings,
            "position_embeddings": position_embeddings,
            "block_caches": block_caches,
            "ln_f_cache": ln_f_cache,
            "ln_f_out": ln_f_out,
            "hidden": hidden,
            "mask": mask,
        }
        return logits

    def backward(self, grad_logits: np.ndarray) -> None:
        cfg = self.config
        self.zero_grad()
        ln_f_out = self.cache["ln_f_out"]
        input_ids = self.cache["input_ids"]
        seq_len = input_ids.shape[1]
        hidden_grad = grad_logits @ self.lm_head_weight.T
        self.grad_lm_head_weight += ln_f_out.reshape(-1, cfg.hidden_size).T @ grad_logits.reshape(-1, cfg.vocab_size)
        self.grad_lm_head_bias += np.sum(grad_logits, axis=(0, 1))

        grad_hidden, grad_gamma, grad_beta = layer_norm_backward(hidden_grad, self.cache["ln_f_cache"])
        self.grad_ln_f_gamma += grad_gamma
        self.grad_ln_f_beta += grad_beta

        for layer in reversed(range(cfg.n_layers)):
            block: TransformerBlockCache = self.cache["block_caches"][layer]
            weights = self.layer_weights[layer]
            grads = self.layer_grads[layer]

            grad_ffn_residual = grad_hidden
            grad_ffn_output = grad_hidden
            grads.ffn_out_weight += block.ffn_hidden.reshape(-1, cfg.ffn_hidden_size).T @ grad_ffn_output.reshape(-1, cfg.hidden_size)
            grads.ffn_out_bias += np.sum(grad_ffn_output, axis=(0, 1))
            grad_ffn_hidden = grad_ffn_output @ weights.ffn_out_weight.T
            grad_ffn_hidden_pre = gelu_backward(block.ffn_hidden_pre, grad_ffn_hidden)
            grads.ffn_in_weight += block.ln2_out.reshape(-1, cfg.hidden_size).T @ grad_ffn_hidden_pre.reshape(-1, cfg.ffn_hidden_size)
            grads.ffn_in_bias += np.sum(grad_ffn_hidden_pre, axis=(0, 1))
            grad_ln2_out = grad_ffn_hidden_pre @ weights.ffn_in_weight.T
            grad_ln2_input, grad_gamma2, grad_beta2 = layer_norm_backward(grad_ln2_out, block.ln2_cache)
            grads.ln2_gamma += grad_gamma2
            grads.ln2_beta += grad_beta2
            grad_after_attn = grad_ffn_residual + grad_ln2_input

            grad_attn_residual = grad_after_attn
            grad_attn_out = grad_after_attn
            grads.attn_out_weight += block.attn_input.reshape(-1, cfg.hidden_size).T @ grad_attn_out.reshape(-1, cfg.hidden_size)
            grads.attn_out_bias += np.sum(grad_attn_out, axis=(0, 1))
            grad_attn_input = grad_attn_out @ weights.attn_out_weight.T
            grad_attn_context = grad_attn_input.reshape(input_ids.shape[0], seq_len, cfg.n_heads, cfg.head_dim).transpose(0, 2, 1, 3)

            grad_attn_probs = grad_attn_context @ block.v.transpose(0, 1, 3, 2)
            grad_v = block.attn_probs.transpose(0, 1, 3, 2) @ grad_attn_context
            dot = np.sum(grad_attn_probs * block.attn_probs, axis=-1, keepdims=True)
            grad_scores = block.attn_probs * (grad_attn_probs - dot)
            grad_scores = grad_scores * (~self.cache["mask"]).astype(grad_scores.dtype)
            grad_scores /= math.sqrt(cfg.head_dim)
            grad_q = grad_scores @ block.k
            grad_k = grad_scores.transpose(0, 1, 3, 2) @ block.q

            grad_q = grad_q.transpose(0, 2, 1, 3).reshape(input_ids.shape[0], seq_len, cfg.hidden_size)
            grad_k = grad_k.transpose(0, 2, 1, 3).reshape(input_ids.shape[0], seq_len, cfg.hidden_size)
            grad_v = grad_v.transpose(0, 2, 1, 3).reshape(input_ids.shape[0], seq_len, cfg.hidden_size)
            grad_qkv = np.concatenate([grad_q, grad_k, grad_v], axis=-1)
            grads.attn_qkv_weight += block.qkv_input.reshape(-1, cfg.hidden_size).T @ grad_qkv.reshape(-1, cfg.hidden_size * 3)
            grads.attn_qkv_bias += np.sum(grad_qkv, axis=(0, 1))
            grad_ln1_out = grad_qkv @ weights.attn_qkv_weight.T
            grad_ln1_input, grad_gamma1, grad_beta1 = layer_norm_backward(grad_ln1_out, block.ln1_cache)
            grads.ln1_gamma += grad_gamma1
            grads.ln1_beta += grad_beta1
            grad_hidden = grad_attn_residual + grad_ln1_input

        grad_token_embeddings = grad_hidden
        self.grad_position_embedding[:seq_len] += np.sum(grad_token_embeddings, axis=0)
        flat_ids = input_ids.reshape(-1)
        flat_grad = grad_token_embeddings.reshape(-1, cfg.hidden_size)
        np.add.at(self.grad_token_embedding, flat_ids, flat_grad)

    def loss(self, input_ids: np.ndarray, target_ids: np.ndarray, ignore_index: int) -> tuple[float, np.ndarray]:
        logits = self.forward(input_ids)
        loss_value, grad_logits = cross_entropy_with_grad(logits, target_ids, ignore_index=ignore_index)
        return loss_value, grad_logits

    def parameter_count(self) -> int:
        return int(sum(param.size for param in self.params.values()))

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self.params.items()}

    def load_state_dict(self, state_dict: dict[str, np.ndarray]) -> None:
        for name in self.params:
            if name not in state_dict:
                raise KeyError(f"缺少参数: {name}")
            self.params[name] = state_dict[name].astype(np.float32)
        self._refresh_parameter_views()
        self.zero_grad()


class AdamOptimizer:
    def __init__(
        self,
        params: dict[str, np.ndarray],
        learning_rate: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.m = {name: np.zeros_like(param) for name, param in params.items()}
        self.v = {name: np.zeros_like(param) for name, param in params.items()}
        self.param_names = tuple(params.keys())

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray], grad_clip: float | None = None) -> float:
        self.step_count += 1
        global_norm_sq = 0.0
        for grad in grads.values():
            global_norm_sq += float(np.sum(grad * grad))
        global_norm = math.sqrt(global_norm_sq)
        scale = 1.0
        if grad_clip and global_norm > grad_clip:
            scale = grad_clip / (global_norm + 1e-12)

        beta1_correction = 1.0 - self.beta1 ** self.step_count
        beta2_correction = 1.0 - self.beta2 ** self.step_count

        for name in self.param_names:
            param = params[name]
            grad = grads[name] * scale
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad * grad)
            m_hat = self.m[name] / beta1_correction
            v_hat = self.v[name] / beta2_correction
            params[name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
        return global_norm

    def state_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "step_count": self.step_count,
            "m": self.m,
            "v": self.v,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.learning_rate = float(state_dict["learning_rate"])
        self.beta1 = float(state_dict["beta1"])
        self.beta2 = float(state_dict["beta2"])
        self.eps = float(state_dict["eps"])
        self.weight_decay = float(state_dict["weight_decay"])
        self.step_count = int(state_dict["step_count"])
        self.m = {name: value.astype(np.float32) for name, value in state_dict["m"].items()}
        self.v = {name: value.astype(np.float32) for name, value in state_dict["v"].items()}
