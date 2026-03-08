from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
DEFAULT_CHECKPOINT_PATH = CHECKPOINT_DIR / "nano_intelligence.npz"
DEFAULT_META_PATH = CHECKPOINT_DIR / "nano_intelligence.meta.json"
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
ENCODED_DATA_CACHE_PATH = DATA_DIR / "encoded_dataset.npz"
GENERATED_DATA_PATH = DATA_DIR / "generated.jsonl"
CRAWLED_DATA_PATH = DATA_DIR / "crawled.jsonl"
MERGED_DATA_PATH = DATA_DIR / "merged.jsonl"


@dataclass(slots=True)
class ModelConfig:
    n_layers: int = 4
    hidden_size: int = 256
    n_heads: int = 4
    ffn_hidden_size: int = 512
    max_seq_len: int = 256
    vocab_size: int = 8000
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 4
    num_epochs: int = 6
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    log_interval: int = 10
    checkpoint_interval: int = 100
    seed: int = 42
    warmup_steps: int = 50
    min_learning_rate: float = 1e-4
    num_workers: int = 0


@dataclass(slots=True)
class DataConfig:
    target_samples: int = 10_000
    max_prompt_chars: int = 96
    max_reply_chars: int = 120
    categories: tuple[str, ...] = (
        "greeting",
        "knowledge",
        "emotion",
        "practical",
        "chitchat",
    )


@dataclass(slots=True)
class ChatConfig:
    max_new_tokens: int = 80
    temperature: float = 0.01
    top_k: int = 0
    top_p: float = 0.01
    stream_delay: float = 0.0


@dataclass(slots=True)
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_runtime_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def get_config() -> ProjectConfig:
    ensure_runtime_dirs()
    return ProjectConfig()
