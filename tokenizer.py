from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]


@dataclass(slots=True)
class EncodedSample:
    input_ids: list[int]
    target_ids: list[int]


class CharTokenizer:
    def __init__(self, stoi: dict[str, int] | None = None, itos: list[str] | None = None) -> None:
        self.stoi = stoi or {}
        self.itos = itos or []
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.sep_token = "<sep>"
        self.unk_token = "<unk>"
        self.pad_token_id = self.stoi.get(self.pad_token, 0)
        self.bos_token_id = self.stoi.get(self.bos_token, 1)
        self.eos_token_id = self.stoi.get(self.eos_token, 2)
        self.sep_token_id = self.stoi.get(self.sep_token, 3)
        self.unk_token_id = self.stoi.get(self.unk_token, 4)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def build_vocab(self, texts: list[str], vocab_limit: int = 8000) -> None:
        char_counts: dict[str, int] = {}
        for text in texts:
            for char in text:
                char_counts[char] = char_counts.get(char, 0) + 1
        sorted_chars = sorted(char_counts.items(), key=lambda item: (-item[1], item[0]))
        vocab_chars = [char for char, _ in sorted_chars[: max(0, vocab_limit - len(SPECIAL_TOKENS))]]
        self.itos = SPECIAL_TOKENS + vocab_chars
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.pad_token_id = self.stoi[self.pad_token]
        self.bos_token_id = self.stoi[self.bos_token]
        self.eos_token_id = self.stoi[self.eos_token]
        self.sep_token_id = self.stoi[self.sep_token]
        self.unk_token_id = self.stoi[self.unk_token]

    def encode_text(self, text: str) -> list[int]:
        return [self.stoi.get(char, self.unk_token_id) for char in text]

    def decode_ids(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens: list[str] = []
        special = set(SPECIAL_TOKENS)
        for token_id in ids:
            if token_id < 0 or token_id >= len(self.itos):
                continue
            token = self.itos[token_id]
            if skip_special_tokens and token in special:
                continue
            tokens.append(token)
        return "".join(tokens)

    def encode_pair(self, prompt: str, reply: str, max_seq_len: int) -> EncodedSample:
        prompt_ids = self.encode_text(prompt)
        reply_ids = self.encode_text(reply)
        sequence = [self.bos_token_id] + prompt_ids + [self.sep_token_id] + reply_ids + [self.eos_token_id]
        if len(sequence) < 2:
            sequence = [self.bos_token_id, self.eos_token_id]
        if len(sequence) > max_seq_len + 1:
            sequence = sequence[: max_seq_len + 1]
            sequence[-1] = self.eos_token_id
        input_ids = sequence[:-1]
        target_ids = sequence[1:]
        return EncodedSample(input_ids=input_ids, target_ids=target_ids)

    def save(self, path: str | Path) -> None:
        payload = {"itos": self.itos}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        itos = payload["itos"]
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)
