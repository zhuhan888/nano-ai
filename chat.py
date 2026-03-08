from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except Exception:  # pragma: no cover
    Console = None
    Panel = None
    Text = None
    RICH_AVAILABLE = False

from config import DEFAULT_CHECKPOINT_PATH, DEFAULT_META_PATH, ProjectConfig, get_config
from model import NanoTransformer, softmax
from tokenizer import CharTokenizer


CONSOLE = Console() if RICH_AVAILABLE else None


CLI_WHITE = "\033[37m"
CLI_RESET = "\033[0m"


def _cli_white(text: str) -> str:
    return f"{CLI_WHITE}{text}{CLI_RESET}"


def _cli_print(text: str = "", end: str = "\n", flush: bool = False) -> None:
    print(_cli_white(text), end=end, flush=flush)


def _cli_display_width(text: str) -> int:
    return sum(2 if ord(char) > 127 else 1 for char in text)


def _cli_pad_right(text: str, width: int) -> str:
    current = _cli_display_width(text)
    if current >= width:
        return text
    return text + (" " * (width - current))


def _cli_make_box(title: str, lines: list[str]) -> str:
    inner_width = max([_cli_display_width(title), *(_cli_display_width(line) for line in lines)] + [0]) + 2
    top = f"┌─{title}{'─' * max(0, inner_width - _cli_display_width(title))}┐"
    middle = "\n".join(f"│ {_cli_pad_right(line, inner_width)} │" for line in lines)
    bottom = f"└{'─' * (inner_width + 2)}┘"
    return "\n".join([top, middle, bottom])


class _CliStreamClosed:
    pass


def _cli_send_payload(stdin: subprocess.Popen[str].stdin, payload: dict[str, object]) -> None:
    if stdin is None:
        raise RuntimeError("无法写入聊天后端")
    stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stdin.flush()


def _cli_read_events(stdout: subprocess.Popen[str].stdout, output_queue: queue.Queue[object]) -> None:
    if stdout is None:
        output_queue.put(_CliStreamClosed())
        return
    for raw_line in stdout:
        line = raw_line.strip()
        if not line:
            continue
        try:
            output_queue.put(json.loads(line))
        except json.JSONDecodeError:
            output_queue.put(line)
    output_queue.put(_CliStreamClosed())


def _cli_receive(output_queue: queue.Queue[object]) -> object:
    return output_queue.get()


def _cli_wait_until_done(output_queue: queue.Queue[object], child_stdin: subprocess.Popen[str].stdin, auto_exit: bool) -> bool:
    while True:
        message = _cli_receive(output_queue)
        if isinstance(message, _CliStreamClosed):
            return False
        if isinstance(message, str):
            _cli_print(message)
            continue
        event_type = str(message.get("type", ""))
        if event_type == "token":
            sys.stdout.write(_cli_white(str(message.get("text", ""))))
            sys.stdout.flush()
        elif event_type == "done":
            if message.get("empty"):
                sys.stdout.write(_cli_white("请输入一句话后再试。"))
            sys.stdout.write("\n")
            sys.stdout.flush()
            if auto_exit:
                _cli_send_payload(child_stdin, {"command": "exit"})
            return True
        elif event_type == "bye":
            _cli_print("已退出。")
            return False
        elif event_type == "error":
            lines = ["聊天后端失败", str(message.get("message", ""))]
            traceback = str(message.get("traceback", "") or "")
            if traceback:
                lines.append(traceback)
            _cli_print(_cli_make_box(" Error ", lines))
            return False


def _run_embedded_cli() -> None:
    raw_args = sys.argv[1:]
    single_prompt: str | None = None
    pass_args: list[str] = []
    index = 0
    while index < len(raw_args):
        arg = raw_args[index]
        if arg == "--prompt":
            single_prompt = raw_args[index + 1] if index + 1 < len(raw_args) else ""
            index += 2
            continue
        pass_args.append(arg)
        index += 1

    env = os.environ.copy()
    env["NANO_JSCLI_CHILD"] = "1"
    child = subprocess.Popen(
        [sys.executable, __file__, *pass_args, "--js-cli-backend"],
        cwd=Path(__file__).resolve().parent,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )

    output_queue: queue.Queue[object] = queue.Queue()
    thread = threading.Thread(target=_cli_read_events, args=(child.stdout, output_queue), daemon=True)
    thread.start()

    ready = False
    while not ready:
        message = _cli_receive(output_queue)
        if isinstance(message, _CliStreamClosed):
            raise SystemExit(child.wait())
        if isinstance(message, str):
            _cli_print(message)
            continue
        event_type = str(message.get("type", ""))
        if event_type == "ready":
            _cli_print(
                _cli_make_box(
                    " Chat ",
                    [
                        "纳米智能 CLI",
                        "模式 独立问答",
                        f"词表 {message.get('vocab_size', 0)}",
                        f"最大生成 {message.get('max_new_tokens', 0)}",
                        f"采样 temp={message.get('temperature', 0)} top_k={message.get('top_k', 0)} top_p={message.get('top_p', 0)}",
                        "输入 quit / exit 退出",
                    ],
                )
            )
            ready = True
        elif event_type == "error":
            _cli_print(_cli_make_box(" Error ", ["聊天后端失败", str(message.get("message", ""))]))
            raise SystemExit(child.wait())

    if child.stdin is None:
        raise RuntimeError("无法写入聊天后端")

    if single_prompt is not None:
        sys.stdout.write(_cli_white("助手 › "))
        sys.stdout.flush()
        _cli_send_payload(child.stdin, {"prompt": single_prompt})
        if _cli_wait_until_done(output_queue, child.stdin, auto_exit=True):
            while True:
                message = _cli_receive(output_queue)
                if isinstance(message, _CliStreamClosed):
                    break
                if isinstance(message, str):
                    _cli_print(message)
                    continue
                if str(message.get("type", "")) == "bye":
                    _cli_print("已退出。")
                    break
    else:
        while True:
            sys.stdout.write(_cli_white("你 › "))
            sys.stdout.flush()
            user_input = sys.stdin.readline()
            if not user_input:
                _cli_send_payload(child.stdin, {"command": "exit"})
                break
            text = user_input.strip()
            if not text or text.lower() in {"quit", "exit"}:
                _cli_send_payload(child.stdin, {"command": "exit"})
                break
            sys.stdout.write(_cli_white("助手 › "))
            sys.stdout.flush()
            _cli_send_payload(child.stdin, {"prompt": text})
            if not _cli_wait_until_done(output_queue, child.stdin, auto_exit=False):
                break

    raise SystemExit(child.wait())
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="纳米智能命令行聊天")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="模型 checkpoint 路径")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META_PATH, help="checkpoint 元信息路径")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=None, help="采样温度")
    parser.add_argument("--top-k", type=int, default=None, help="top-k 采样")
    parser.add_argument("--top-p", type=float, default=None, help="top-p 采样")
    parser.add_argument("--prompt", type=str, default=None, help="单次提问（供 JS CLI 透传）")
    parser.add_argument("--js-cli-backend", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def launch_js_cli() -> None:
    _run_embedded_cli()


def emit_event(event: dict[str, object]) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def validate_runtime_assets(
    checkpoint_path: Path,
    meta_path: Path,
    meta: dict[str, object],
    tokenizer: CharTokenizer,
    state: np.lib.npyio.NpzFile,
) -> None:
    required_keys = {"config", "tokenizer_path"}
    missing_meta_keys = sorted(key for key in required_keys if key not in meta)
    if missing_meta_keys:
        raise ValueError(f"checkpoint 元信息缺少字段: {', '.join(missing_meta_keys)}")

    token_embedding_key = "model::token_embedding"
    lm_head_key = "model::lm_head.weight"
    position_embedding_key = "model::position_embedding"
    missing_state_keys = [key for key in (token_embedding_key, lm_head_key, position_embedding_key) if key not in state.files]
    if missing_state_keys:
        raise ValueError(f"checkpoint 缺少关键参数: {', '.join(missing_state_keys)}")

    token_embedding = state[token_embedding_key]
    lm_head_weight = state[lm_head_key]
    position_embedding = state[position_embedding_key]
    meta_config = meta["config"]
    meta_model_config = meta_config["model"]

    errors: list[str] = []
    if token_embedding.shape[0] != tokenizer.vocab_size:
        errors.append(
            f"tokenizer 词表大小为 {tokenizer.vocab_size}，但 checkpoint token embedding 词表大小为 {token_embedding.shape[0]}"
        )
    if lm_head_weight.shape[1] != tokenizer.vocab_size:
        errors.append(
            f"tokenizer 词表大小为 {tokenizer.vocab_size}，但 checkpoint lm_head 输出维度为 {lm_head_weight.shape[1]}"
        )
    if int(meta_model_config["vocab_size"]) != tokenizer.vocab_size:
        errors.append(
            f"meta 中 vocab_size={meta_model_config['vocab_size']}，但 tokenizer 词表大小为 {tokenizer.vocab_size}"
        )
    if token_embedding.shape[1] != int(meta_model_config["hidden_size"]):
        errors.append(
            f"checkpoint hidden_size={token_embedding.shape[1]}，但 meta 中 hidden_size={meta_model_config['hidden_size']}"
        )
    if position_embedding.shape[0] != int(meta_model_config["max_seq_len"]):
        errors.append(
            f"checkpoint max_seq_len={position_embedding.shape[0]}，但 meta 中 max_seq_len={meta_model_config['max_seq_len']}"
        )
    if tokenizer.eos_token_id >= tokenizer.vocab_size or tokenizer.sep_token_id >= tokenizer.vocab_size:
        errors.append("tokenizer 特殊 token 索引无效")

    if errors:
        message = [
            "检测到 checkpoint、meta、tokenizer 不一致，无法安全启动聊天。",
            f"checkpoint: {checkpoint_path}",
            f"meta: {meta_path}",
            f"tokenizer: {meta['tokenizer_path']}",
            "请重新运行 `python train.py` 生成一套一致的产物。",
            "详细问题:",
            *[f"- {error}" for error in errors],
        ]
        raise ValueError("\n".join(message))


def load_runtime(args: argparse.Namespace) -> tuple[ProjectConfig, NanoTransformer, CharTokenizer]:
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    base_config = get_config()
    config = ProjectConfig(
        model=replace(base_config.model, **meta["config"]["model"]),
        train=replace(base_config.train, **meta["config"]["train"]),
        data=replace(base_config.data, **meta["config"]["data"]),
        chat=replace(base_config.chat, **meta["config"]["chat"]),
    )
    if args.max_new_tokens is not None:
        config.chat.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        config.chat.temperature = args.temperature
    if args.top_k is not None:
        config.chat.top_k = args.top_k
    if args.top_p is not None:
        config.chat.top_p = args.top_p

    tokenizer_path = Path(meta["tokenizer_path"])
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"未找到 tokenizer 文件: {tokenizer_path}，请先运行 python train.py")
    tokenizer = CharTokenizer.load(tokenizer_path)
    config.model.vocab_size = tokenizer.vocab_size
    state = np.load(args.checkpoint, allow_pickle=False)
    validate_runtime_assets(args.checkpoint, args.meta, meta, tokenizer, state)
    model = NanoTransformer(config.model, rng=np.random.default_rng(config.train.seed))
    model.load_state_dict({key.split("::", 1)[1]: state[key] for key in state.files if key.startswith("model::")})
    return config, model, tokenizer


def sample_next_token(logits: np.ndarray, temperature: float, top_k: int, top_p: float, rng: np.random.Generator) -> int:
    logits = logits.astype(np.float64)
    logits = logits / max(temperature, 1e-5)
    if 0 < top_k < logits.shape[-1]:
        indices = np.argpartition(logits, -top_k)[-top_k:]
        filtered = np.full_like(logits, -1e9)
        filtered[indices] = logits[indices]
        logits = filtered
    probs = softmax(logits, axis=-1)
    if 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)
        cutoff = int(np.searchsorted(cumulative, top_p, side="left"))
        keep_mask = np.zeros_like(sorted_probs, dtype=bool)
        keep_mask[: cutoff + 1] = True
        filtered_probs = np.zeros_like(probs)
        filtered_probs[sorted_indices[keep_mask]] = probs[sorted_indices[keep_mask]]
        probs = filtered_probs
    probs = probs / np.sum(probs)
    return int(rng.choice(np.arange(logits.shape[-1]), p=probs))


def stream_generate_reply(
    prompt: str,
    config: ProjectConfig,
    model: NanoTransformer,
    tokenizer: CharTokenizer,
    rng: np.random.Generator,
):
    prefix_ids = [tokenizer.bos_token_id] + tokenizer.encode_text(prompt) + [tokenizer.sep_token_id]
    prefix_ids = prefix_ids[-config.model.max_seq_len :]
    context_ids = prefix_ids.copy()
    kv_cache = model.init_kv_cache()
    last_logits: np.ndarray | None = None

    for token_id in prefix_ids:
        last_logits, kv_cache = model.forward_with_kv_cache(np.asarray([[token_id]], dtype=np.int32), kv_cache)

    remaining_steps = config.chat.max_new_tokens
    while remaining_steps > 0:
        if last_logits is None:
            break
        cache_len = 0
        if kv_cache and kv_cache[0] is not None:
            cache_len = kv_cache[0].k.shape[2]

        if cache_len < config.model.max_seq_len:
            next_token = sample_next_token(last_logits[0, -1], config.chat.temperature, config.chat.top_k, config.chat.top_p, rng)
        else:
            context = context_ids[-config.model.max_seq_len :]
            last_logits = model.forward(np.asarray([context], dtype=np.int32))
            next_token = sample_next_token(last_logits[0, -1], config.chat.temperature, config.chat.top_k, config.chat.top_p, rng)

        remaining_steps -= 1
        if next_token == tokenizer.eos_token_id:
            break
        if next_token == tokenizer.pad_token_id:
            continue
        context_ids.append(next_token)
        piece = tokenizer.decode_ids([next_token]).strip()
        if piece:
            yield piece
        if cache_len < config.model.max_seq_len:
            last_logits, kv_cache = model.forward_with_kv_cache(np.asarray([[next_token]], dtype=np.int32), kv_cache)


def generate_reply(prompt: str, config: ProjectConfig, model: NanoTransformer, tokenizer: CharTokenizer, rng: np.random.Generator) -> str:
    parts = list(stream_generate_reply(prompt, config, model, tokenizer, rng))
    return "".join(parts).strip() or "我还在学习中，请换个简单问题试试。"


def render_chat_header(config: ProjectConfig, tokenizer: CharTokenizer) -> None:
    if not RICH_AVAILABLE or CONSOLE is None or Panel is None:
        print("纳米智能已启动。当前模式为独立问答：每次输入都不会记住上一轮。输入 quit 退出。")
        return
    body = (
        f"[bold cyan]模式[/bold cyan] 独立问答\n"
        f"[bold cyan]词表[/bold cyan] {tokenizer.vocab_size}\n"
        f"[bold cyan]最大生成[/bold cyan] {config.chat.max_new_tokens}\n"
        f"[bold cyan]采样[/bold cyan] temp={config.chat.temperature} top_k={config.chat.top_k} top_p={config.chat.top_p}\n"
        "[dim]输入 quit / exit 退出[/dim]"
    )
    CONSOLE.print(Panel(body, title="[bold magenta]纳米智能 CLI[/bold magenta]", border_style="bright_blue"))


def main() -> None:
    args = parse_args()
    if not args.js_cli_backend and os.environ.get("NANO_JSCLI_CHILD") != "1":
        launch_js_cli()

    if not args.checkpoint.exists() or not args.meta.exists():
        raise FileNotFoundError("未找到 checkpoint，请先运行 python train.py")
    config, model, tokenizer = load_runtime(args)
    rng = np.random.default_rng(config.train.seed)

    emit_event({
        "type": "ready",
        "vocab_size": tokenizer.vocab_size,
        "max_new_tokens": config.chat.max_new_tokens,
        "temperature": config.chat.temperature,
        "top_k": config.chat.top_k,
        "top_p": config.chat.top_p,
        "max_seq_len": config.model.max_seq_len,
    })

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        payload = json.loads(raw)
        command = payload.get("command")
        if command == "exit":
            emit_event({"type": "bye"})
            break
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            emit_event({"type": "done", "text": "", "empty": True})
            continue
        emit_event({"type": "start", "prompt": prompt})
        parts: list[str] = []
        for piece in stream_generate_reply(prompt, config, model, tokenizer, rng):
            parts.append(piece)
            emit_event({"type": "token", "text": piece})
        full_text = "".join(parts).strip() or "我还在学习中，请换个简单问题试试。"
        emit_event({"type": "done", "text": full_text, "empty": False})


if __name__ == "__main__":
    main()
