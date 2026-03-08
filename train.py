from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from config import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_META_PATH,
    ENCODED_DATA_CACHE_PATH,
    GENERATED_DATA_PATH,
    TOKENIZER_PATH,
    ProjectConfig,
    ensure_runtime_dirs,
    get_config,
)
from data_generator import main as generate_data
from model import AdamOptimizer, NanoTransformer
from tokenizer import CharTokenizer

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.table import Table
    RICH_AVAILABLE = True
except Exception:  # pragma: no cover
    Console = None
    Panel = None
    Progress = None
    Table = None
    RICH_AVAILABLE = False


ENCODED_CACHE_FORMAT_VERSION = 2
LENGTH_BUCKET_MULTIPLIER = 32
CONSOLE = Console() if RICH_AVAILABLE else None


CLI_WHITE = "\033[37m"
CLI_RESET = "\033[0m"
CLI_CLEAR_LINE = "\r\033[2K"


def _cli_white(text: str) -> str:
    return f"{CLI_WHITE}{text}{CLI_RESET}"


def _cli_print(text: str = "", end: str = "\n", flush: bool = False) -> None:
    print(_cli_white(text), end=end, flush=flush)


def _cli_flush() -> None:
    sys.stdout.flush()


def _cli_clear_line() -> None:
    sys.stdout.write(CLI_CLEAR_LINE)
    _cli_flush()


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


def _cli_fmt_metric(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not (number == number) or number in (float("inf"), float("-inf")):
        return "--"
    return f"{number:.{digits}f}"


def _cli_fmt_eta(value: object) -> str:
    try:
        seconds = max(0, round(float(value)))
    except (TypeError, ValueError):
        return "--"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _cli_progress_bar(current: int, total: int, width: int) -> str:
    if total <= 0:
        return "░" * width
    ratio = max(0.0, min(1.0, current / total))
    filled = min(width, round(width * ratio))
    return ("█" * filled) + ("░" * (width - filled))


def _cli_terminal_columns(default: int = 120) -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return default


def _render_cli_progress(event: dict[str, object]) -> str:
    columns = _cli_terminal_columns(120)
    bar_width = 18 if columns < 100 else 28
    batch = int(event.get("batch", 0) or 0)
    batches = int(event.get("batches", 0) or 0)
    epoch = int(event.get("epoch", 0) or 0)
    epochs = int(event.get("epochs", 0) or 0)
    seq_len = int(event.get("seq_len", 0) or 0)
    bar = _cli_progress_bar(batch, batches, bar_width)
    return (
        f"epoch {epoch}/{epochs} [{bar}] {batch}/{batches} | "
        f"loss {_cli_fmt_metric(event.get('loss'), 4)} | "
        f"best {_cli_fmt_metric(event.get('best_loss'), 4)} | "
        f"lr {_cli_fmt_metric(event.get('lr'), 6)} | "
        f"len {seq_len} | tok/s {_cli_fmt_metric(event.get('tokens_per_second'), 0)} | "
        f"ETA {_cli_fmt_eta(event.get('eta_seconds'))}"
    )


def _run_embedded_cli() -> None:
    env = os.environ.copy()
    env["NANO_JSCLI_CHILD"] = "1"
    child = subprocess.Popen(
        [sys.executable, __file__, *sys.argv[1:], "--js-cli-backend"],
        cwd=Path(__file__).resolve().parent,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    if child.stdout is None:
        raise RuntimeError("无法读取训练后端输出")

    started = False
    progress_active = False

    for raw_line in child.stdout:
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if progress_active:
                _cli_clear_line()
                sys.stdout.write("\n")
                progress_active = False
            _cli_print(line)
            continue

        if not isinstance(event, dict):
            if progress_active:
                _cli_clear_line()
                sys.stdout.write("\n")
                progress_active = False
            _cli_print(line)
            continue

        event_type = str(event.get("type", ""))
        if event_type == "start":
            started = True
            lines = [
                "纳米智能训练 CLI",
                f"样本 {event.get('samples', 0)}",
                f"参数量 {event.get('parameters', 0)}",
                f"Batch Size {event.get('batch_size', 0)}",
                f"平均序列长度 {_cli_fmt_metric(event.get('avg_seq_len'), 2)}",
                f"编码缓存 {'命中' if event.get('used_encoded_cache') else '重建'}",
                f"Checkpoint {event.get('checkpoint', '')}",
            ]
            _cli_print(_cli_make_box(" Training ", lines))
            continue

        if event_type == "progress":
            sys.stdout.write(f"\r\033[2K{_cli_white(_render_cli_progress(event))}")
            sys.stdout.flush()
            progress_active = True
            continue

        if progress_active:
            _cli_clear_line()
            sys.stdout.write("\n")
            progress_active = False

        if event_type == "checkpoint":
            _cli_print(f"✓ 已保存 checkpoint | epoch {event.get('epoch')} | step {event.get('global_step')} | {event.get('path', '')}")
        elif event_type == "epoch_done":
            _cli_print(f"ℹ epoch {event.get('epoch')}/{event.get('epochs')} 完成，best loss={_cli_fmt_metric(event.get('best_loss'), 4)}")
        elif event_type == "done":
            _cli_print(_cli_make_box(" Done ", ["训练完成", f"最佳 loss: {_cli_fmt_metric(event.get('best_loss'), 4)}", str(event.get('path', ''))]))
        elif event_type == "resume":
            _cli_print(f"↻ 从 epoch={event.get('epoch')}, step={event.get('step')} 恢复训练")
        elif event_type == "info":
            _cli_print(f"ℹ {event.get('message', '')}")
        elif event_type == "error":
            lines = ["训练失败", str(event.get("message", ""))]
            traceback = str(event.get("traceback", "") or "")
            if traceback:
                lines.append(traceback)
            _cli_print(_cli_make_box(" Error ", lines))

    if progress_active:
        _cli_clear_line()
        sys.stdout.write("\n")

    exit_code = child.wait()
    if not started and exit_code != 0:
        _cli_print(f"训练进程退出，code={exit_code}")
    raise SystemExit(exit_code)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练纳米智能模型")
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 恢复训练")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="checkpoint 文件路径")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META_PATH, help="checkpoint 元信息路径")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖默认训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖默认 batch size")
    parser.add_argument("--log-interval", type=int, default=None, help="覆盖默认日志间隔")
    parser.add_argument("--js-cli-backend", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def compute_tokenizer_signature(tokenizer: CharTokenizer) -> str:
    joined = "\u241f".join(tokenizer.itos)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_tokenizer(records: list[dict[str, object]], config: ProjectConfig) -> CharTokenizer:
    tokenizer = CharTokenizer()
    texts: list[str] = []
    for record in records:
        conversations = record["conversations"]
        texts.extend([message["content"] for message in conversations])
    tokenizer.build_vocab(texts, vocab_limit=config.model.vocab_size)
    tokenizer.save(TOKENIZER_PATH)
    return tokenizer


def encode_records(records: list[dict[str, object]], tokenizer: CharTokenizer, max_seq_len: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for record in records:
        conversations = record["conversations"]
        prompt = conversations[0]["content"]
        reply = conversations[1]["content"]
        encoded = tokenizer.encode_pair(prompt, reply, max_seq_len=max_seq_len)
        inputs.append(np.asarray(encoded.input_ids[:max_seq_len], dtype=np.int32))
        targets.append(np.asarray(encoded.target_ids[:max_seq_len], dtype=np.int32))
    return inputs, targets


def _pack_sequences(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([sequence.shape[0] for sequence in sequences], dtype=np.int32)
    if len(sequences) == 0:
        return np.empty((0,), dtype=np.int32), lengths
    packed = np.concatenate(sequences, axis=0).astype(np.int32, copy=False)
    return packed, lengths


def _unpack_sequences(packed: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    sequences: list[np.ndarray] = []
    offset = 0
    for length in lengths.astype(np.int32, copy=False):
        next_offset = offset + int(length)
        sequences.append(packed[offset:next_offset].astype(np.int32, copy=False))
        offset = next_offset
    return sequences


def load_encoded_cache(
    cache_path: Path,
    data_path: Path,
    tokenizer: CharTokenizer,
    max_seq_len: int,
) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    if not cache_path.exists():
        return None
    try:
        cache = np.load(cache_path, allow_pickle=False)
        if int(cache["format_version"][0]) != ENCODED_CACHE_FORMAT_VERSION:
            return None
        if int(cache["max_seq_len"][0]) != max_seq_len:
            return None
        expected_signature = compute_file_signature(data_path)
        if int(cache["data_size"][0]) != expected_signature["size"]:
            return None
        if int(cache["data_mtime_ns"][0]) != expected_signature["mtime_ns"]:
            return None
        cache_tokenizer_signature = str(cache["tokenizer_signature"][0])
        if cache_tokenizer_signature != compute_tokenizer_signature(tokenizer):
            return None
        input_lengths = cache["input_lengths"].astype(np.int32, copy=False)
        target_lengths = cache["target_lengths"].astype(np.int32, copy=False)
        return (
            _unpack_sequences(cache["inputs"].astype(np.int32, copy=False), input_lengths),
            _unpack_sequences(cache["targets"].astype(np.int32, copy=False), target_lengths),
        )
    except Exception:
        return None


def save_encoded_cache(
    cache_path: Path,
    data_path: Path,
    tokenizer: CharTokenizer,
    max_seq_len: int,
    inputs: list[np.ndarray],
    targets: list[np.ndarray],
) -> None:
    signature = compute_file_signature(data_path)
    packed_inputs, input_lengths = _pack_sequences(inputs)
    packed_targets, target_lengths = _pack_sequences(targets)
    np.savez(
        cache_path,
        format_version=np.asarray([ENCODED_CACHE_FORMAT_VERSION], dtype=np.int32),
        inputs=packed_inputs,
        targets=packed_targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        max_seq_len=np.asarray([max_seq_len], dtype=np.int32),
        data_size=np.asarray([signature["size"]], dtype=np.int64),
        data_mtime_ns=np.asarray([signature["mtime_ns"]], dtype=np.int64),
        tokenizer_signature=np.asarray([compute_tokenizer_signature(tokenizer)]),
    )


def prepare_training_dataset(records: list[dict[str, object]], tokenizer: CharTokenizer, max_seq_len: int) -> tuple[list[np.ndarray], list[np.ndarray], bool]:
    cached = load_encoded_cache(ENCODED_DATA_CACHE_PATH, GENERATED_DATA_PATH, tokenizer, max_seq_len)
    if cached is not None:
        return cached[0], cached[1], True
    inputs, targets = encode_records(records, tokenizer, max_seq_len=max_seq_len)
    save_encoded_cache(ENCODED_DATA_CACHE_PATH, GENERATED_DATA_PATH, tokenizer, max_seq_len, inputs, targets)
    return inputs, targets, False


def iterate_minibatches(
    inputs: list[np.ndarray],
    targets: list[np.ndarray],
    batch_size: int,
    pad_id: int,
    rng: np.random.Generator,
):
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    lengths = np.asarray([sequence.shape[0] for sequence in inputs], dtype=np.int32)
    indices = np.arange(len(inputs))
    rng.shuffle(indices)

    bucket_size = min(len(indices), max(batch_size, batch_size * LENGTH_BUCKET_MULTIPLIER))
    ordered_chunks: list[np.ndarray] = []
    for start in range(0, len(indices), bucket_size):
        chunk = indices[start : start + bucket_size]
        chunk_lengths = lengths[chunk]
        order = np.argsort(chunk_lengths, kind="stable")
        ordered_chunks.append(chunk[order])

    if ordered_chunks:
        chunk_order = np.arange(len(ordered_chunks))
        rng.shuffle(chunk_order)
        indices = np.concatenate([ordered_chunks[index] for index in chunk_order])

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch_inputs_list = [inputs[index] for index in batch_indices]
        batch_targets_list = [targets[index] for index in batch_indices]
        batch_max_len = max(sequence.shape[0] for sequence in batch_inputs_list)
        batch_inputs = np.full((len(batch_indices), batch_max_len), pad_id, dtype=np.int32)
        batch_targets = np.full((len(batch_indices), batch_max_len), pad_id, dtype=np.int32)
        for row, (input_ids, target_ids) in enumerate(zip(batch_inputs_list, batch_targets_list, strict=False)):
            seq_len = input_ids.shape[0]
            batch_inputs[row, :seq_len] = input_ids
            batch_targets[row, :seq_len] = target_ids
        yield batch_inputs, batch_targets


def launch_js_cli() -> None:
    _run_embedded_cli()


def _sanitize_event(value: object) -> object:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _sanitize_event(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_event(item) for item in value]
    return value


def emit_event(event: dict[str, object]) -> None:
    print(json.dumps(_sanitize_event(event), ensure_ascii=False, allow_nan=False), flush=True)


def compute_learning_rate(base_lr: float, min_lr: float, step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_lr + (base_lr - min_lr) * cosine


def render_training_summary(
    sample_count: int,
    tokenizer: CharTokenizer,
    model: NanoTransformer,
    config: ProjectConfig,
    avg_seq_len: float,
    used_cache: bool,
) -> None:
    if not RICH_AVAILABLE or CONSOLE is None or Table is None or Panel is None:
        print(json.dumps({
            "samples": sample_count,
            "vocab_size": tokenizer.vocab_size,
            "parameters": model.parameter_count(),
            "epochs": config.train.num_epochs,
            "batch_size": config.train.batch_size,
            "avg_seq_len": round(avg_seq_len, 2),
            "max_seq_len": config.model.max_seq_len,
            "encoded_cache": str(ENCODED_DATA_CACHE_PATH),
            "used_encoded_cache": used_cache,
        }, ensure_ascii=False, indent=2))
        return

    table = Table.grid(expand=True)
    table.add_column(style="bold cyan")
    table.add_column(style="white")
    table.add_row("样本数", str(sample_count))
    table.add_row("词表大小", str(tokenizer.vocab_size))
    table.add_row("参数量", f"{model.parameter_count():,}")
    table.add_row("训练轮数", str(config.train.num_epochs))
    table.add_row("Batch Size", str(config.train.batch_size))
    table.add_row("平均序列长度", f"{avg_seq_len:.2f}")
    table.add_row("最大序列长度", str(config.model.max_seq_len))
    table.add_row("编码缓存", "命中" if used_cache else "重建")
    table.add_row("缓存路径", str(ENCODED_DATA_CACHE_PATH))
    CONSOLE.print(Panel(table, title="[bold green]纳米智能训练[/bold green]", border_style="bright_blue"))


def create_training_progress() -> Progress | None:
    if not RICH_AVAILABLE or Progress is None:
        return None
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("[magenta]loss[/] {task.fields[loss]}"),
        TextColumn("[yellow]lr[/] {task.fields[lr]}"),
        TextColumn("[green]len[/] {task.fields[seq_len]}"),
        TextColumn("[cyan]tok/s[/] {task.fields[tokens]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
        transient=False,
    )


def save_checkpoint(
    checkpoint_path: Path,
    meta_path: Path,
    model: NanoTransformer,
    optimizer: AdamOptimizer,
    config: ProjectConfig,
    tokenizer_path: Path,
    epoch: int,
    step: int,
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for name, value in model.state_dict().items():
        arrays[f"model::{name}"] = value
    optimizer_state = optimizer.state_dict()
    for name, value in optimizer_state["m"].items():
        arrays[f"opt_m::{name}"] = value
    for name, value in optimizer_state["v"].items():
        arrays[f"opt_v::{name}"] = value
    np.savez(checkpoint_path, **arrays)
    meta = {
        "epoch": epoch,
        "step": step,
        "tokenizer_path": str(tokenizer_path),
        "config": config.to_dict(),
        "optimizer": {
            "learning_rate": optimizer_state["learning_rate"],
            "beta1": optimizer_state["beta1"],
            "beta2": optimizer_state["beta2"],
            "eps": optimizer_state["eps"],
            "weight_decay": optimizer_state["weight_decay"],
            "step_count": optimizer_state["step_count"],
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint_metadata(meta_path: Path) -> tuple[ProjectConfig, Path, dict[str, float | int]]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    base_config = get_config()
    config = ProjectConfig(
        model=replace(base_config.model, **meta["config"]["model"]),
        train=replace(base_config.train, **meta["config"]["train"]),
        data=replace(base_config.data, **meta["config"]["data"]),
        chat=replace(base_config.chat, **meta["config"]["chat"]),
    )
    return config, Path(meta["tokenizer_path"]), meta["optimizer"]


def load_checkpoint(checkpoint_path: Path, meta_path: Path, model: NanoTransformer, optimizer: AdamOptimizer) -> tuple[int, int, ProjectConfig, Path]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    state = np.load(checkpoint_path, allow_pickle=False)
    model_state = {key.split("::", 1)[1]: state[key] for key in state.files if key.startswith("model::")}
    optimizer_m = {key.split("::", 1)[1]: state[key] for key in state.files if key.startswith("opt_m::")}
    optimizer_v = {key.split("::", 1)[1]: state[key] for key in state.files if key.startswith("opt_v::")}
    model.load_state_dict(model_state)
    optimizer.load_state_dict(
        {
            **meta["optimizer"],
            "m": optimizer_m,
            "v": optimizer_v,
        }
    )
    cfg_dict = meta["config"]
    config = ProjectConfig(
        model=replace(get_config().model, **cfg_dict["model"]),
        train=replace(get_config().train, **cfg_dict["train"]),
        data=replace(get_config().data, **cfg_dict["data"]),
        chat=replace(get_config().chat, **cfg_dict["chat"]),
    )
    return int(meta["epoch"]), int(meta["step"]), config, Path(meta["tokenizer_path"])


def main() -> None:
    args = parse_args()
    if not args.js_cli_backend and os.environ.get("NANO_JSCLI_CHILD") != "1":
        launch_js_cli()

    ensure_runtime_dirs()
    config = get_config()
    if args.epochs is not None:
        config.train.num_epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.log_interval is not None:
        config.train.log_interval = args.log_interval

    if not GENERATED_DATA_PATH.exists():
        emit_event({"type": "info", "message": "未发现训练数据，开始生成 data/generated.jsonl"})
        generate_data(verbose=False)
        emit_event({"type": "info", "message": "训练数据生成完成"})

    records = load_records(GENERATED_DATA_PATH)
    tokenizer: CharTokenizer
    if args.resume and args.checkpoint.exists() and args.meta.exists():
        config, tokenizer_path, _ = load_checkpoint_metadata(args.meta)
        if args.epochs is not None:
            config.train.num_epochs = args.epochs
        if args.batch_size is not None:
            config.train.batch_size = args.batch_size
        if args.log_interval is not None:
            config.train.log_interval = args.log_interval
        tokenizer = CharTokenizer.load(tokenizer_path)
    else:
        tokenizer = build_tokenizer(records, config)

    config.model.vocab_size = tokenizer.vocab_size
    inputs, targets, used_cache = prepare_training_dataset(records, tokenizer, config.model.max_seq_len)

    rng = np.random.default_rng(config.train.seed)
    model = NanoTransformer(config.model, rng=rng)
    optimizer = AdamOptimizer(
        model.params,
        learning_rate=config.train.learning_rate,
        beta1=config.train.adam_beta1,
        beta2=config.train.adam_beta2,
        eps=config.train.adam_eps,
        weight_decay=config.train.weight_decay,
    )

    start_epoch = 0
    start_step = 0
    if args.resume and args.checkpoint.exists() and args.meta.exists():
        start_epoch, start_step, config, tokenizer_path = load_checkpoint(args.checkpoint, args.meta, model, optimizer)
        tokenizer = CharTokenizer.load(tokenizer_path)
        emit_event({"type": "resume", "epoch": start_epoch, "step": start_step})

    total_steps_per_epoch = math.ceil(len(inputs) / config.train.batch_size)
    total_steps = total_steps_per_epoch * config.train.num_epochs
    global_step = start_step
    best_loss = float("inf")
    avg_seq_len = float(sum(sequence.shape[0] for sequence in inputs) / max(1, len(inputs)))

    emit_event({
        "type": "start",
        "samples": len(inputs),
        "vocab_size": tokenizer.vocab_size,
        "parameters": model.parameter_count(),
        "epochs": config.train.num_epochs,
        "batch_size": config.train.batch_size,
        "avg_seq_len": round(avg_seq_len, 2),
        "max_seq_len": config.model.max_seq_len,
        "used_encoded_cache": used_cache,
        "checkpoint": str(args.checkpoint),
        "meta": str(args.meta),
        "total_steps_per_epoch": total_steps_per_epoch,
        "total_steps": total_steps,
        "checkpoint_interval": config.train.checkpoint_interval,
    })

    training_start_time = time.perf_counter()
    recent_token_count = 0
    recent_window_start = training_start_time

    for epoch in range(start_epoch, config.train.num_epochs):
        emit_event({"type": "epoch_start", "epoch": epoch + 1, "epochs": config.train.num_epochs})
        for batch_index, (batch_inputs, batch_targets) in enumerate(iterate_minibatches(inputs, targets, config.train.batch_size, tokenizer.pad_token_id, rng)):
            step_start_time = time.perf_counter()
            optimizer.learning_rate = compute_learning_rate(
                config.train.learning_rate,
                config.train.min_learning_rate,
                global_step,
                total_steps,
                config.train.warmup_steps,
            )
            loss_value, grad_logits = model.loss(batch_inputs, batch_targets, ignore_index=tokenizer.pad_token_id)
            model.backward(grad_logits)
            grad_norm = optimizer.step(model.params, model.grads, grad_clip=config.train.grad_clip)
            global_step += 1
            best_loss = min(best_loss, loss_value)
            progress_percent = 100.0 * global_step / max(1, total_steps)

            step_elapsed = max(time.perf_counter() - step_start_time, 1e-6)
            step_tokens = int(batch_inputs.size)
            step_token_rate = step_tokens / step_elapsed
            recent_token_count += step_tokens
            now = time.perf_counter()
            recent_elapsed = max(now - recent_window_start, 1e-6)
            smoothed_token_rate = recent_token_count / recent_elapsed
            remaining_steps = max(0, total_steps - global_step)
            avg_step_time = (now - training_start_time) / max(1, global_step - start_step)
            eta_seconds = remaining_steps * avg_step_time

            emit_event({
                "type": "progress",
                "epoch": epoch + 1,
                "epochs": config.train.num_epochs,
                "batch": batch_index + 1,
                "batches": total_steps_per_epoch,
                "global_step": global_step,
                "total_steps": total_steps,
                "loss": float(loss_value),
                "best_loss": float(best_loss),
                "grad_norm": float(grad_norm),
                "lr": float(optimizer.learning_rate),
                "progress": round(progress_percent, 3),
                "seq_len": int(batch_inputs.shape[1]),
                "tokens": step_tokens,
                "tokens_per_second": float(smoothed_token_rate),
                "step_tokens_per_second": float(step_token_rate),
                "eta_seconds": float(eta_seconds),
            })
            recent_token_count = 0
            recent_window_start = now

            if global_step % config.train.checkpoint_interval == 0:
                save_checkpoint(args.checkpoint, args.meta, model, optimizer, config, TOKENIZER_PATH, epoch, global_step)
                emit_event({"type": "checkpoint", "epoch": epoch + 1, "global_step": global_step, "path": str(args.checkpoint)})

        save_checkpoint(args.checkpoint, args.meta, model, optimizer, config, TOKENIZER_PATH, epoch + 1, global_step)
        emit_event({"type": "epoch_done", "epoch": epoch + 1, "epochs": config.train.num_epochs, "global_step": global_step, "best_loss": float(best_loss), "path": str(args.checkpoint)})

    emit_event({"type": "done", "best_loss": (None if best_loss == float("inf") else float(best_loss)), "path": str(args.checkpoint)})


if __name__ == "__main__":
    main()
