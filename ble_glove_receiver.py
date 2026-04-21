"""
Windows 本地手套推理系统
通过 BLE 连接左右手套，接收传感器数据，合并为 22 通道帧后送入 MindSpore CTC 模型推理。
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_PROC_BIND"] = "FALSE"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["GOMP_CPU_AFFINITY"] = "0-3"

import asyncio
import json
import os
import sys
import time
import threading
import signal
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from llm_reformatter import LLMReformatter, create_reformatter

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context

from bleak import BleakClient, BleakScanner, BleakGATTCharacteristic
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text



# ==========================================
# 一、配置
# ==========================================

SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
DEVICE_NAME_LEFT = "Glove_LE"
DEVICE_NAME_RIGHT = "Glove_RI"

SAMPLE_RATE_HZ = 50
FRAME_INTERVAL_MS = 20
INFERENCE_EVERY_N_FRAMES = 64
MAX_BUFFER_FRAMES = 512
MAX_SEQ_LEN = 128
BLANK_THRESHOLD_FRAMES = 20
NUM_CLASSES = 171
BLANK_ID = 0

CKPT_PATH = "gesture_best_performance.ckpt"
LABEL_LIBRARY_PATH = "label_library.json"
SENTENCE_LABELS_PATH = "sentence_labels.json"

FRAME_MATCH_WINDOW_MS = 50

# LLM reformatter configuration
LLM_MODEL_PATH = "models/qwen2-0_5b-instruct-q4_k_m.gguf"
LLM_REFORMAT_TRIGGER_WORDS = 3
LLM_SENTENCE_TIMEOUT_SEC = 5.0
LLM_N_CTX = 128
LLM_N_THREADS = 1  # auto-detect
LLM_N_GPU_LAYERS = 0  # set > 0 if using GPU/NPU acceleration

console = Console()


# ==========================================
# 二、模型定义（与 SentenceFineTuneAug.py 一致）
# ==========================================

class GestureTransformer(nn.Cell):
    def __init__(self, input_dim=22, num_classes=171):
        super(GestureTransformer, self).__init__()
        self.embedding = nn.Dense(input_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=512, dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.classifier = nn.SequentialCell([
            nn.Dense(128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Dense(256, num_classes)
        ])

    def construct(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = ops.reduce_mean(x, 1)
        return self.classifier(x)


class CTCModel(nn.Cell):
    """CTC 模型包装：embedding -> transformer -> fc(blank=0)"""
    def __init__(self, backbone: GestureTransformer):
        super().__init__()
        self.backbone_embedding = backbone.embedding
        self.backbone_transformer = backbone.transformer
        self.fc = nn.Dense(128, NUM_CLASSES)

    def construct(self, x):
        x = self.backbone_embedding(x)
        x = self.backbone_transformer(x)
        return self.fc(x)


# ==========================================
# 三、标签库与模板
# ==========================================

def load_label_resources():
    """加载标签库和句子模板，返回反向映射"""
    with open(LABEL_LIBRARY_PATH, 'r', encoding='utf-8') as f:
        lib_data = json.load(f)

    id_to_word = {v: k for k, v in lib_data.items()}

    templates = []
    active_ids = {BLANK_ID}
    try:
        with open(SENTENCE_LABELS_PATH, 'r', encoding='utf-8') as f:
            sent_data = json.load(f)
        for words in sent_data.values():
            t = [lib_data[w] for w in words if w in lib_data]
            if t and t not in templates:
                templates.append(t)
            for i in t:
                active_ids.add(i)
    except Exception:
        sent_data = {}

    return id_to_word, templates, active_ids, lib_data


# ==========================================
# 四、CTC 解码与置信度
# ==========================================

def softmax_np(x, axis=-1):
    """NumPy softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def ctc_greedy_decode(logits, id_to_word, blank_id=BLANK_ID):
    """
    CTC greedy decode + 置信度计算
    logits: np.ndarray, shape (T, C)
    返回: (words_with_conf, raw_ids, argmax_seq, trailing_blank_run)
        words_with_conf: list of (word_str, confidence_float)
        raw_ids: list of int (非 blank 的词 ID)
        argmax_seq: np.ndarray (T,) 原始 argmax 序列
        trailing_blank_run: int 末尾连续 blank 帧数（用于句子边界检测）
    """
    probs = softmax_np(logits, axis=-1)

    argmax_seq = np.argmax(probs, axis=-1)

    trailing_blank_run = 0
    for item in reversed(argmax_seq):
        if item == blank_id:
            trailing_blank_run += 1
        else:
            break

    raw_ids = []
    last_id = -1
    for item in argmax_seq:
        if item != blank_id and item != last_id:
            raw_ids.append(int(item))
        last_id = item

    words_with_conf = []
    for wid in raw_ids:
        word_str = id_to_word.get(wid, f"<ID:{wid}>")
        indices = np.where(argmax_seq == wid)[0]
        if indices.size > 0:
            conf = float(probs[indices, wid].mean())
        else:
            conf = 0.0
        words_with_conf.append((word_str, conf))

    return words_with_conf, raw_ids, argmax_seq, trailing_blank_run


def detect_sentence_boundary(raw_ids, blank_id=BLANK_ID, threshold=BLANK_THRESHOLD_FRAMES):
    """
    检测句子边界：blank 连续 threshold 帧以上返回 True
    返回: (is_end, blank_run_length)
    """
    blank_run = 0
    for fid in reversed(raw_ids):
        if fid == blank_id:
            blank_run += 1
        else:
            break
    return blank_run >= threshold, blank_run


def template_match(raw_ids, templates, id_to_word):
    """
    模板匹配：找相似度最高的句子模板
    raw_ids: list of int (非 blank 的词 ID)
    templates: list of list of int
    返回: (matched_words, similarity)
    """
    if not raw_ids or not templates:
        return [id_to_word.get(wid, f"<ID:{wid}>") for wid in raw_ids], 0.0

    import difflib
    raw_str = "".join([chr(2000 + i) for i in raw_ids])

    best_template = raw_ids
    max_sim = -1.0
    for t in templates:
        t_str = "".join([chr(2000 + i) for i in t])
        sim = difflib.SequenceMatcher(None, raw_str, t_str).ratio()
        if sim > max_sim:
            max_sim = sim
            best_template = t

    matched_words = [id_to_word.get(wid, f"<ID:{wid}>") for wid in best_template]
    return matched_words, max_sim


# ==========================================
# 五、MindSpore 模型加载与推理
# ==========================================

def load_model():
    """加载 MindSpore 模型（CPU 模式）

    checkpoint 直接保存自 CTCModel，包含：
    - backbone_embedding, backbone_transformer (来自 GestureTransformer)
    - fc (171 类输出，blank=0)
    因此直接加载到 CTCModel 即可。
    """
    context.set_context(mode=context.GRAPH_MODE)
    ms.set_device("CPU")

    backbone = GestureTransformer(num_classes=NUM_CLASSES)
    model = CTCModel(backbone)
    ms.load_checkpoint(CKPT_PATH, net=model)

    model.set_train(False)

    dummy = Tensor(np.zeros((1, MAX_SEQ_LEN, 22), np.float32))
    _ = model(dummy).asnumpy()
    console.print(f"[green]模型加载成功: {CKPT_PATH}[/green]")
    return model


def normalize_feats(feats: np.ndarray, global_mean=None, global_std=None) -> np.ndarray:
    """Z-Score 归一化。如果提供了 global_mean/std，用它；否则用 feats 自身统计。"""
    if global_mean is None or global_std is None:
        mean = np.mean(feats, axis=0)
        std = np.std(feats, axis=0)
    else:
        mean = global_mean
        std = global_std
    return (feats - mean) / (std + 1e-6)


def pad_or_truncate(feats: np.ndarray, target_len: int) -> np.ndarray:
    """填充或截断到固定长度"""
    T = feats.shape[0]
    if T < target_len:
        pad = np.zeros((target_len - T, feats.shape[1]), dtype=np.float32)
        feats = np.vstack([feats, pad])
    elif T > target_len:
        feats = feats[:target_len, :]
    return feats.astype(np.float32)


# ==========================================
# 六、帧对齐与缓冲区管理
# ==========================================

class GloveFrameBuffer:
    """
    双缓冲结构：
    - raw_buffer: 接收线程写入，deque(maxlen=512)
    - inference_buffer: 推理线程独立副本
    """

    def __init__(self, maxlen=MAX_BUFFER_FRAMES):
        self.lock = threading.Lock()
        self.left_buffer = deque(maxlen=maxlen)
        self.right_buffer = deque(maxlen=maxlen)
        self.last_left = None
        self.last_right = None
        self.merged_count = 0

    def push_left(self, frame: np.ndarray, timestamp_ms: float):
        self.left_buffer.append((frame, timestamp_ms))

    def push_right(self, frame: np.ndarray, timestamp_ms: float):
        self.right_buffer.append((frame, timestamp_ms))

    def get_merged_frames(self, window_ms=FRAME_MATCH_WINDOW_MS, num_frames=None, offset=0):
        """
        合并左右手帧并返回。

        默认返回最新的 num_frames 帧（滑动窗口），即每次推理喂入最新的手势序列。
        offset=0 时取最新帧，offset=-128 时取倒数第 128~256 帧。

        Args:
            window_ms: 时间窗口（毫秒），用于左右手帧对齐
            num_frames: 返回帧数，默认 MAX_SEQ_LEN（最新 128 帧）
            offset: 从 buffer 末尾往前的偏移量，默认 0（最新帧）
        """
        if num_frames is None:
            num_frames = MAX_SEQ_LEN

        with self.lock:
            left_list = list(self.left_buffer)
            right_list = list(self.right_buffer)

        if not left_list:
            return np.array([], dtype=np.float32).reshape(0, 22)

        # 取最新的 num_frames 帧（从末尾 offset 开始）
        if offset >= len(left_list):
            left_list = []
        else:
            left_list = left_list[offset:]

        if not left_list:
            return np.array([], dtype=np.float32).reshape(0, 22)

        merged = []
        r_idx = 0
        last_r_frame = right_list[-1][0] if right_list else None
        # 预处理右手时间戳用于二分查找
        right_ts = [r[1] for r in right_list] if right_list else []

        for l_frame, l_ts in left_list:
            # 找最近的右手帧（l_ts - window_ms <= r_ts <= l_ts）
            r_frame = None
            best_r_idx = -1
            best_delta = float('inf')
            for i in range(r_idx, len(right_list)):
                delta = abs(right_ts[i] - l_ts)
                if delta <= window_ms and delta < best_delta:
                    best_delta = delta
                    best_r_idx = i
                    r_frame = right_list[i][0]
                    if right_ts[i] > l_ts:
                        r_idx = i
                        break
                elif right_ts[i] > l_ts + window_ms:
                    r_idx = i
                    break

            if r_frame is None:
                r_frame = last_r_frame if last_r_frame is not None else l_frame

            combined = np.concatenate([l_frame, r_frame])
            if combined.shape[0] == 22:
                merged.append(combined)

        # 取最新的 num_frames 帧
        if len(merged) > num_frames:
            merged = merged[-num_frames:]

        return np.array(merged, dtype=np.float32) if merged else np.array([], dtype=np.float32).reshape(0, 22)

    def frame_count(self):
        with self.lock:
            return len(self.left_buffer)


# ==========================================
# 六-b、词语缓冲与 LLM 句子重组
# ==========================================

class WordBuffer:
    """
    Accumulates recognized words from the inference loop and triggers
    the local LLM to reorder them into natural sentences.
    """

    def __init__(
        self,
        max_words: int = 20,
        trigger_words: int = LLM_REFORMAT_TRIGGER_WORDS,
        reformatter: Optional[LLMReformatter] = None,
    ):
        self.words: list[tuple[str, float]] = []  # [(word, confidence), ...]
        self.max_words = max_words
        self.trigger_words = trigger_words
        self.reformatter = reformatter
        self.pending_reformat: Optional[tuple[list[str], float]] = None
        self._lock = threading.Lock()
        self._reformat_callback: Optional[callable] = None

    def set_callback(self, callback: callable):
        """Set callback(word_list, reformatted_sentence) called after reformat."""
        self._reformat_callback = callback

    def add(self, words_with_conf: list[tuple[str, float]]):
        """
        Add new words (with confidence) to the buffer.
        Deduplicates consecutive repeated words.
        """
        with self._lock:
            for w, c in words_with_conf:
                if not self.words or self.words[-1][0] != w:
                    self.words.append((w, c))
            if len(self.words) > self.max_words:
                self.words = self.words[-self.max_words:]

    def count(self) -> int:
        with self._lock:
            return len(self.words)

    def get_words_str(self) -> str:
        with self._lock:
            return "".join(w for w, _ in self.words)

    def get_word_list(self) -> list[str]:
        with self._lock:
            return [w for w, _ in self.words]

    def should_reformat(self) -> bool:
        return self.count() >= self.trigger_words

    def reformat_sync(self, timeout: float = LLM_SENTENCE_TIMEOUT_SEC) -> Optional[str]:
        """
        Call the LLM synchronously to reorder the accumulated words.
        Returns the reformatted sentence, or None if unavailable.
        """
        if not self.reformatter or not self.reformatter.is_available:
            return None

        words = self.get_word_list()
        if not words:
            return None

        reformatted = self.reformatter.reformat(words, timeout=timeout)
        if reformatted and self._reformat_callback:
            try:
                self._reformat_callback(words, reformatted)
            except Exception:
                pass
        return reformatted

    def reformat_async(self, timeout: float = LLM_SENTENCE_TIMEOUT_SEC):
        """
        Call the LLM asynchronously. Result is delivered via the callback.
        """
        if not self.reformatter or not self.reformatter.is_available:
            return

        words = self.get_word_list()
        if not words:
            return

        self.reformatter.reformat_async(
            words,
            callback=lambda result: self._deliver_result(words, result),
            timeout=timeout,
        )

    def _deliver_result(self, words: list[str], reformatted: Optional[str]):
        if reformatted and self._reformat_callback:
            try:
                self._reformat_callback(words, reformatted)
            except Exception:
                pass

    def clear(self):
        with self._lock:
            self.words = []


# ==========================================
# 七、BLE 连接管理
# ==========================================

async def find_glove_device(name_prefix: str) -> Optional[str]:
    """扫描 BLE 设备，返回匹配名称的地址"""
    console.print(f"[cyan]正在扫描 {name_prefix}...[/cyan]")
    devices = await BleakScanner.discover(timeout=5.0)
    for d in devices:
        if d.name and name_prefix in d.name:
            console.print(f"[green]找到设备: {d.name} ({d.address})[/green]")
            return d.address
    console.print(f"[red]未找到设备: {name_prefix}[/red]")
    return None


async def connect_and_listen(
    address: str,
    buffer: GloveFrameBuffer,
    side: str,
):
    """连接单个手套并监听通知"""
    def parse_glove_data(data: bytearray) -> Optional[tuple]:
        """解析 BLE 通知数据"""
        try:
            text = data.decode('utf-8', errors='replace').strip()
            parts = [p.strip() for p in text.split(',')]
            if len(parts) < 13:
                return None

            frame_no = int(parts[0])
            timestamp_ms = int(parts[1])
            values = [float(parts[i]) for i in range(2, 13)]
            sensor_vals = values[:11]
            return (frame_no, timestamp_ms, np.array(sensor_vals, dtype=np.float32))
        except (ValueError, IndexError) as e:
            return None

    _debug_count = 0

    def notification_handler(_: BleakGATTCharacteristic, data: bytearray):
        nonlocal _debug_count
        raw_text = data.decode('utf-8', errors='replace').strip()
        if _debug_count < 3:
            console.print(f"[dim]DEBUG {side} raw: {raw_text}[/dim]")
        result = parse_glove_data(data)
        if result is None:
            if _debug_count < 3:
                console.print(f"[red]DEBUG {side} parse FAILED[/red]")
            return
        _, timestamp_ms, sensor_vals = result
        if side == "LEFT":
            buffer.push_left(sensor_vals, timestamp_ms)
        else:
            buffer.push_right(sensor_vals, timestamp_ms)
        _debug_count += 1
        if _debug_count % 50 == 0:
            console.print(f"[dim]DEBUG {side} received {_debug_count} frames, buf_L={len(buffer.left_buffer)}, buf_R={len(buffer.right_buffer)}[/dim]")

    reconnect_delay = 2.0
    max_reconnect_delay = 30.0

    while True:
        try:
            async with BleakClient(address) as client:
                console.print(f"[green][OK] 已连接 {side}: {address}[/green]")
                reconnect_delay = 2.0

                await client.start_notify(CHAR_UUID, notification_handler)
                console.print(f"[yellow]{side} 开始接收数据...[/yellow]")

                while client.is_connected:
                    await asyncio.sleep(0.1)

        except Exception as e:
            console.print(f"[red][X] {side} 连接断开: {e}[/red]")
            console.print(f"[yellow]  {reconnect_delay}s 后重连...[/yellow]")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)


# ==========================================
# 八、推理循环
# ==========================================

def inference_loop(
    buffer: GloveFrameBuffer,
    model,
    id_to_word,
    active_ids,
    lib_data,
    stop_event: threading.Event,
    word_buffer: Optional[WordBuffer] = None,
):
    """
    滑动窗口推理：始终取 buffer 中最新的 MAX_SEQ_LEN 帧。
    CTC greedy decode 输出词序列；连续 BLANK_THRESHOLD_FRAMES 个 blank 帧视为句子结束。
    """

    mask = np.full((NUM_CLASSES,), -1e9, dtype=np.float32)
    for idx in active_ids:
        mask[idx] = 0.0

    running_mean = None
    running_std = None
    norm_alpha = 0.1

    last_inference_left_count = 0
    last_sentence_time = time.time()
    sentence_timeout = 5.0

    while not stop_event.is_set():
        time.sleep(0.05)

        current_count = buffer.frame_count()

        if current_count < MAX_SEQ_LEN:
            continue

        if current_count - last_inference_left_count < INFERENCE_EVERY_N_FRAMES:
            if word_buffer and word_buffer.count() > 0 and (time.time() - last_sentence_time) > sentence_timeout:
                original = word_buffer.get_words_str()
                reformatted = word_buffer.reformat_sync(timeout=LLM_SENTENCE_TIMEOUT_SEC)
                if reformatted:
                    print_sentence(word_buffer.words, reformatted)
                    word_buffer.clear()
                    last_sentence_time = time.time()
            continue

        frames = buffer.get_merged_frames(num_frames=MAX_SEQ_LEN, offset=0)
        if frames.shape[0] < 10:
            continue

        n_real = min(frames.shape[0], MAX_SEQ_LEN)
        feats = np.zeros((MAX_SEQ_LEN, 22), dtype=np.float32)
        feats[:n_real] = frames[:n_real]

        buf_mean = np.mean(feats[:n_real], axis=0)
        buf_std = np.std(feats[:n_real], axis=0)
        if buf_std.min() < 1e-4:
            buf_std = buf_std + 1e-3

        if running_mean is None:
            running_mean = buf_mean.copy()
            running_std = buf_std.copy()
        else:
            running_mean = norm_alpha * buf_mean + (1 - norm_alpha) * running_mean
            running_std = norm_alpha * buf_std + (1 - norm_alpha) * running_std

        feats[:n_real] = (feats[:n_real] - running_mean) / (running_std + 1e-6)

        last_inference_left_count = current_count

        diag = current_count % 32 == 0

        batch = Tensor(feats.reshape(1, MAX_SEQ_LEN, 22), ms.float32)

        try:
            logits = model(batch).asnumpy()
            logits = logits[0] + mask
            logits[:, 0] -= 40.0  # raw blank(~0.89) vs best_nonblank(~0.63), gap=0.26; need big penalty

            if diag:
                probs_for_diag = softmax_np(logits, axis=-1)
                console.print(f"[dim]  >> [{current_count}帧] feats=[{feats[:n_real].min():.2f}, {feats[:n_real].max():.2f}], "
                              f"logits=[{logits.min():.2f}, {logits.max():.2f}], mean={logits.mean():.2f}[/dim]")
                top5_idx = np.argsort(probs_for_diag[0])[-5:][::-1]
                top5 = [(id_to_word.get(i, f"ID:{i}"), probs_for_diag[0, i]) for i in top5_idx]
                console.print(f"[dim]     top5: {', '.join(f'{w}({v:.2f})' for w, v in top5)}[/dim]")

            words_with_conf, raw_ids, _, trailing_blank_run = ctc_greedy_decode(logits, id_to_word)
            is_end = trailing_blank_run >= BLANK_THRESHOLD_FRAMES

            if words_with_conf:
                word_strs = [w for w, _ in words_with_conf]
                word_confs = [c for _, c in words_with_conf]
                last_sentence_time = time.time()
                print_realtime(word_strs, word_confs, trailing_blank_run, current_count)

                if word_buffer:
                    word_buffer.add(words_with_conf)

                    if word_buffer.should_reformat():
                        reformatted = word_buffer.reformat_sync(timeout=LLM_SENTENCE_TIMEOUT_SEC)
                        if reformatted:
                            print_sentence(word_buffer.words, reformatted)
                            word_buffer.clear()
                            last_sentence_time = time.time()

        except Exception as e:
            console.print(f"[red]推理错误: {e}[/red]")
            import traceback
            traceback.print_exc()
            time.sleep(1)


def print_realtime(word_strs, word_confs, blank_run, frame_count):
    """打印实时推理结果"""
    colored_words = []
    for w, c in zip(word_strs, word_confs):
        if c < 0.3:
            colored_words.append(f"[red]{w}({c:.0%})[/red]")
        elif c < 0.5:
            colored_words.append(f"[yellow]{w}({c:.0%})[/yellow]")
        else:
            colored_words.append(f"[green]{w}({c:.0%})[/green]")

    if colored_words:
        console.print(
            f"[dim]>> [{frame_count}帧] blank_run={blank_run}[/dim] "
            + " ".join(colored_words)
        )


def print_sentence(accumulated, reformatted=None, similarity=0.0):
    """
    Print a completed sentence.
    If `reformatted` is provided, also shows the LLM-reordered version.
    `accumulated` is a list of (word, confidence) tuples.
    """
    words = [w for w, _ in accumulated]
    confs = [c for _, c in accumulated]
    avg_conf = np.mean(confs) * 100 if confs else 0.0

    if reformatted:
        console.print(
            f"[bold cyan]* [完整句子] 原始顺序: \"{''.join(words)}\" "
            f"→ LLM调整后: \"{reformatted}\" "
            f"[置信度:{avg_conf:.1f}%]"
            f"[/bold cyan]"
        )
    else:
        console.print(
            f"[bold cyan]* [完整句子] \"{''.join(words)}\" "
            f"[置信度:{avg_conf:.1f}%]"
            f"[/bold cyan]"
        )


# ==========================================
# 九、主程序入口
# ==========================================

async def main():
    console.print(Panel.fit(
        "[bold magenta]手套推理系统[/bold magenta]\n"
        "BLE 连接 · 帧对齐 · CTC 推理 · LLM 语序调整",
        border_style="cyan",
    ))

    model = load_model()
    id_to_word, _, active_ids, lib_data = load_label_resources()
    console.print(f"[dim]标签库: {len(lib_data)} 词 | 有效类别: {len(active_ids)}[/dim]")

    buffer = GloveFrameBuffer()

    console.print("[bold yellow]步骤: 初始化本地 LLM...[/bold yellow]")
    reformatter = create_reformatter(
        model_path=LLM_MODEL_PATH,
        n_ctx=LLM_N_CTX,
        n_threads=LLM_N_THREADS,
        n_gpu_layers=LLM_N_GPU_LAYERS,
    )
    word_buffer = WordBuffer(
        max_words=20,
        trigger_words=LLM_REFORMAT_TRIGGER_WORDS,
        reformatter=reformatter,
    )

    console.print("\n[bold yellow]步骤 1: 扫描并连接左手[/bold yellow]")
    addr_left = await find_glove_device(DEVICE_NAME_LEFT)
    if not addr_left:
        await connect_and_listen(addr_left, buffer, "LEFT")
        console.print("[red]未找到左手设备，使用模拟模式[/red]")
        addr_left = None

    await asyncio.sleep(2.0)

    console.print("\n[bold yellow]步骤 2: 扫描并连接右手[/bold yellow]")
    addr_right = await find_glove_device(DEVICE_NAME_RIGHT)
    if not addr_right:
        await connect_and_listen(addr_left, buffer, "RIGHT")
        console.print("[red]未找到右手设备，使用模拟模式[/red]")
        addr_right = None

    stop_event = threading.Event()

    if addr_left or addr_right:
        tasks = []
        if addr_left:
            task_left = asyncio.create_task(connect_and_listen(addr_left, buffer, "LEFT"))
            tasks.append(task_left)
            await asyncio.sleep(0.5)
        if addr_right:
            task_right = asyncio.create_task(connect_and_listen(addr_right, buffer, "RIGHT"))
            tasks.append(task_right)
            await asyncio.sleep(0.5)

        infer_thread = threading.Thread(
            target=inference_loop,
            args=(buffer, model, id_to_word, active_ids, lib_data, stop_event, word_buffer),
            daemon=True,
        )
        infer_thread.start()

        console.print("\n[green]系统运行中，按 Ctrl+C 退出[/green]")

        loop = asyncio.get_running_loop()
        shutdown_event = asyncio.Event()

        def _sig_handler():
            stop_event.set()
            console.print("\n[yellow]收到退出信号，正在关闭...[/yellow]")
            shutdown_event.set()

        try:
            loop.add_signal_handler(signal.SIGINT, _sig_handler)
        except (NotImplementedError, ValueError):
            signal.signal(signal.SIGINT, lambda s, f: _sig_handler())

        gather_coro = asyncio.gather(*tasks, return_exceptions=True)
        shutdown_waiter = asyncio.create_task(shutdown_event.wait())
        done, pending = await asyncio.wait(
            {gather_coro, shutdown_waiter},
            return_when=asyncio.FIRST_COMPLETED
        )
        if gather_coro in pending:
            gather_coro.cancel()
            await asyncio.gather(gather_coro, return_exceptions=True)
    else:
        console.print("[red]无 BLE 设备，请使用 --simulate 模式测试[/red]")
        sys.exit(1)


# ==========================================
# 十、模拟测试模式
# ==========================================

def run_simulation():
    """
    用 CSV 数据模拟 BLE 输入，验证推理脚本正确性
    使用方法: python ble_glove_receiver.py --simulate
    """
    console.print(Panel.fit(
        "[bold yellow]模拟测试模式[/bold yellow]\n使用现有 CSV 数据模拟手套输入",
        border_style="yellow",
    ))

    model = load_model()
    id_to_word, _, active_ids, lib_data = load_label_resources()

    console.print("[bold yellow]初始化本地 LLM...[/bold yellow]")
    reformatter = create_reformatter(
        model_path=LLM_MODEL_PATH,
        n_ctx=LLM_N_CTX,
        n_threads=LLM_N_THREADS,
        n_gpu_layers=LLM_N_GPU_LAYERS,
    )
    if not reformatter.is_available:
        console.print("[yellow]LLM 不可用，跳过语序调整测试[/yellow]")

    sensor_cols = [
        'b1_L', 'b2_L', 'b3_L', 'b4_L', 'b5_L',
        'ax_L', 'ay_L', 'az_L', 'gx_L', 'gy_L', 'gz_L',
        'b1_R', 'b2_R', 'b3_R', 'b4_R', 'b5_R',
        'ax_R', 'ay_R', 'az_R', 'gx_R', 'gy_R', 'gz_R',
    ]

    import glob
    csv_files = glob.glob("rawData/*.csv")
    if not csv_files:
        console.print("[red]未找到 rawData/*.csv 文件[/red]")
        return

    csv_files.sort()

    mask = np.full((NUM_CLASSES,), -1e9, dtype=np.float32)
    for idx in active_ids:
        mask[idx] = 0.0

    total_correct = 0
    total_samples = 0

    table = Table(title="模拟测试结果", show_header=True, header_style="bold magenta")
    table.add_column("文件", style="cyan")
    table.add_column("真实句子", style="green")
    table.add_column("识别结果", style="yellow")
    table.add_column("LLM调整", style="magenta")
    table.add_column("置信度", style="dim")
    table.add_column("状态", justify="center")

    for csv_path in csv_files:
        try:
            fname = os.path.basename(csv_path)
            try:
                df = pd.read_csv(csv_path, encoding='utf-8-sig')
            except Exception:
                try:
                    df = pd.read_csv(csv_path, encoding='gbk')
                except Exception:
                    df = pd.read_csv(csv_path, encoding='latin1')
            feats = df[sensor_cols].values.astype(np.float32)
            feats = normalize_feats(feats)
            feats = pad_or_truncate(feats, MAX_SEQ_LEN)
            batch = Tensor(feats.reshape(1, MAX_SEQ_LEN, 22), ms.float32)

            logits = model(batch).asnumpy()
            logits = logits[0] + mask
            logits[:, 0] -= 40.0

            words_with_conf, raw_ids, _, _ = ctc_greedy_decode(logits, id_to_word)
            recognized = "".join([w for w, _ in words_with_conf])
            avg_conf = np.mean([c for _, c in words_with_conf]) if words_with_conf else 0.0

            reformatted = None
            if reformatter.is_available and len(words_with_conf) >= 2:
                words_list = [w for w, _ in words_with_conf]
                reformatted = reformatter.reformat(words_list, timeout=5.0)
                if reformatted:
                    reformatted_str = reformatted
                else:
                    reformatted_str = "(超时)"
            else:
                reformatted_str = "(跳过)" if reformatter.is_available else "(LLM未加载)"

            expected_words = []
            try:
                sent_json_path = SENTENCE_LABELS_PATH
                with open(sent_json_path, 'r', encoding='utf-8') as f:
                    sent_data = json.load(f)
                if fname in sent_data:
                    expected_words = sent_data[fname]
            except Exception:
                pass

            expected = "".join(expected_words) if expected_words else "(未知)"

            is_correct = recognized == expected if expected_words else None
            if is_correct is True:
                total_correct += 1
                status = "[green]OK[/green]"
            elif is_correct is False:
                status = "[red]FAIL[/red]"
            else:
                status = "[dim]?[/dim]"

            total_samples += 1
            table.add_row(
                fname,
                expected,
                recognized,
                reformatted_str,
                f"{avg_conf:.1%}",
                status,
            )

        except Exception as e:
            table.add_row(fname, "(错误)", str(e), "-", "-", "[red]FAIL[/red]")
            total_samples += 1

    console.print(table)
    if total_samples > 0 and total_correct > 0:
        console.print(f"\n[bold]准确率: {total_correct}/{total_samples} = {total_correct/total_samples*100:.1f}%[/bold]")
    elif total_samples > 0:
        console.print(f"\n[bold]完成: {total_samples} 样本已测试[/bold]\n")


# ==========================================
# 入口
# ==========================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--simulate":
        run_simulation()
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            console.print("\n[yellow]用户退出[/yellow]")
