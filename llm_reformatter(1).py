# -*- coding: utf-8 -*-
"""
LLM Reformatter - Local LLM for sentence reordering.
Uses llama-cpp-python to load a GGUF quantized model and reformat
out-of-order words from the glove gesture recognition into natural sentences.
"""
import os
import json
import time
import threading
from typing import Optional

try:
    from llama_cpp import Llama
    _LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None  # type: ignore
    _LLAMA_CPP_AVAILABLE = False

SYSTEM_PROMPT = (
    "\u4f60\u662f\u4e00\u4e2a\u53e5\u5b50\u91cd\u6392\u5e16\u624b\u3002"
    "\u7528\u6237\u4f1a\u8f93\u5165\u4e00\u7ec4\u624b\u52bf\u8bc6\u522b\u51fa\u7684\u8bcd\u8bed\uff08\u987a\u5e8f\u53ef\u80fd\u662f\u4e71\u7684\uff09\uff0c"
    "\u8bf7\u5c06\u5b83\u4eec\u91cd\u65b0\u6392\u5217\u6210\u4e00\u4e2a\u901a\u987a\u7684\u4e2d\u6587\u53e5\u5b50\u3002\u53ea\u8f93\u51fa\u53e5\u5b50\u672c\u8eab\uff0c\u4e0d\u8981\u4efb\u4f55\u89e3\u91ca\u3001\u4e0d\u8981\u6807\u70b9\u3001\u4e0d\u8981\u5e8f\u53f7\u3001\u4e0d\u8981\u5f15\u53f7\u3002"
    "\u5982\u679c\u8bcd\u8bed\u4e2d\u5305\u542b\u6570\u5b57\u6216\u91cf\u8bcd\uff08\u5982\u201c\u65a4\u201d\u3001\u201c\u5757\u94b1\u201d\u3001\u201c\u5468\u516d\u65e5\u201d\uff09\uff0c\u8bf7\u4fdd\u7559\u5b83\u4eec\u7684\u4f4d\u7f6e\u4f7f\u5176\u8bed\u4e49\u6b63\u786e\u3002"
    "\u53ea\u8f93\u51fa\u91cd\u7ec4\u540e\u7684\u53e5\u5b50\u3002"
)

DEFAULT_REFORMAT_PROMPT_TEMPLATE = (
    "\u8bf7\u5c06\u4ee5\u4e0b\u8bcd\u8bed\u91cd\u65b0\u6392\u5217\u6210\u901a\u987a\u7684\u4e2d\u6587\u53e5\u5b50\uff1a\n"
    "{words}\n"
    "\u53ea\u8f93\u51fa\u53e5\u5b50\uff0c\u4e0d\u8981\u89e3\u91ca\u3002"
)

PROMPT_WITH_CONTEXT = (
    "\u4f60\u662f\u4e00\u4e2a\u83dc\u5e02\u573a\u7528\u8bed\u91cd\u6392\u5e16\u624b\u3002\u7528\u6237\u901a\u8fc7\u624b\u5957\u624b\u52bf\u8bc6\u522b\u51fa\u4e00\u7cfb\u5217\u8bcd\u8bed\uff08\u987a\u5e8f\u53ef\u80fd\u662f\u4e71\u7684\uff09\uff0c"
    "\u8bf7\u5c06\u5b83\u4eec\u91cd\u65b0\u6392\u5217\u6210\u4e00\u4e2a\u901a\u987a\u3001\u7b26\u5408\u83dc\u5e02\u573a\u4ea4\u6613\u573a\u666f\u7684\u4e2d\u6587\u53e5\u5b50\u3002"
    "\u5982\u679c\u8bcd\u8bed\u4e2d\u6709\u6570\u5b57\u3001\u6570\u91cf\u8bcd\u3001\u91cf\u8bcd\uff08\u5982\u201c\u65a4\u201d\u3001\u201c\u5757\u94b1\u201d\u3001\u201c\u5757\u201d\u3001\u201c\u94b1\u201d\u3001\u201c\u51e0\u201d\uff09\uff0c"
    "\u8bf7\u5408\u7406\u6392\u5217\u4f7f\u8bed\u4e49\u901a\u987a\uff08\u5982\u201c\u571f\u8c46\u591a\u5c11\u94b1\u4e00\u65a4\u201d\uff09\u3002"
    "\u53ea\u8f93\u51fa\u91cd\u7ec4\u540e\u7684\u53e5\u5b50\uff0c\u4e0d\u8981\u4efb\u4f55\u89e3\u91ca\u3001\u4e0d\u8981\u6807\u70b9\u7b26\u53f7\u3001\u4e0d\u8981\u5f15\u53f7\u3001\u4e0d\u8981\u5e8f\u53f7\u3002"
    "\u8bcd\u8bed\u5217\u8868\uff1a{words}"
)


class LLMReformatter:
    """
    Wraps a local GGUF LLM via llama-cpp-python.
    Provides sentence reordering based on accumulated gesture words.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        use_context_prompt: bool = True,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.use_context_prompt = use_context_prompt

        self._lock = threading.Lock()
        self._llm: Optional[Llama] = None
        self._init_error: Optional[Exception] = None
        self._loaded = False

        self.n_threads = n_threads or max(1, os.cpu_count() - 1)

        if not _LLAMA_CPP_AVAILABLE:
            self._init_error = ImportError(
                "llama-cpp-python is not installed. "
                "Please install it with: pip install llama-cpp-python"
            )
            return

        if not os.path.exists(model_path):
            self._init_error = FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please download a GGUF model and place it in the models/ directory.\n"
                f"See models/README.md for download instructions."
            )
            return

        self._load_model()

    def _load_model(self):
        try:
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
            )
            self._loaded = True
        except Exception as e:
            self._init_error = e
            self._loaded = False

    @property
    def is_available(self) -> bool:
        return self._loaded and self._llm is not None

    @property
    def init_error(self) -> Optional[Exception]:
        return self._init_error

    def _build_prompt(self, words: list[str]) -> str:
        words_str = "\u3001".join(words)
        if self.use_context_prompt:
            return PROMPT_WITH_CONTEXT.format(words=words_str)
        return DEFAULT_REFORMAT_PROMPT_TEMPLATE.format(words=words_str)

    def _build_chat_messages(self, words: list[str]):
        words_str = "\u3001".join(words)
        return [
            {
                "role": "system",
                "content": (
                    "\u4f60\u662f\u4e00\u4e2a\u83dc\u5e02\u573a\u7528\u8bed\u91cd\u6392\u5e16\u624b\u3002"
                    "\u4f5c\u4e1a\uff1a1)\u5c06\u8bcd\u8bed\u91cd\u6392\u6210\u901a\u987a\u53e5\u5b50\uff1b"
                    "2)\u5220\u9664\u91cd\u590d\u8bcd\u8bed\uff08\u5982\u201c\u8fd9\u8fd9\u201d\u53ea\u7559\u4e00\u4e2a\uff09\uff1b"
                    "3)\u53ea\u8f93\u51fa\u91cd\u7ec4\u540e\u7684\u53e5\u5b50\uff0c\u4e0d\u8981\u4efb\u4f55\u5176\u4ed6\u5185\u5bb9\u3002"
                ),
            },
            {
                "role": "user",
                "content": f"\u8bf7\u91cd\u6392\u8bcd\u8bed\u6210\u901a\u987a\u53e5\u5b50\uff1a{words_str}",
            },
        ]

    def _deduplicate_consecutive(self, text: str) -> str:
        """Remove consecutive duplicate characters (e.g. '这这这' -> '这')"""
        if not text:
            return text
        result = [text[0]]
        for ch in text[1:]:
            if ch != result[-1]:
                result.append(ch)
        return "".join(result)

    def _smart_deduplicate(self, text: str, max_chars: int) -> str:
        """
        Remove all duplicate characters, keeping first occurrence only,
        then truncate to max_chars.
        Example: '斤钱几斤几斤钱' -> '斤钱几' (if max_chars=3)
        """
        seen = []
        for ch in text:
            if ch not in seen:
                seen.append(ch)
                if len(seen) >= max_chars:
                    break
        return "".join(seen)

    def reformat(self, words: list[str], timeout: float = 10.0) -> Optional[str]:
        """
        Synchronously call the LLM to reorder words into a natural sentence.

        Args:
            words: List of recognized gesture words (may be out of order).
            timeout: Maximum time in seconds to wait for the LLM response.

        Returns:
            The reformatted sentence, or None if the LLM is unavailable or timed out.
        """
        if not self.is_available:
            return None

        with self._lock:
            if not self.is_available:
                return None

            messages = self._build_chat_messages(words)

            try:
                result = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=16,
                    temperature=0.05,
                )
                raw_output = result["choices"][0]["message"]["content"].strip()

                output = raw_output
                for ch in "\uff0c\u3002\uff01\uff1f\u3001\uff0b\uff1a\u300c\u300d\u300e\u300f\u201c\u201d\u2018\u2019\uff08\uff09":
                    output = output.replace(ch, "")

                output = self._deduplicate_consecutive(output)

                max_unique_chars = len(set("".join(words)))
                output = self._smart_deduplicate(output, max_unique_chars)

                return output if output else None

            except Exception as e:
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                return None

    def reformat_async(self, words: list[str], callback, timeout: float = 10.0):
        """
        Asynchronously call the LLM and invoke callback with the result.

        Args:
            words: List of recognized gesture words.
            callback: Callable[[Optional[str]], None] - called with the result.
            timeout: Maximum time in seconds.
        """
        thread = threading.Thread(
            target=self._async_wrapper,
            args=(words, callback, timeout),
            daemon=True,
        )
        thread.start()

    def _async_wrapper(self, words: list[str], callback, timeout: float):
        result = self.reformat(words, timeout=timeout)
        try:
            callback(result)
        except Exception:
            pass


def create_reformatter(
    model_path: str,
    n_ctx: int = 2048,
    n_threads: Optional[int] = None,
    n_gpu_layers: int = 0,
    verbose: bool = False,
) -> LLMReformatter:
    """
    Factory function to create an LLMReformatter instance.
    Checks for model availability and logs status.
    """
    reformatter = LLMReformatter(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )

    if reformatter.is_available:
        print(f"[LLM] Model loaded: {model_path}")
    else:
        if reformatter.init_error:
            print(f"[LLM] Failed to load model: {reformatter.init_error}")
        else:
            print(f"[LLM] Model not found: {model_path}")
        print("[LLM] LLM reformatter is disabled. Install llama-cpp-python and download a model.")
        print("[LLM] See models/README.md for instructions.")

    return reformatter
