"""Chinese text resources, initialized only by operations that need them."""

from threading import RLock

_lock = RLock()
_tokenizer = None
_converter = None


def get_tokenizer():
    global _tokenizer
    with _lock:
        if _tokenizer is None:
            import jieba3

            _tokenizer = jieba3.jieba3()
        return _tokenizer


def get_converter():
    global _converter
    with _lock:
        if _converter is None:
            import opencc

            _converter = opencc.OpenCC("s2twp")
        return _converter
