"""Tests BPE training, serialization, Unicode fallback, and round trips."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from tokenizer import ByteBPETokenizer, split_byte_runs, train_byte_bpe


class TokenizerTest(unittest.TestCase):
    def test_train_save_load_round_trip(self) -> None:
        corpus = [
            "Once upon a time there was a tiny GPU.",
            "Once upon a time there was a tiny model.",
            "CUDA makes the tiny model fast.",
        ]
        pieces: Counter[bytes] = Counter()
        for text in corpus:
            pieces.update(split_byte_runs(text.encode("utf-8")))

        tokenizer = train_byte_bpe(pieces, 300, progress_interval=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.bin"
            tokenizer.save(path)
            loaded = ByteBPETokenizer.load(path)

        for text in corpus + ["Unicode fallback: café 小模型"]:
            tokens = loaded.encode(text, add_bos=True, add_eos=True)
            self.assertEqual(loaded.bos_id, tokens[0])
            self.assertEqual(loaded.eos_id, tokens[-1])
            self.assertEqual(text, loaded.decode(tokens))

    def test_merges_do_not_cross_byte_classes(self) -> None:
        pieces = Counter({b"hello": 10, b" ": 10, b"world": 10})
        tokenizer = train_byte_bpe(pieces, 280, progress_interval=0)
        self.assertEqual(
            tokenizer.encode("hello world"),
            tokenizer.encode("hello")
            + tokenizer.encode(" ")
            + tokenizer.encode("world"),
        )


if __name__ == "__main__":
    unittest.main()
