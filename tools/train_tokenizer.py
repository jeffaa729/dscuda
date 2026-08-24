"""Train the dscuda byte-level BPE tokenizer from a text corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

from tokenizer import ByteBPETokenizer, split_byte_runs, train_byte_bpe


DEFAULT_DELIMITER = b"<|endoftext|>"


def iter_documents(
    path: str | Path,
    *,
    delimiter: bytes = DEFAULT_DELIMITER,
    chunk_size: int = 1 << 20,
):
    pending = b""
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            pending += chunk
            parts = pending.split(delimiter)
            pending = parts.pop()
            for document in parts:
                document = document.strip(b"\r\n")
                if document:
                    yield document
        pending = pending.strip(b"\r\n")
        if pending:
            yield pending


def count_pieces(
    path: str | Path,
    *,
    max_documents: int | None = None,
    max_bytes: int | None = None,
) -> tuple[Counter[bytes], int, int]:
    frequencies: Counter[bytes] = Counter()
    document_count = 0
    byte_count = 0

    for document in iter_documents(path):
        if max_documents is not None and document_count >= max_documents:
            break
        if max_bytes is not None and byte_count >= max_bytes:
            break
        if max_bytes is not None:
            document = document[: max_bytes - byte_count]
        frequencies.update(split_byte_runs(document))
        document_count += 1
        byte_count += len(document)

    return frequencies, document_count, byte_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="training text file")
    parser.add_argument("--output", required=True, help="output tokenizer.bin")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--max-bytes", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Counting byte runs in {args.input}")
    pieces, documents, byte_count = count_pieces(
        args.input,
        max_documents=args.max_documents,
        max_bytes=args.max_bytes,
    )
    if not pieces:
        raise RuntimeError("the tokenizer training corpus is empty")

    print(
        f"Training from {documents:,} documents, {byte_count:,} bytes, "
        f"and {len(pieces):,} unique byte runs"
    )
    tokenizer = train_byte_bpe(pieces, args.vocab_size)
    tokenizer.save(args.output)

    sample = b"Once upon a time, a little model learned CUDA."
    encoded = tokenizer.encode(sample, add_eos=True)
    decoded = tokenizer.decode(encoded)
    if decoded != sample.decode("utf-8"):
        raise RuntimeError("tokenizer round-trip verification failed")

    print(f"Saved {tokenizer.vocab_size:,}-token vocabulary to {args.output}")
    print(f"Round trip: {len(sample)} bytes -> {len(encoded)} tokens")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Tokenizer training failed: {error}", file=sys.stderr)
        sys.exit(1)
