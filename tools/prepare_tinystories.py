"""Download a bounded TinyStories corpus and prepare it for dscuda training."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from urllib.request import Request, urlopen

from prepare_dataset import encode_split
from tokenizer import ByteBPETokenizer, train_byte_bpe
from train_tokenizer import DEFAULT_DELIMITER, count_pieces


BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
TRAIN_URL = f"{BASE_URL}/TinyStories-train.txt"
VALIDATION_URL = f"{BASE_URL}/TinyStories-valid.txt"


def download_complete_documents(
    url: str,
    path: Path,
    max_bytes: int,
) -> None:
    if path.exists() and path.stat().st_size > 0:
        print(f"Using existing {path} ({path.stat().st_size:,} bytes)")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    request = Request(url, headers={"User-Agent": "dscuda-dataset-preparation/1.0"})
    received = 0
    pending = b""

    print(f"Downloading up to {max_bytes:,} bytes from {url}")
    with urlopen(request) as response, temporary.open("wb") as output:
        while received < max_bytes:
            chunk = response.read(min(1 << 20, max_bytes - received))
            if not chunk:
                break
            received += len(chunk)
            pending += chunk
            last_boundary = pending.rfind(DEFAULT_DELIMITER)
            if last_boundary >= 0:
                boundary_end = last_boundary + len(DEFAULT_DELIMITER)
                output.write(pending[:boundary_end])
                pending = pending[boundary_end:]
            print(
                f"\r  received {received / (1 << 20):7.1f} MiB",
                end="",
                flush=True,
            )
    print()

    if temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("download did not contain a complete story")
    shutil.move(temporary, path)
    print(f"Saved {path} ({path.stat().st_size:,} complete-document bytes)")


def parse_size(text: str) -> int:
    suffixes = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}
    value = text.strip().lower()
    if value[-1:] in suffixes:
        return int(float(value[:-1]) * suffixes[value[-1]])
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/tinystories")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--train-bytes", type=parse_size, default=parse_size("32m"))
    parser.add_argument(
        "--validation-bytes", type=parse_size, default=parse_size("4m")
    )
    parser.add_argument(
        "--tokenizer-bytes", type=parse_size, default=parse_size("16m")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    train_text = raw_dir / "train.txt"
    validation_text = raw_dir / "validation.txt"
    tokenizer_path = output_dir / "tokenizer.bin"

    download_complete_documents(TRAIN_URL, train_text, args.train_bytes)
    download_complete_documents(
        VALIDATION_URL,
        validation_text,
        args.validation_bytes,
    )

    if tokenizer_path.exists():
        print(f"Using existing tokenizer {tokenizer_path}")
        tokenizer = ByteBPETokenizer.load(tokenizer_path)
        if tokenizer.vocab_size != args.vocab_size:
            raise RuntimeError(
                "existing tokenizer vocabulary does not match --vocab-size; "
                "remove the output directory or choose a new one"
            )
    else:
        print("Counting byte runs for tokenizer training")
        pieces, documents, byte_count = count_pieces(
            train_text,
            max_bytes=args.tokenizer_bytes,
        )
        print(
            f"Training {args.vocab_size:,}-token BPE from "
            f"{documents:,} stories and {byte_count:,} bytes"
        )
        tokenizer = train_byte_bpe(pieces, args.vocab_size)
        tokenizer.save(tokenizer_path)

    print("Encoding train.bin")
    train = encode_split(train_text, output_dir / "train.bin", tokenizer)
    print("Encoding val.bin")
    validation = encode_split(
        validation_text,
        output_dir / "val.bin",
        tokenizer,
    )

    import json

    metadata = {
        "format": "dscuda-token-stream-v1",
        "source": "roneneldan/TinyStories",
        "dtype": "uint16-le",
        "vocab_size": tokenizer.vocab_size,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
        "train": train,
        "validation": validation,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print("\nTinyStories is ready for the training loop")
    print(f"  tokenizer: {tokenizer_path}")
    print(f"  train:     {output_dir / 'train.bin'} ({train['tokens']:,} tokens)")
    print(
        f"  validation:{output_dir / 'val.bin'} "
        f"({validation['tokens']:,} tokens)"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"TinyStories preparation failed: {error}", file=sys.stderr)
        sys.exit(1)
