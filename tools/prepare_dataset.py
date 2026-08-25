"""Encode train/validation text into dscuda uint16 token streams."""

from __future__ import annotations

import argparse
from array import array
import json
from pathlib import Path
import sys

from tokenizer import ByteBPETokenizer
from train_tokenizer import iter_documents


def encode_split(
    input_path: str | Path,
    output_path: str | Path,
    tokenizer: ByteBPETokenizer,
    *,
    max_documents: int | None = None,
) -> dict[str, int]:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document_count = 0
    byte_count = 0
    token_count = 0
    buffer = array("H")

    with destination.open("wb") as output:
        for document in iter_documents(input_path):
            if max_documents is not None and document_count >= max_documents:
                break
            tokens = tokenizer.encode(document, add_eos=True)
            buffer.extend(tokens)
            document_count += 1
            byte_count += len(document)
            token_count += len(tokens)

            if len(buffer) >= 1 << 20:
                if sys.byteorder != "little":
                    buffer.byteswap()
                buffer.tofile(output)
                buffer = array("H")

        if sys.byteorder != "little":
            buffer.byteswap()
        buffer.tofile(output)

    return {
        "documents": document_count,
        "text_bytes": byte_count,
        "tokens": token_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train-documents", type=int)
    parser.add_argument("--max-validation-documents", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteBPETokenizer.load(args.tokenizer)

    print(f"Encoding training data from {args.train}")
    train = encode_split(
        args.train,
        output_dir / "train.bin",
        tokenizer,
        max_documents=args.max_train_documents,
    )
    print(f"Encoding validation data from {args.validation}")
    validation = encode_split(
        args.validation,
        output_dir / "val.bin",
        tokenizer,
        max_documents=args.max_validation_documents,
    )

    metadata = {
        "format": "dscuda-token-stream-v1",
        "dtype": "uint16-le",
        "vocab_size": tokenizer.vocab_size,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
        "train": train,
        "validation": validation,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    for name, values in (("train", train), ("validation", validation)):
        ratio = values["text_bytes"] / values["tokens"]
        print(
            f"  {name:10s} {values['documents']:8,d} documents  "
            f"{values['tokens']:12,d} tokens  {ratio:.2f} bytes/token"
        )
    print(f"Wrote token streams and metadata to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Dataset preparation failed: {error}", file=sys.stderr)
        sys.exit(1)
