"""Byte-level BPE training and serialization for dscuda.

The tokenizer starts with one token for every byte and learns merges inside
simple byte-class runs. This keeps training, Python preprocessing, and the C++
runtime implementation deterministic without external tokenizer libraries.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import heapq
from pathlib import Path
import struct
from typing import Iterable, Iterator


MAGIC = b"DSBPE01\0"
VERSION = 1
BYTE_VOCAB_SIZE = 256
BOS_ID = 256
EOS_ID = 257
FIRST_MERGE_ID = 258
HEADER = struct.Struct("<8s6I")
MERGE = struct.Struct("<3I")


def _byte_class(value: int) -> int:
    if 65 <= value <= 90 or 97 <= value <= 122:
        return 0
    if 48 <= value <= 57:
        return 1
    if value in (9, 10, 11, 12, 13, 32):
        return 2
    return 3


def split_byte_runs(data: bytes) -> Iterator[bytes]:
    """Yield maximal letter, digit, whitespace, or other byte runs."""
    if not data:
        return

    start = 0
    current_class = _byte_class(data[0])
    for index in range(1, len(data)):
        next_class = _byte_class(data[index])
        if next_class != current_class:
            yield data[start:index]
            start = index
            current_class = next_class
    yield data[start:]


@dataclass(frozen=True)
class Merge:
    left: int
    right: int
    result: int


class ByteBPETokenizer:
    def __init__(self, merges: Iterable[Merge] = ()) -> None:
        self.merges = list(merges)
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self._merge_by_pair: dict[tuple[int, int], tuple[int, int]] = {}

        for rank, merge in enumerate(self.merges):
            expected_result = FIRST_MERGE_ID + rank
            if merge.result != expected_result:
                raise ValueError("merge result IDs must be sequential")
            self._merge_by_pair[(merge.left, merge.right)] = (
                rank,
                merge.result,
            )

        self._token_bytes = [bytes((value,)) for value in range(256)]
        self._token_bytes.extend((b"", b""))
        for merge in self.merges:
            if merge.left >= len(self._token_bytes) or merge.right >= len(
                self._token_bytes
            ):
                raise ValueError("merge references a token that is not defined")
            self._token_bytes.append(
                self._token_bytes[merge.left] + self._token_bytes[merge.right]
            )

    @property
    def vocab_size(self) -> int:
        return FIRST_MERGE_ID + len(self.merges)

    def _encode_piece(self, piece: bytes) -> list[int]:
        if len(piece) < 2 or not self.merges:
            return list(piece)

        tokens = list(piece)
        previous = [index - 1 for index in range(len(tokens))]
        following = [index + 1 for index in range(len(tokens))]
        following[-1] = -1
        alive = [True] * len(tokens)
        queue: list[tuple[int, int]] = []

        def push_pair(left_index: int) -> None:
            if left_index < 0 or not alive[left_index]:
                return
            right_index = following[left_index]
            if right_index < 0:
                return
            merge = self._merge_by_pair.get(
                (tokens[left_index], tokens[right_index])
            )
            if merge is not None:
                heapq.heappush(queue, (merge[0], left_index))

        for index in range(len(tokens) - 1):
            push_pair(index)

        while queue:
            rank, left_index = heapq.heappop(queue)
            if not alive[left_index]:
                continue
            right_index = following[left_index]
            if right_index < 0 or not alive[right_index]:
                continue

            merge = self._merge_by_pair.get(
                (tokens[left_index], tokens[right_index])
            )
            if merge is None or merge[0] != rank:
                continue

            tokens[left_index] = merge[1]
            alive[right_index] = False
            following[left_index] = following[right_index]
            if following[right_index] >= 0:
                previous[following[right_index]] = left_index

            push_pair(previous[left_index])
            push_pair(left_index)

        return [tokens[index] for index in range(len(tokens)) if alive[index]]

    def encode(
        self,
        text: str | bytes,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        data = text.encode("utf-8") if isinstance(text, str) else text
        tokens = [self.bos_id] if add_bos else []
        for piece in split_byte_runs(data):
            tokens.extend(self._encode_piece(piece))
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, tokens: Iterable[int], *, skip_special: bool = True) -> str:
        output = bytearray()
        for token in tokens:
            if token in (self.bos_id, self.eos_id):
                if skip_special:
                    continue
                marker = b"<bos>" if token == self.bos_id else b"<eos>"
                output.extend(marker)
                continue
            if token < 0 or token >= len(self._token_bytes):
                raise ValueError(f"token ID {token} is outside the vocabulary")
            output.extend(self._token_bytes[token])
        return output.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as file:
            file.write(
                HEADER.pack(
                    MAGIC,
                    VERSION,
                    self.vocab_size,
                    len(self.merges),
                    self.bos_id,
                    self.eos_id,
                    0,
                )
            )
            for merge in self.merges:
                file.write(MERGE.pack(merge.left, merge.right, merge.result))

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        with Path(path).open("rb") as file:
            header_data = file.read(HEADER.size)
            if len(header_data) != HEADER.size:
                raise ValueError("tokenizer header is truncated")
            magic, version, vocab_size, merge_count, bos_id, eos_id, _ = (
                HEADER.unpack(header_data)
            )
            if magic != MAGIC or version != VERSION:
                raise ValueError("unsupported tokenizer format")
            if bos_id != BOS_ID or eos_id != EOS_ID:
                raise ValueError("unsupported special-token layout")

            merges = []
            for _ in range(merge_count):
                record = file.read(MERGE.size)
                if len(record) != MERGE.size:
                    raise ValueError("tokenizer merge table is truncated")
                merges.append(Merge(*MERGE.unpack(record)))

            if file.read(1):
                raise ValueError("tokenizer file has trailing data")

        tokenizer = cls(merges)
        if tokenizer.vocab_size != vocab_size:
            raise ValueError("tokenizer vocabulary size does not match merges")
        return tokenizer


def train_byte_bpe(
    piece_frequencies: Counter[bytes],
    vocab_size: int,
    *,
    progress_interval: int = 100,
) -> ByteBPETokenizer:
    """Train byte BPE from weighted byte runs using incremental pair counts."""
    if vocab_size < FIRST_MERGE_ID or vocab_size > 65535:
        raise ValueError("vocabulary size must be in [258, 65535]")

    words: list[list[int]] = []
    frequencies: list[int] = []
    for piece, frequency in sorted(piece_frequencies.items()):
        if piece:
            words.append(list(piece))
            frequencies.append(frequency)

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    for word_index, word in enumerate(words):
        local_counts = Counter(zip(word, word[1:]))
        for pair, count in local_counts.items():
            pair_counts[pair] += count * frequencies[word_index]
            pair_words[pair].add(word_index)

    queue = [(-count, pair[0], pair[1]) for pair, count in pair_counts.items()]
    heapq.heapify(queue)
    merges: list[Merge] = []

    while len(merges) < vocab_size - FIRST_MERGE_ID:
        while queue:
            negative_count, left, right = heapq.heappop(queue)
            pair = (left, right)
            if pair_counts.get(pair, 0) == -negative_count:
                break
        else:
            break

        if -negative_count <= 0:
            break

        result = FIRST_MERGE_ID + len(merges)
        affected_words = sorted(pair_words.get(pair, ()))
        touched_pairs: set[tuple[int, int]] = set()

        for word_index in affected_words:
            word = words[word_index]
            before = Counter(zip(word, word[1:]))
            if pair not in before:
                continue

            frequency = frequencies[word_index]
            for old_pair, count in before.items():
                pair_counts[old_pair] -= count * frequency
                pair_words[old_pair].discard(word_index)
                touched_pairs.add(old_pair)

            merged_word: list[int] = []
            index = 0
            while index < len(word):
                if (
                    index + 1 < len(word)
                    and word[index] == left
                    and word[index + 1] == right
                ):
                    merged_word.append(result)
                    index += 2
                else:
                    merged_word.append(word[index])
                    index += 1
            words[word_index] = merged_word

            after = Counter(zip(merged_word, merged_word[1:]))
            for new_pair, count in after.items():
                pair_counts[new_pair] += count * frequency
                pair_words[new_pair].add(word_index)
                touched_pairs.add(new_pair)

        merges.append(Merge(left, right, result))
        for touched_pair in touched_pairs:
            count = pair_counts.get(touched_pair, 0)
            if count > 0:
                heapq.heappush(
                    queue,
                    (-count, touched_pair[0], touched_pair[1]),
                )

        if progress_interval and (
            len(merges) % progress_interval == 0
            or len(merges) == vocab_size - FIRST_MERGE_ID
        ):
            print(
                f"  learned {len(merges):4d} merges; "
                f"latest pair frequency {-negative_count:,}"
            )

    return ByteBPETokenizer(merges)
