import re

TOKENS = [
    "<cls>",  # cls is also bos?
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
    "<null_1>",
    "<mask>",
]

TOKENIZER = {t: idx for (idx, t) in enumerate(TOKENS)}

PAD_IDX: int = 1
MASK_IDX: int = 32

assert TOKENIZER["<pad>"] == PAD_IDX
assert TOKENIZER["<mask>"] == MASK_IDX


def tokenize(s: str) -> list[int]:
    tokens = re.findall(r"<\w+>|[A-Z\.\-]", s)
    return [TOKENIZER[t] for t in tokens]


def decode(tokens) -> str:
    return "".join([TOKENS[i] for i in tokens])
