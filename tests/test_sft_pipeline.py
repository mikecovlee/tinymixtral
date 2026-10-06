"""Offline unit tests for the SFT data/training/publish pipeline."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

REPO = Path(__file__).resolve().parents[1]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_build = _load("v30it_build_dataset", "data/v3/sft/build_dataset.py")
_train = _load("v30it_train_sft", "versions/v3.0-it/train_sft.py")

_NUM_PERM = _build._NUM_PERM
_norm = _build._norm
accept = _build.accept
build_eval_grams = _build.build_eval_grams
egrams = _build.egrams
first_user_hash = _build.first_user_hash
np_jaccard = _build.np_jaccard
shingle_arr = _build.shingle_arr
signature = _build.signature
format_conversation = _train.format_conversation
pack_sequences = _train.pack_sequences
tokenize_with_mask = _train.tokenize_with_mask


# ---------------------------------------------------------------- format/mask
def test_format_conversation_spans_cover_assistant_only():
    turns = [
        {"from": "human", "value": "explain the moon"},
        {"from": "gpt", "value": "the moon orbits earth"},
    ]
    text, spans = format_conversation(turns)
    assert len(spans) == 1
    start, end = spans[0]
    seg = text[start:end]
    assert seg.startswith("<|assistant|>")
    assert "the moon orbits earth" in seg
    assert "<|user|>" not in seg
    assert text[:start].startswith("<|user|>")


def _tiny_tokenizer(text):
    vocab = {w: i for i, w in enumerate(sorted(set(text.split())))}
    vocab["[UNK]"] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]")


def test_tokenize_with_mask_labels_assistant_span():
    turns = [
        {"from": "human", "value": "explain the moon"},
        {"from": "gpt", "value": "the moon orbits earth"},
    ]
    text, spans = format_conversation(turns)
    tok = _tiny_tokenizer(text)
    ids, labels = tokenize_with_mask(text, spans, tok)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    assert len(ids) == len(labels) == len(enc["offset_mapping"])
    (s, e), = spans
    for i, (cs, ce) in enumerate(enc["offset_mapping"]):
        overlaps = cs < e and ce > s and ce > cs
        if overlaps:
            assert labels[i] == ids[i], f"token {i} inside span must be labeled"
        else:
            assert labels[i] == -100, f"token {i} outside span must be masked"


# ---------------------------------------------------------------- packing
def test_pack_sequences_shift_pad_truncate():
    pad = 0
    # single short example: padded, labels shifted by one, last col -100
    ids = [5, 6, 7]
    labs = [5, 6, 7]
    packed_ids, packed_labels = pack_sequences([(ids, labs)], 4, pad)
    assert len(packed_ids) == 1
    row_i, row_l = packed_ids[0], packed_labels[0]
    assert row_i.shape == (4,) and row_l.shape == (4,)
    assert row_i.dtype == np.int32 and row_l.dtype == np.int32
    assert list(row_i) == [5, 6, 7, pad]
    assert list(row_l) == [6, 7, -100, -100]

    # two examples crossing the buffer boundary -> two rows
    a = ([1, 2, 3], [1, 2, 3])
    b = ([4, 5, 6], [4, 5, 6])
    packed_ids, packed_labels = pack_sequences([a, b], 4, pad)
    assert len(packed_ids) == 2
    assert all(r.shape == (4,) for r in packed_ids + packed_labels)
    assert list(packed_labels[0])[-1] == -100

    # over-long example is truncated, no exception
    big = (list(range(1, 11)), list(range(1, 11)))
    packed_ids, packed_labels = pack_sequences([big], 4, pad)
    assert len(packed_ids) == 1
    assert list(packed_ids[0]) == [1, 2, 3, 4]


# ---------------------------------------------------------------- build_dataset helpers
def test_norm_and_hashes():
    assert _norm("  A\tB \n c ") == "a b c"
    conv = [
        {"from": "human", "value": "What is 2+2?"},
        {"from": "gpt", "value": "four"},
    ]
    import hashlib

    assert first_user_hash(conv) == hashlib.sha1(b"what is 2+2?").hexdigest()
    assert first_user_hash([{"from": "gpt", "value": "x"}]) is None


def test_accept_boundaries():
    good_a = "x" * 40
    assert accept([{"from": "human", "value": "q"}, {"from": "gpt", "value": good_a}])
    assert not accept([{"from": "human", "value": "q"}])  # len < 2
    assert not accept([{"from": "human", "value": "q"}, {"from": "human", "value": good_a}])
    assert not accept([{"from": "human", "value": "q"}, {"from": "gpt", "value": "x" * 39}])
    assert not accept([{"from": "human", "value": "q"}, {"from": "gpt", "value": "x" * 12001}])
    assert not accept([{"from": "human", "value": "  "}, {"from": "gpt", "value": good_a}])


def test_minhash_helpers():
    t = "the quick brown fox jumps over the lazy dog and keeps running far"
    a = shingle_arr(t)
    b = shingle_arr(t)
    assert a.size > 0 and np.array_equal(a, b)  # deterministic
    assert shingle_arr("ab").size == 0  # shorter than n=5
    long_t = t * 50
    assert shingle_arr(long_t).size <= 64  # bounded sketch

    perms = np.arange(_NUM_PERM, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    sig_a = signature(a, perms)
    assert sig_a.shape == (_NUM_PERM,)
    assert np.array_equal(sig_a, signature(b, perms))
    assert (signature(np.zeros(0, dtype=np.uint64), perms) == 0).all()

    assert np_jaccard(a, a) == 1.0
    assert np_jaccard(a, np.zeros(0, dtype=np.uint64)) == 0.0
    disjoint = np.array([x for x in range(1000, 1000 + a.size)], dtype=np.uint64)
    assert np_jaccard(a, disjoint) == 0.0

    # word 10-grams: 12 words -> 3 windows; shared window -> shared digest
    w12 = " ".join(f"w{i}" for i in range(12))
    assert len(egrams(w12)) == 3
    w12_shift = " ".join(f"w{i}" for i in range(2, 14))
    assert egrams(w12) & egrams(w12_shift)
    assert len(egrams("only few words")) == 1


def test_build_eval_grams_disabled_is_offline():
    grams, used = build_eval_grams(False)
    assert grams == set()
    assert used == []


# ---------------------------------------------------------------- publish_hf
TINY = dict(
    vocab_size=1024,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=64,
    num_local_experts=3,
    num_experts_per_tok=2,
    expert_intermediate_size=48,
    router_jitter_noise=0.0,
)


def test_publish_hf_whitelist(tmp_path):
    from model.topk.config import TinyMixtralConfig
    from model.topk.modeling import TinyMixtralForCausalLM

    torch.manual_seed(0)
    model = TinyMixtralForCausalLM(TinyMixtralConfig(**TINY))
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    torch.save(model.state_dict(), ckpt / "pytorch_model.bin")
    (ckpt / "config.json").write_text(json.dumps(model.config.to_dict()), encoding="utf-8")

    tokdir = tmp_path / "tok"
    tokdir.mkdir()
    kept = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]
    for name in kept + ["model.safetensors", "zzz.bin"]:
        (tokdir / name).write_text("dummy", encoding="utf-8")

    out = tmp_path / "pub"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "publish_hf.py"),
            "--checkpoint", str(ckpt),
            "--output", str(out),
            "--tokenizer", str(tokdir),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "forward loss" in proc.stdout

    names = {p.name for p in out.iterdir()}
    assert "pytorch_model.bin" in names
    assert "model.safetensors" not in names
    assert "LICENSE" in names
    assert {"configuration_tinymixtral.py", "modeling_tinymixtral.py"} <= names
    assert set(kept) <= names
    assert "zzz.bin" not in names

    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["auto_map"]["AutoModelForCausalLM"] == "modeling_tinymixtral.TinyMixtralForCausalLM"
    assert cfg["auto_map"]["AutoConfig"] == "configuration_tinymixtral.TinyMixtralConfig"
