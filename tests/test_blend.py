# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.make_blend_shards import interleave


def make_entries():
    return [("fineweb", 7, [f"f{i}" for i in range(7)]),
            ("cosmo", 3, [f"c{i}" for i in range(3)])]


def test_ratio_exact_and_prefix_bounded():
    order = interleave(make_entries())
    assert len(order) == 10
    assert sum(x.startswith("f") for x in order) == 7
    prefix_f = sum(x.startswith("f") for x in order[:4])
    assert 1 <= prefix_f <= 4
    assert sum(x.startswith("f") for x in order[:2]) >= 1


def test_deterministic_and_source_order_preserved():
    a = interleave(make_entries())
    b = interleave(make_entries())
    assert a == b
    fws = [x for x in a if x.startswith("f")]
    cws = [x for x in a if x.startswith("c")]
    assert fws == sorted(fws) and cws == sorted(cws)


def test_end_to_end_hardlinks_and_val(tmp_path):
    fw = tmp_path / "fineweb3"
    co = tmp_path / "cosmopedia3"
    fw.mkdir()
    co.mkdir()
    for i in range(5):
        (fw / f"train_{i:04d}.pt").write_bytes(b"x" * 8)
    for i in range(3):
        (co / f"train_{i:04d}.pt").write_bytes(b"y" * 8)
    script = Path(__file__).parent.parent / "scripts" / "make_blend_shards.py"
    subprocess.run([
        sys.executable, str(script), "--output", str(tmp_path / "blend"),
        "--source", str(fw), "--take", "3", "--val-take", "1",
        "--source", str(co), "--take", "2", "--val-take", "1",
    ], check=True, cwd=tmp_path)
    train = sorted((tmp_path / "blend").glob("train_*.pt"))
    val = sorted((tmp_path / "blend_val").glob("val_*.pt"))
    assert len(train) == 5 and len(val) == 2
    assert train[0].stat().st_nlink == 2
    src_f = (fw / "train_0000.pt").read_bytes()
    assert src_f == train[0].read_bytes() or src_f == train[1].read_bytes()
    val_bytes = val[0].read_bytes() + val[1].read_bytes()
    assert b"x" * 8 in val_bytes and b"y" * 8 in val_bytes
    used = [p.name for p in train] + [p.name for p in val]
    assert len(set(used)) == 7
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([
            sys.executable, str(script), "--output", str(tmp_path / "blend"),
            "--source", str(fw), "--take", "1", "--val-take", "0",
            "--source", str(co), "--take", "1", "--val-take", "0",
        ], check=True, capture_output=True, cwd=tmp_path)


def test_start_offset_disjoint(tmp_path):
    src = tmp_path / "fineweb3"
    src.mkdir()
    for i in range(10):
        (src / f"train_{i:04d}.pt").write_bytes(b"x" * 8)
    script = Path(__file__).parent.parent / "scripts" / "make_blend_shards.py"
    subprocess.run([
        sys.executable, str(script), "--output", str(tmp_path / "blend_a"),
        "--source", str(src), "--take", "4", "--val-take", "0",
    ], check=True, cwd=tmp_path)
    subprocess.run([
        sys.executable, str(script), "--output", str(tmp_path / "blend_b"),
        "--source", str(src), "--take", "4", "--val-take", "0", "--start", "4",
    ], check=True, cwd=tmp_path)
    inos_a = {p.stat().st_ino for p in (tmp_path / "blend_a").glob("train_*.pt")}
    inos_b = {p.stat().st_ino for p in (tmp_path / "blend_b").glob("train_*.pt")}
    assert len(inos_a) == 4 and len(inos_b) == 4
    assert inos_a.isdisjoint(inos_b)


def test_start_out_of_range_fails(tmp_path):
    src = tmp_path / "fineweb3"
    src.mkdir()
    for i in range(5):
        (src / f"train_{i:04d}.pt").write_bytes(b"x" * 8)
    script = Path(__file__).parent.parent / "scripts" / "make_blend_shards.py"
    proc = subprocess.run([
        sys.executable, str(script), "--output", str(tmp_path / "blend"),
        "--source", str(src), "--take", "2", "--val-take", "0", "--start", "4",
    ], capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode != 0
    assert "only 5 available" in proc.stderr
    assert not (tmp_path / "blend").exists()


def test_start_val_isolation(tmp_path):
    src = tmp_path / "fineweb3"
    src.mkdir()
    for i in range(6):
        (src / f"train_{i:04d}.pt").write_bytes(bytes([i + 1]) * 8)
    script = Path(__file__).parent.parent / "scripts" / "make_blend_shards.py"
    subprocess.run([
        sys.executable, str(script), "--output", str(tmp_path / "blend"),
        "--source", str(src), "--take", "2", "--val-take", "2", "--start", "2",
    ], check=True, cwd=tmp_path)
    train = sorted((tmp_path / "blend").glob("train_*.pt"))
    val = sorted((tmp_path / "blend_val").glob("val_*.pt"))
    assert len(train) == 2 and len(val) == 2
    assert train[0].read_bytes() == b"\x03" * 8
    assert train[1].read_bytes() == b"\x04" * 8
    assert val[0].read_bytes() == b"\x05" * 8
    assert val[1].read_bytes() == b"\x06" * 8
