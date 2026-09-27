# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

import os

# The unit suite is CPU-only. Pin it so a saturated/foreign-occupied local GPU
# can't OOM torch's per-step CUDA health check on shared hosts.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT
