from __future__ import annotations

from pathlib import Path

import pytest

import scripts.resume as resume


def _checkpoint_tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "checkpoints"
    checkpoint = root / "step_0000001"
    checkpoint.mkdir(parents=True)
    for name in ("config.json", "pytorch_model.bin", "training_state.pt"):
        (checkpoint / name).write_bytes(b"placeholder")
    return root, checkpoint


def test_resume_rejects_redirected_training_state_during_candidate_scan(
    tmp_path,
    monkeypatch,
    capsys,
):
    root, _ = _checkpoint_tree(tmp_path)
    monkeypatch.setattr(resume, "reject_uninitialized_torchrun_environment", lambda: 1)
    monkeypatch.setattr(
        resume,
        "_is_link_or_reparse_point",
        lambda path: Path(path).name == "training_state.pt",
    )
    monkeypatch.setattr(
        resume.torch,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("torch.load must not read a redirected state file")
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["resume.py", "--checkpoint-dir", str(root)],
    )

    with pytest.raises(SystemExit):
        resume.main()

    assert "checkpoint files must not be links or reparse points" in capsys.readouterr().err


def test_resume_rechecks_training_state_immediately_before_torch_load(
    tmp_path,
    monkeypatch,
):
    root, _ = _checkpoint_tree(tmp_path)
    state_checks = 0

    def changes_after_scan(path):
        nonlocal state_checks
        if Path(path).name != "training_state.pt":
            return False
        state_checks += 1
        return state_checks >= 2

    monkeypatch.setattr(resume, "reject_uninitialized_torchrun_environment", lambda: 1)
    monkeypatch.setattr(resume, "_is_link_or_reparse_point", changes_after_scan)
    monkeypatch.setattr(
        resume.torch,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("torch.load must not win the reparse race")
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["resume.py", "--checkpoint-dir", str(root)],
    )

    with pytest.raises(RuntimeError, match="regular local file before loading"):
        resume.main()

    assert state_checks == 2
