from pathlib import Path

import pytest

from cmcert.certification import require_native_do
from cmcert.errors import NativeDoUnavailableError, PreregistrationError
from cmcert.prereg import verify_prereg, write_lock


def test_non_native_do_dataset_is_structurally_rejected():
    with pytest.raises(NativeDoUnavailableError):
        require_native_do("waterbirds")


def test_lock_rejects_placeholders_and_changed_files(tmp_path: Path):
    thresholds = tmp_path / "thresholds.yaml"
    thresholds.write_text("alpha: ???\n", encoding="utf-8")
    with pytest.raises(PreregistrationError, match="incomplete"):
        write_lock(thresholds)

    thresholds.write_text("alpha: 0.05\n", encoding="utf-8")
    write_lock(thresholds)
    verify_prereg(thresholds)
    thresholds.write_text("alpha: 0.01\n", encoding="utf-8")
    with pytest.raises(PreregistrationError, match="differs"):
        verify_prereg(thresholds)


def test_forbidden_audit_in_training_loop_does_not_exist():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "src" / "cmcert" / "training" / "loop_x.py").exists()
