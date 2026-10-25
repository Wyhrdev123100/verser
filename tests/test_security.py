import os
import zipfile
import pytest

from uatp.checkpoint.manager import CheckpointManager


def test_path_traversal_blocked(tmp_path):
    cm = CheckpointManager(str(tmp_path / "runs"))
    # create malicious zip with ../
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("../evil.txt", "pwned")
    try:
        cm.import_checkpoint(str(zip_path), "runx")
        # if no exception, ensure no file escaped
    except RuntimeError:
        pass
    # ensure evil.txt not created in parent
    assert not os.path.exists(tmp_path / "evil.txt")


def test_size_limit_not_enforced_but_archive_checked(tmp_path):
    # just sanity: import non-zip non-dir raises
    cm = CheckpointManager(str(tmp_path / "runs"))
    with pytest.raises(ValueError):
        cm.import_checkpoint(str(tmp_path / "nope.bin"), "runy")