import hashlib
import json

from forte_vision.audit import save_audit


def test_audit_is_opt_in_and_local_private(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("FORTE_LOCAL_AUDIT_DIR", raising=False)
    save_audit("../job", {"private": "observation"}, {}, [], [])
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setenv("FORTE_LOCAL_AUDIT_DIR", str(tmp_path / "audit"))
    save_audit("../job", {"private": "observation"}, {}, [(1., 2.)], [])
    target = tmp_path / "audit" / (hashlib.sha256(b"../job").hexdigest() + ".json")
    assert json.loads(target.read_text())["raw_intervals"] == [[1., 2.]]
    assert target.stat().st_mode & 0o777 == 0o600
    assert target.parent.stat().st_mode & 0o777 == 0o700
    assert not capsys.readouterr().out
