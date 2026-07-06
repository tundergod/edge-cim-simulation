"""Phase 2.4 hardening — the three validator gates whose main() now returns a real
process exit code (nonzero on failure, 0 on pass): validate_contention,
validate_m7_energy, validate_m5_trace. Each test redirects the validator's own
output-path constant (OUT / ROOT) to tmp_path so nothing under the committed
validation/reports/ tree (or simulator/models/params/) is ever touched.

    .venv/bin/pytest tests/test_gate_exit_codes.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))
import validate_contention  # noqa: E402
import validate_m5_trace  # noqa: E402
import validate_m7_energy  # noqa: E402
from simulator.runtime.resources import SharedBandwidth  # noqa: E402


# ---------------------------------------------------------------------------
# validate_contention: gate = rising_then_flat AND off_linear (both SHAPE checks
# on SharedBandwidth.aggregate_GBs)
# ---------------------------------------------------------------------------

def test_contention_gate_fails_on_forced_constant_bandwidth(tmp_path, monkeypatch):
    # a constant aggregate_GBs (ignores k / contention) breaks BOTH shape checks at once:
    # rising_then_flat needs sweep[1] < sweep[2]; off_linear needs sweep_off[4] == 4*EFF_BW.
    monkeypatch.setattr(validate_contention, "OUT", tmp_path)
    monkeypatch.setattr(SharedBandwidth, "aggregate_GBs", lambda self, k, contention=True: 10.0)
    assert validate_contention.main() != 0


def test_contention_gate_passes_on_real_data(tmp_path, monkeypatch):
    monkeypatch.setattr(validate_contention, "OUT", tmp_path)
    assert validate_contention.main() == 0


# ---------------------------------------------------------------------------
# validate_m7_energy: gate = all_positive & monotonic & plausible-power & no_flip_pm20
# ---------------------------------------------------------------------------

def _m7_dirs(tmp_path):
    (tmp_path / "validation" / "reports" / "phase1.1").mkdir(parents=True)
    (tmp_path / "simulator" / "models" / "params").mkdir(parents=True)


def test_m7_gate_fails_when_cim_energy_dominates(tmp_path, monkeypatch):
    _m7_dirs(tmp_path)
    monkeypatch.setattr(validate_m7_energy, "ROOT", tmp_path)
    # cim_tops_w near-zero -> CIM projection energy swamps DRAM streaming -> "memory
    # dominates" flips at every +/-20% corner (no_flip_pm20 becomes False).
    bad_params = dict(validate_m7_energy.PARAMS, cim_tops_w=1e-4)
    monkeypatch.setattr(validate_m7_energy, "PARAMS", bad_params)
    assert validate_m7_energy.main() != 0


def test_m7_gate_passes_on_real_params(tmp_path, monkeypatch):
    _m7_dirs(tmp_path)
    monkeypatch.setattr(validate_m7_energy, "ROOT", tmp_path)
    assert validate_m7_energy.main() == 0


# ---------------------------------------------------------------------------
# validate_m5_trace: gate = all_semantic_covered AND zero orphan ops, per model
# ---------------------------------------------------------------------------

def _write_m5_fixture(inv_dir, prof_dir, model, *, covered=True, orphan=False):
    inv = {"expected_ops_check": {"all_semantic_covered": covered},
           "distinct_ops": ["aten.mm.default"]}
    (inv_dir / f"{model}.json").write_text(json.dumps(inv))
    rows = [{"op": "aten.mm.default"}]
    if orphan:
        rows.append({"op": "aten.orphan.default"})   # not in distinct_ops -> orphan
    (prof_dir / f"{model}_task.json").write_text(json.dumps({"rows": rows}))


def _m5_dirs(tmp_path, monkeypatch, model):
    inv_dir, prof_dir = tmp_path / "inv", tmp_path / "prof"
    inv_dir.mkdir(); prof_dir.mkdir()
    (tmp_path / "validation" / "reports" / "phase1.1").mkdir(parents=True)
    monkeypatch.setattr(validate_m5_trace, "INV", inv_dir)
    monkeypatch.setattr(validate_m5_trace, "PROF", prof_dir)
    monkeypatch.setattr(validate_m5_trace, "MODELS", [model])
    monkeypatch.setattr(validate_m5_trace, "ROOT", tmp_path)
    return inv_dir, prof_dir


def test_m5_gate_passes_on_good_fixture(tmp_path, monkeypatch):
    model = "fake-model"
    inv_dir, prof_dir = _m5_dirs(tmp_path, monkeypatch, model)
    _write_m5_fixture(inv_dir, prof_dir, model)
    assert validate_m5_trace.main() == 0


def test_m5_gate_fails_on_orphan_op(tmp_path, monkeypatch):
    model = "fake-model"
    inv_dir, prof_dir = _m5_dirs(tmp_path, monkeypatch, model)
    _write_m5_fixture(inv_dir, prof_dir, model, orphan=True)
    assert validate_m5_trace.main() != 0


def test_m5_gate_fails_on_uncovered_inventory(tmp_path, monkeypatch):
    model = "fake-model"
    inv_dir, prof_dir = _m5_dirs(tmp_path, monkeypatch, model)
    _write_m5_fixture(inv_dir, prof_dir, model, covered=False)
    assert validate_m5_trace.main() != 0
