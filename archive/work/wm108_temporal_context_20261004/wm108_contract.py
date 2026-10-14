"""Frozen WM108 metadata contract. Stdlib only; no model or physics imports."""
from pathlib import Path
import hashlib
import json

B = Path(__file__).resolve().parent
BASE = B.parent / 'wm105_native_phase_baseline_20261004'
BASE_PROTOCOL = BASE / 'runtime/candidate_protocol.json'
PROTOCOL = B / 'candidate_protocol.json'
GATE = B / 'root_execution_gate.json'
ARMS = ('deadline', 'history3', 'history3_deadline')
BASE_PROTOCOL_SHA = 'b446474d47f84f88c236b5405e69851e09888e6ec43231212bf3d7648c664361'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def jr(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def verify_protocol(path=PROTOCOL, require_gate=True):
    path = Path(path)
    p = jr(path)
    assert p['schema'] == 'wm108-temporal-context-v1'
    assert p['status'] == 'prepared_for_root_review'
    assert p['draws'] == 50 and p['cases'] == 150 and p['arms'] == list(ARMS)
    assert sha(BASE_PROTOCOL) == BASE_PROTOCOL_SHA == p['baseline_protocol_sha256']
    assert p['scientific_lock_sha256'] == sha(B/'scientific_lock.txt')
    baseline = jr(BASE_PROTOCOL)
    expected = [(i, arm) for i in range(50) for arm in ARMS]
    assert [(e['draw_index'], e['arm']) for e in p['queue']] == expected
    for e in p['queue']:
        ref = baseline['queue'][e['draw_index']]
        for key in ('dataset', 'task_id', 'trajectory_id', 'offset', 'horizon', 'initial_group_id',
                    'trajectory_group_id', 'env_seed', 'planner_seed', 'reference_cache', 'files'):
            assert e[key] == ref[key], (e['run_name'], key)
        assert e['run_name'] == f"draw_{e['draw_index']:03d}_{e['arm']}"
        assert Path(e['output']).resolve() == (B/'execution_v1'/e['run_name']).resolve()
        assert e['baseline_output'] == ref['output']
        assert e['history_frames'] == (3 if 'history3' in e['arm'] else 1)
        assert e['deadline'] == ('deadline' in e['arm'])
        assert e['kv_factor'] == e['action_gain'] == e['moment_factor'] == 1.0
    for section in ('source_hashes', 'input_hashes'):
        for source, digest in p[section].items():
            assert sha(source) == digest, (section, source)
    if require_gate:
        gate = jr(GATE)
        assert gate['status'] == 'passed' and gate['root_authorized_execution'] is True
        assert gate['protocol_sha256'] == sha(path)
        assert gate['source_hashes'] == p['source_hashes']
        assert gate['input_hashes'] == p['input_hashes']
    return p
