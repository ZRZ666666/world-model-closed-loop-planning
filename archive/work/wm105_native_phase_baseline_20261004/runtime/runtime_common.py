"""WM105 native-only runtime contract. Importing never loads model or data arrays."""
from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parent;ROOT=B.parent;W=ROOT.parent
PROTOCOL=B/'candidate_protocol.json';GOALS=ROOT/'data/goal_receipt.json'
GATE=B/'root_execution_gate.json';SCIENCE=ROOT/'scientific_question.txt'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def jr(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    def default(x):
        if isinstance(x,Path):return str(x)
        if hasattr(x,'tolist'):return x.tolist()
        if hasattr(x,'item'):return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=default)+'\n',encoding='utf-8')
def verify_protocol(path=PROTOCOL,require_gate=True):
    path=Path(path).resolve();assert path==PROTOCOL.resolve()
    p=jr(path)
    assert p['schema']=='wm105-native-random-phase-v1' and p['dataset']=='val'
    assert p['draws']==50 and len(p['queue'])==50
    assert p['goal_receipt_sha256']==sha(GOALS) and p['scientific_question_sha256']==sha(SCIENCE)
    for source,h in p['source_hashes'].items():assert sha(source)==h,source
    assert [e['draw_index'] for e in p['queue']]==list(range(50))
    assert len({e['run_name'] for e in p['queue']})==50
    for e in p['queue']:
        assert e['dataset']=='val' and 0<=e['trajectory_id']<=20 and e['offset']>=0 and e['horizon']==25
        assert e['env_seed']==100*e['draw_index']+1 and e['planner_seed']==100
        assert e['arm']=='native_frozen' and e['kv_factor']==e['action_gain']==e['moment_factor']==1
        assert Path(e['output']).resolve()==(B/'execution_v1'/e['run_name']).resolve()
        assert Path(e['reference_cache']).resolve()==(ROOT/'data/references_v1'/e['task_id']).resolve()
    if require_gate:
        g=jr(GATE)
        assert g['status']=='passed' and g['root_authorized_execution'] is True
        assert g['protocol_sha256']==sha(path) and g['source_hashes']==p['source_hashes']
    return p
