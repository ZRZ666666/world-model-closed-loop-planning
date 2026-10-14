"""WM105 draw-keyed cache adapter; planner never reads expert or block state."""
from pathlib import Path
from runtime_common import B,W,ROOT,PROTOCOL,GOALS,sha,jr,save,verify_protocol
WM03=W/'wm03_20261002';MANIFEST=GOALS;REFERENCE_ROOT=ROOT/'data/references_v1'
def cache_entry(reference,protocol=PROTOCOL):
    p=verify_protocol(protocol);reference=Path(reference).resolve(strict=True)
    assert reference.parent==REFERENCE_ROOT.resolve(strict=True)
    manifest=jr(MANIFEST)
    assert manifest['schema']=='wm105-consumed-random-phase-reference-v1' and manifest['status']=='complete_all50_retained'
    assert manifest['dataset']=='val' and manifest['tasks']==len(manifest['entries'])==50
    matches=[e for e in manifest['entries'] if Path(e['reference_cache']).resolve()==reference];assert len(matches)==1
    e=matches[0];assert e['dataset']=='val' and e['env_seed']==100*e['draw_index']+1 and e['planner_seed']==100 and e['shape']=='T'
    assert e['horizon']==25 and 0<=e['draw_index']<50 and 0<=e['trajectory_id']<=20
    match=[x for x in p['queue'] if x['task_id']==e['task_id']];assert len(match)==1
    assert all(match[0][k]==e[k] for k in e)
    assert sha(reference/'terminal.json')==e['goal_terminal_sha256']
    return reference,e
def protocol_entry(reference,moment_factor,seed=100,kv_factor=1.,protocol=PROTOCOL):
    reference,e=cache_entry(reference,protocol);p=verify_protocol(protocol)
    matches=[c for c in p['queue'] if c['task_id']==e['task_id']];assert len(matches)==1
    c=matches[0];assert seed==c['planner_seed']==100 and moment_factor==c['moment_factor']==1 and c['action_gain']==kv_factor==c['kv_factor']==1
    for name,h in e['files'].items():assert sha(reference/name)==h,name
    return c

# Reader implementations are appended mechanically from WM56 by derive_runtime.py.

def read_planner_reference(reference,protocol=PROTOCOL):
    import numpy as np
    reference,entry=cache_entry(reference,protocol)
    path=reference/'planner_observations.npz'
    assert sha(path)==entry['files'][path.name]
    fields=('initial_rgb','initial_proprio','goal_rgb','goal_proprio')
    with np.load(path,allow_pickle=False) as z:
        assert set(z.files)==set(fields)
        arrays={k:z[k].copy() for k in fields}
    for key,a in arrays.items():
        if key.endswith('_rgb'):assert a.shape==(224,224,3) and a.dtype==np.uint8
        else:assert a.shape==(4,) and a.dtype==np.float32 and np.isfinite(a).all()
    task={k:entry[k] for k in ('dataset','draw_index','task_id','trajectory_id','offset','initial_group_id','trajectory_group_id','env_seed','planner_seed','shape')}
    return task,arrays,{str(MANIFEST):sha(MANIFEST),str(path):entry['files'][path.name]}

def read_environment_reference(reference,protocol=PROTOCOL):
    import numpy as np
    reference,entry=cache_entry(reference,protocol)
    task,observations,hashes=read_planner_reference(reference,protocol)
    metadata=reference/'reset_metadata.npz';evaluation=reference/'task_realization.npz'
    for path in (metadata,evaluation):
        assert sha(path)==entry['files'][path.name]
        hashes[str(path)]=entry['files'][path.name]
    with np.load(metadata,allow_pickle=False) as z:
        assert set(z.files)=={'initial_state','env_seed','shape','task_id'}
        assert str(z['task_id'].item())==task['task_id'] and str(z['shape'].item())==task['shape']=='T'
        assert int(z['env_seed'].item())==task['env_seed']
        task['initial_state']=z['initial_state'].copy()
    assert task['initial_state'].shape==(7,) and task['initial_state'].dtype==np.float32 and np.isfinite(task['initial_state']).all()
    fields=('initial_rgb','initial_state','initial_proprio','goal_rgb','goal_state','goal_proprio')
    with np.load(evaluation,allow_pickle=False) as z:
        assert set(z.files)==set(fields)
        arrays={k:z[k].copy() for k in fields}
    assert all(np.array_equal(arrays[k],observations[k]) for k in observations)
    # Raw reset input is distinct from observed post-reset state: the native
    # _set_state performs one 0.01s physics step. The worker prepares this raw
    # input and compares its actual RGB/state/proprio to cached observations.
    for k in ('initial_state','goal_state'):assert arrays[k].shape==(7,) and arrays[k].dtype==np.float32 and np.isfinite(arrays[k]).all()
    return task,arrays,hashes
