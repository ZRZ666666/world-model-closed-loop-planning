"""Constants and read-only provenance for the isolated Temporal runner."""
from pathlib import Path
import hashlib
import json
import subprocess

BASE = Path(__file__).resolve().parent
WORK = BASE.parent.parent
WM03 = WORK / 'wm03_20261002'
WM09 = WORK / 'wm09_20261002'
REPO = WM09 / 'temporal_straightening_official'
ASSETS = WM09 / 'assets_temporal_straightening'
VARIANT = ASSETS / 'pusht_aggmlpcos1e-1_resnetplus_dim8_hw14_sgTrue_lr1e-5'
CHECKPOINT = VARIANT / 'checkpoints/model_latest.pth'
CONFIG = VARIANT / 'hydra.yaml'
REFERENCE_ROOT = WORK / 'wm12_20261002/next_development/reference_goals_v1'
CACHE_MANIFEST = BASE / 'reference_cache_manifest.json'
CACHE_MANIFEST_SHA = 'a18b60ec73c50141cef141213919694d7b246018d48f3d59bab8bd542d86a8bd'
PROTOCOL = BASE.parent / 'preregistered_question.json'
PROTOCOL_SHA = '18d7be4c348d41c5b86339b180db9e2de27263ca27ccab2462bba091265222c9'
COMMIT = '2c3c7666a69a730042590d548c6731259c4183ac'
CKPT_SHA = '30b34eb854cead76287cb3796db22f9d61f725a658527003d65b3b9124634eaa'
CFG_SHA = '307b43a7bb72ffef7e2e604fd0d5b73b78cf33ed723990fc612738c48d690d05'
COMPONENTS = ('encoder', 'proprio_encoder', 'action_encoder', 'predictor', 'decoder')
ACTION_MEAN = [-0.0087, 0.0068]
ACTION_STD = [0.2019, 0.2002]
PROPRIO_MEAN = [236.6155, 264.5674, -2.93032027, 2.54307914]
PROPRIO_STD = [101.1202, 87.0112, 74.84556075, 74.14009094]


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def jr(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    def default(v):
        if isinstance(v, Path):
            return str(v)
        if hasattr(v, 'tolist'):
            return v.tolist()
        if hasattr(v, 'item'):
            return v.item()
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default), encoding='utf-8')


def provenance():
    """No task/dataset/checkpoint reads; hash source and existing receipts only."""
    assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    assert subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip() == ''
    sources = [BASE / x for x in ('common.py', 'env_worker.py', 'run_temporal_mpc.py', 'reference_cache_manifest.json')]
    sources += [PROTOCOL]
    sources += [REPO / x for x in ('plan.py', 'planning/gd.py', 'planning/mpc.py',
        'planning/objectives.py', 'planning/base_planner.py', 'preprocessor.py', 'utils.py',
        'models/visual_world_model.py', 'models/proprio.py', 'models/vit.py',
        'models/encoder/resnet.py', 'datasets/img_transforms.py', 'datasets/pusht_dset.py',
        'conf/plan_gd_mpc.yaml', 'README.md')]
    sources += [WM03 / 'run_native_pusht.py']
    sources += [WM03 / 'jepa-wms' / p for p in (
        'evals/simu_env_planning/envs/pusht_gym_wrap.py',
        'evals/simu_env_planning/envs/pusht_env/pusht_env.py',
        'evals/simu_env_planning/envs/wrappers/tensor.py',
        'evals/simu_env_planning/envs/wrappers/pixels.py',
        'evals/simu_env_planning/envs/wrappers/time_limit.py')]
    sources += [WORK / 'wm01_20261002/runtime/pymunk' / p for p in ('body.py', 'shapes.py', 'space.py', '_chipmunk.pyd')]
    return {str(p): sha(p) for p in sources}


def cache_entry(reference):
    """Pure manifest metadata. Never deserialize observations, state or actions."""
    reference = Path(reference).resolve(strict=True)
    assert reference.parent == REFERENCE_ROOT.resolve(strict=True)
    assert sha(CACHE_MANIFEST) == CACHE_MANIFEST_SHA
    manifest = jr(CACHE_MANIFEST)
    assert manifest['schema'] == 'wm13-observation-cache-contract-v1'
    assert [row['trajectory_id'] for row in manifest['entries']] == [83,85,99,100,293,294,296,397]
    matches = [row for row in manifest['entries'] if row['task_id'] == reference.name]
    assert len(matches) == 1, 'Reference must be one of the eight frozen development caches'
    return reference, matches[0]


def protocol_entry(reference, moment_factor, seed=1):
    """Validate a member of the prelocked queue without reading state arrays."""
    reference, entry = cache_entry(reference)
    assert sha(PROTOCOL) == PROTOCOL_SHA
    protocol = jr(PROTOCOL)
    assert protocol['schema'] == 'wm13-paired-native-mpc-development-v1'
    assert protocol['targets']['ids'] == [83,85,99,100,293,294,296,397]
    assert len(protocol['queue']) == 16
    matches = [row for row in protocol['queue'] if row['task_id'] == entry['task_id']
        and row['moment_factor'] == moment_factor and row['planner_seed'] == seed]
    assert len(matches) == 1
    row = matches[0]
    assert Path(row['reference_cache']).resolve() == reference
    assert row['env_seed'] == entry['env_seed'] and row['action_gain'] == 1
    for filename in ('planner_observations.npz','reset_metadata.npz','task_realization.npz'):
        field = filename.removesuffix('.npz') + '_sha256'
        assert row[field] == entry['files'][filename]
        # Hash bytes for integrity only; full state arrays are deserialized
        # exclusively by read_environment_reference in the CPU worker.
        assert sha(reference / filename) == row[field]
    for path, digest in protocol['source_receipts'].items():
        assert sha(path) == digest
    assert protocol['model']['checkpoint_sha256'] == CKPT_SHA
    assert protocol['algorithm']['adam_iterations_per_round'] == 100
    assert protocol['algorithm']['max_rounds'] == 20 and protocol['algorithm']['max_raw_actions'] == 100
    return row


def read_planner_reference(reference):
    """Model process reads only cached current/goal RGB and pusher proprio."""
    import numpy as np
    reference, entry = cache_entry(reference)
    path = reference / 'planner_observations.npz'
    assert sha(path) == entry['files'][path.name]
    fields = ('initial_rgb', 'initial_proprio', 'goal_rgb', 'goal_proprio')
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == set(fields)
        realization = {key:data[key].copy() for key in fields}
    for key, value in realization.items():
        if key.endswith('_rgb'):
            assert value.shape == (224,224,3) and value.dtype == np.uint8
        else:
            assert value.shape == (4,) and value.dtype == np.float32 and np.isfinite(value).all()
    task = {key:entry[key] for key in ('task_id','trajectory_id','env_seed','shape')}
    hashes = {str(CACHE_MANIFEST):CACHE_MANIFEST_SHA, str(path):entry['files'][path.name]}
    return task, realization, hashes


def read_environment_reference(reference):
    """CPU worker only: raw reset state and goal evaluation state, no experts."""
    import numpy as np
    reference, entry = cache_entry(reference)
    task, observations, hashes = read_planner_reference(reference)
    metadata = reference / 'reset_metadata.npz'
    evaluation = reference / 'task_realization.npz'
    for path in (metadata,evaluation):
        assert sha(path) == entry['files'][path.name]
        hashes[str(path)] = entry['files'][path.name]
    with np.load(metadata,allow_pickle=False) as data:
        assert set(data.files) == {'initial_state','env_seed','shape','task_id'}
        assert str(data['task_id'].item()) == task['task_id'] and str(data['shape'].item()) == task['shape'] == 'T'
        assert int(data['env_seed'].item()) == task['env_seed']
        task['initial_state'] = data['initial_state'].copy()
    assert task['initial_state'].shape == (7,) and task['initial_state'].dtype == np.float32 and np.isfinite(task['initial_state']).all()
    fields = ('initial_rgb','initial_state','initial_proprio','goal_rgb','goal_state','goal_proprio')
    with np.load(evaluation,allow_pickle=False) as data:
        assert set(data.files) == set(fields)
        realization = {key:data[key].copy() for key in fields}
    assert all(np.array_equal(realization[key],observations[key]) for key in observations)
    assert all(realization[key].shape == (7,) and realization[key].dtype == np.float32 and np.isfinite(realization[key]).all() for key in ('initial_state','goal_state'))
    return task, realization, hashes
