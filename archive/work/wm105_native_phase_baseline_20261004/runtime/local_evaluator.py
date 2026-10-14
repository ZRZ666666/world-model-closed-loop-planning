"""Keep native feedback methods; only constructor selects the separate kv worker.

No mutable native-module globals are reassigned. No simulator is initialized
until the root's actual runner instantiates KvPersistentEvaluator.
"""
from pathlib import Path
import subprocess
import sys

NATIVE_RUNNER=Path(__file__).resolve().parent.parent.parent/'wm13_20261002/runner'
sys.path.insert(0,str(NATIVE_RUNNER))
from run_temporal_mpc import PersistentEvaluator
from task_contract import save,sha


class KvPersistentEvaluator(PersistentEvaluator):
    # receive/read_trajectory/assign_init_cond/get_init_cond/eval_actions/close
    # remain inherited. Cumulative prefixes and single live worker are native.
    def __init__(self,output,max_actions,preprocessor,expected_realization,reference_cache,
                 action_gain=1.0,moment_factor=1.0,kv_factor=1.0,protocol=None):
        import numpy as np
        assert max_actions==100 and action_gain==moment_factor==1.0 and kv_factor == 1.0
        self.output,self.preprocessor=output,preprocessor
        self.executed_model=np.empty((1,0,10),dtype=np.float32)
        self.last=None
        self.stderr=(output/'env_stderr.log').open('w',encoding='utf-8')
        worker=Path(__file__).with_name('env_worker.py').resolve()
        command=[sys.executable,str(worker),'--execute','--max-actions',str(max_actions),
                 '--action-gain',str(action_gain),'--moment-factor',str(moment_factor),
                 '--kv-factor',str(kv_factor),'--reference-cache',str(reference_cache),
                 '--output',str(output/'environment'),'--protocol',str(protocol)]
        save(output/'kv_worker_dispatch.json',dict(command=command,worker_path=str(worker),worker_sha256=sha(worker),
            action_gain=action_gain,moment_factor=moment_factor,kv_factor=kv_factor,
            planned_physics='Only pusher k_v=20*factor; k_p100 and original block inertia/mass unchanged.',
            perturbation_parameter_passed_to_world_model=False))
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,
            encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            self.last=self.receive()
            obs,state=self.read_trajectory(self.last['path'])
            for key,ref_key in (('visual','initial_rgb'),('proprio','initial_proprio')):
                assert np.array_equal(obs[key][0,0],expected_realization[ref_key])
            # Hidden block state stays evaluation bookkeeping, never fit input.
            self.obs_0,self.state_0=obs,state[:,0]
            self.initial_obs,self.initial_state=obs,state[:,0]
        except BaseException:
            self.close();raise

    def close(self):
        # Additional process-exit gate; no physics or action changes.
        if self.stderr.closed:return
        try:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write('{"command": "close"}\n');self.process.stdin.flush()
                except (BrokenPipeError,OSError):pass
                try:self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=10)
                    raise RuntimeError('Worker failed to close; process stopped, case retained')
            assert self.process.returncode==0,('Worker nonzero exit',self.process.returncode)
        finally:self.stderr.close()
