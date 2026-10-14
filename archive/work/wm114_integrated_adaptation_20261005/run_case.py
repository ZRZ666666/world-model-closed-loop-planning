"""WM114: consumed-development B/BR/BG/BRG integration. Serial GPU owner only."""
from pathlib import Path
from collections import Counter
from contextlib import nullcontext
import argparse, hashlib, json, os, random, sys, time, traceback, types

B=Path(__file__).resolve().parent
W=B.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(W/'wm13_20261002/runner'))
sys.path.insert(0,str(W/'wm105_native_phase_baseline_20261004/runtime'))
sys.path.insert(0,str(W/'wm29_20261003'))
from task_contract import read_planner_reference
BASE_PROTOCOL=W/'wm105_native_phase_baseline_20261004/runtime/candidate_protocol.json'
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def jr(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def verify(pth):
    p=jr(pth)
    assert p['schema']=='wm114-integrated-development-v1'
    for k,v in p['source_hashes'].items():assert sha(k)==v,k
    assert sha(BASE_PROTOCOL)==p['baseline_protocol_sha256']
    return p

def main(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();evaluator=None;phase='load';active_round=None
    counts=Counter();planning_times=[];training_times=[];monitor_times=[]
    try:
        p=verify(a.protocol);entry=next(e for e in p['queue'] if e['run_name']==a.run)
        assert out==Path(entry['output']).resolve()
        arm=entry['arm'];use_r='R' in arm;use_g='G' in arm
        save(out/'pre_run_lock.json',dict(protocol_sha256=sha(a.protocol),entry=entry,command=[sys.executable,*sys.argv],source_hashes=p['source_hashes']))
        import numpy as np
        import torch
        from common import ACTION_MEAN,ACTION_STD,PROPRIO_MEAN,PROPRIO_STD
        from run_temporal_mpc import load_model,state_digest,verify_plain_tensors,LocalLog
        from local_evaluator import KvPersistentEvaluator
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        seed=entry['planner_seed'];assert seed==100
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
        task,realization,refs=read_planner_reference(Path(entry['reference_cache']),BASE_PROTOCOL)
        save(out/'reference_receipt.json',dict(task=task,source_hashes=refs,expert_arrays=0))
        model,config=load_model(out,device);counts['model_loads']=1
        before=state_digest(model);assert before==p['expected_model_state_sha256']
        from preprocessor import Preprocessor
        from datasets.img_transforms import default_transform
        from planning.objectives import create_objective_fn
        from planning.mpc import MPCPlanner
        preprocessor=Preprocessor(action_mean=torch.tensor(ACTION_MEAN,dtype=torch.float32),action_std=torch.tensor(ACTION_STD,dtype=torch.float32),state_mean=None,state_std=None,
            proprio_mean=torch.tensor(PROPRIO_MEAN,dtype=torch.float32),proprio_std=torch.tensor(PROPRIO_STD,dtype=torch.float32),transform=default_transform(224))
        native_predict,native_encode,native_act=model.predict,model.encode_obs,model.encode_act
        def predict(self,z):
            counts[phase+'_predict_calls']+=1;counts[phase+'_context_frames']+=int(z.shape[0]*z.shape[1]);return native_predict(z)
        def encode(self,obs):
            counts[phase+'_encode_calls']+=1;counts[phase+'_encoded_frames']+=int(obs['visual'].shape[0]*obs['visual'].shape[1]);return native_encode(obs)
        def encode_act(self,act):
            counts[phase+'_action_encode_calls']+=1;counts[phase+'_action_tokens']+=int(act.shape[0]*act.shape[1]);return native_act(act)
        model.predict=types.MethodType(predict,model);model.encode_obs=types.MethodType(encode,model);model.encode_act=types.MethodType(encode_act,model)
        adapter=None
        if use_r:
            from sandwich_core import SandwichCore
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                adapter=SandwichCore(device=device)
                # Deliberate local variant: exact identity initially, removing pre-learning changes.
                with torch.no_grad():
                    for branch in (adapter.visual_in,adapter.visual_out):branch[-1].weight.zero_();branch[-1].bias.zero_()
                adapter._initial_state={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
            adapter.install(model)
            np.savez_compressed(out/'adapter_initial.npz',**{k:v.detach().cpu().numpy() for k,v in adapter.state_dict().items()})
        def disabled():return adapter.disabled() if adapter is not None else nullcontext()
        def trans(obs):return {k:v.to(device) for k,v in preprocessor.transform_obs({k:x.copy() for k,x in obs.items()}).items()}
        def arrays(z):return {k:v.detach().cpu().numpy().copy() for k,v in z.items()}
        def byte_equal(x,y):return x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
        def rms18(x,y):
            return float(torch.sqrt((8*(x['visual']-y['visual']).square().mean()+10*(x['proprio']-y['proprio']).square().mean())/18))
        evaluator=KvPersistentEvaluator(out,100,preprocessor,realization,reference_cache=Path(entry['reference_cache']),protocol=BASE_PROTOCOL)
        goal=dict(visual=realization['goal_rgb'][None,None],proprio=realization['goal_proprio'][None,None])
        objective=create_objective_fn(alpha=1,base=2,mode='staged');last_objective=create_objective_fn(alpha=1,base=2,mode='last')
        native_rollout=model.rollout
        history_count=1;past=None;cache=None;pending=None;last_gate=None
        replan_rounds=[];reuse_rounds=[];proposals=[];losses=[]
        def objective_at(pred,target,step):
            if step<5:
                q=5-step;return last_objective({k:v[:,q:q+1] for k,v in pred.items()},target,step)
            return objective(pred,target,step)
        def variant(obs,act):
            assert obs['visual'].shape[1]==history_count and past.shape==(1,history_count-1,10)
            joined=act if history_count==1 else torch.cat((past,act),dim=1)
            full,z=native_rollout(obs,joined)
            crop=history_count-1
            return ({k:v[:,crop:] for k,v in full.items()} if crop else full),z[:,crop:]
        def rollout(self,obs_0,act):
            assert phase=='planning' and act.shape==(1,5,10) and act.requires_grad
            proposals.append(act.detach().cpu().numpy().copy());return variant(obs_0,act)
        def score(pred,target,step=None):
            assert phase=='planning' and pred['visual'].shape[1]==6
            loss=objective_at(pred,target,int(step));assert torch.isfinite(loss).all()
            def grad(g):counts['action_backward_calls']+=1;assert torch.isfinite(g).all();return g
            loss.register_hook(grad);losses.append(float(loss.detach()));return loss
        model.rollout=types.MethodType(rollout,model)
        planner=MPCPlanner(max_iter=20,n_taken_actions=1,sub_planner=dict(target='planning.gd.GDPlanner',lr=.1,sample_type='zero',action_noise=0,horizon=5,var_scale=1,opt_steps=100,eval_every=-1),
            wm=model,env=None,action_dim=10,objective_fn=score,preprocessor=preprocessor,evaluator=evaluator,wandb_run=LocalLog(out),log_filename=str(out/'official_mpc_logs.jsonl'))
        native_plan=planner.sub_planner.plan;native_evaluate=evaluator.eval_actions
        def plan(self,obs_0,obs_g,actions=None,step=None):
            nonlocal phase,active_round,history_count,past,cache,pending
            active_round=int(step);r=active_round;assert int(evaluator.last['actions'])==5*r and pending is None
            history_count=min(3,r+1);first=r-history_count+1;indices=np.arange(first,r+1)*5
            source=Path(evaluator.last['path'])
            with np.load(source,allow_pickle=False) as z:
                hist=dict(visual=z['rgb'][indices][None].copy(),proprio=z['proprio'][indices][None].copy())
            assert all(np.array_equal(hist[k][:,-1:],obs_0[k]) for k in hist)
            past_np=evaluator.executed_model[:,first:r].copy();past=torch.from_numpy(past_np).to(device).detach()
            planning_obs=obs_0 if history_count==1 else hist
            rd=out/f'round_{r:02d}';rd.mkdir()
            np.savez_compressed(rd/'history_inputs.npz',history_rgb=hist['visual'],history_proprio=hist['proprio'],raw_action_indices=indices,past_actions_model=past_np,
                memo_actions=np.empty((1,0,10),np.float32) if actions is None else actions.detach().cpu().numpy())
            age=r-cache['origin'] if cache is not None else None
            reuse=bool(use_g and cache is not None and actions is not None and actions.shape[1]>0 and age<=p['gate']['max_reuses'] and last_gate['ratio']<=p['gate']['relative_error_threshold'])
            reason='reuse' if reuse else ('no_cache' if cache is None else 'disabled' if not use_g else 'age_cap' if age>p['gate']['max_reuses'] else 'error_threshold')
            gate=dict(round=r,raw_time=5*r,use_gate=use_g,reuse=reuse,reason=reason,cache_origin_before=None if cache is None else cache['origin'],age_before=age,
                valid_suffix_before=0 if actions is None else int(actions.shape[1]),feedback=last_gate,adapter_version=0 if adapter is None else adapter.optimizer_steps,
                deadline_slot=5-r if r<5 else None,history_frames=history_count)
            save(rd/'decision.json',gate)
            if reuse:
                phase='reuse';result=actions.detach().clone();lengths=np.array([np.inf]);reuse_rounds.append(r)
                expected=cache['actions'][:,age:]
                assert np.array_equal(result.cpu().numpy(),expected) and result.shape[1]==5-age
            else:
                phase='planning';proposals.clear();losses.clear();t=time.perf_counter()
                result,lengths=native_plan(obs_0=planning_obs,obs_g=obs_g,actions=actions,step=r)
                torch.cuda.synchronize();planning_times.append(time.perf_counter()-t);replan_rounds.append(r)
                assert len(losses)==len(proposals)==100
                np.savez_compressed(rd/'optimization.npz',proposals_before_update=np.stack(proposals),final_actions_model=result.detach().cpu().numpy())
                save(rd/'losses.json',losses)
            result=result.detach()
            np.savez_compressed(rd/'selected_plan.npz',actions_model=result.cpu().numpy())
            t=time.perf_counter();phase='monitor_plan';th=trans(planning_obs)
            if not reuse:
                with torch.no_grad(),disabled():frozen,_=variant(th,result)
                cache=dict(origin=r,actions=result.cpu().numpy().copy(),frozen={k:v.detach().clone() for k,v in frozen.items()})
                np.savez_compressed(rd/'frozen_origin_prediction.npz',**arrays(frozen))
            phase='prequential_prediction'
            with torch.no_grad():adapted,_=variant(th,result[:,:1])
            next_prediction={k:v[:,1:2].detach().clone() for k,v in adapted.items()}
            np.savez_compressed(rd/'prequential_prediction.npz',**arrays(next_prediction))
            torch.cuda.synchronize();monitor_times.append(time.perf_counter()-t)
            pending=dict(round=r,rd=rd,current={k:v[:,-1:].detach().clone() for k,v in th.items()},command=result[:,:1].clone(),prediction=next_prediction,version=0 if adapter is None else adapter.optimizer_steps)
            if r==0:
                old=Path(entry['historical_B'])/'round_00/optimization.npz'
                with np.load(old,allow_pickle=False) as z:same=byte_equal(z['final_actions_model'],result.cpu().numpy())
                save(out/'first_plan_identity.json',dict(status='passed' if same else 'failed_retained',historical_path=str(old),historical_sha256=sha(old),exact_identity_initialization=True,actions_byte_equal=same))
                assert same,'First plan differs from historical B before any training or reuse'
            return result,lengths
        def evaluate(self,actions,action_len=None,filename='output',save_video=False):
            nonlocal phase,pending,last_gate
            assert pending is not None
            logs,successes,obs,states=native_evaluate(actions,action_len,filename,save_video)
            r=pending['round'];rd=pending['rd'];end=int(evaluator.last['actions']);counts['physical_actions']=end;assert end==5*(r+1)
            t=time.perf_counter();nxt=trans({k:v[:,-1:].copy() for k,v in obs.items()});phase='monitor_actual'
            with torch.no_grad(),disabled():actual=model.encode_obs(nxt)
            slot=r+1-cache['origin'];assert 1<=slot<=5
            origin={k:v[:,:1] for k,v in cache['frozen'].items()};pred={k:v[:,slot:slot+1] for k,v in cache['frozen'].items()}
            d=rms18(actual,pred);motion=rms18(pred,origin);scale=max(motion,p['gate']['scale_floor'])
            last_gate=dict(d=d,predicted_motion=motion,scale=scale,ratio=d/scale,cache_origin=cache['origin'],slot=slot,monitor_coordinates='frozen_original_18channel_RMS',prediction_before_execution=True)
            np.savez_compressed(rd/'actual_encoding.npz',**arrays(actual))
            save(rd/'gate_feedback.json',last_gate)
            phase='prequential_actual'
            with torch.no_grad():actual_adapted=model.encode_obs(nxt)
            pre=dict(adapter_version=pending['version'],rms18=rms18(actual_adapted,pending['prediction']),scored_before_fit=True,coordinates='same_adapter_version_for_prediction_and_realized_encoding')
            np.savez_compressed(rd/'adapted_actual_encoding.npz',**arrays(actual_adapted));save(rd/'prequential_error.json',pre)
            torch.cuda.synchronize();monitor_times.append(time.perf_counter()-t)
            terminal=bool(successes[0]) or end==100
            if adapter is not None and not terminal:
                phase='training';t=time.perf_counter()
                with torch.random.fork_rng(devices=[0]):fit=adapter.fit_completed(r,5*r,end,pending['current'],nxt,pending['command'])
                torch.cuda.synchronize();training_times.append(time.perf_counter()-t)
                counts['residual_optimizer_steps']+=fit['Adam_steps_this_fit'];counts['residual_training_backward_calls']+=fit['backward_calls_this_fit']
                fit.update(initial_identity=True,initialization='WM114 exact-zero final visual layers; all linear residuals zero',terminal_unused_fit=False)
                save(rd/'training_receipt.json',fit)
                torch.save(dict(adapter=adapter.state_dict(),optimizer=adapter.optimizer.state_dict()),rd/'adapter_optimizer.pt')
                assert all(v.grad is None and not v.requires_grad for v in model.parameters())
            save(rd/'completed.json',dict(round=r,physical_actions=end,success=bool(successes[0]),replanned=r in replan_rounds,adapter_updates=0 if adapter is None else adapter.optimizer_steps,counts=dict(counts)))
            pending=None;phase='between_rounds';return logs,successes,obs,states
        planner.sub_planner.plan=types.MethodType(plan,planner.sub_planner);evaluator.eval_actions=types.MethodType(evaluate,evaluator)
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        np.savez_compressed(out/'rng_before_plan.npz',cpu=torch.get_rng_state().numpy(),cuda=torch.cuda.get_rng_state(device).cpu().numpy())
        result,lengths=planner.plan(evaluator.initial_obs,goal)
        np.savez_compressed(out/'selected_actions.npz',actions_model=result.detach().cpu().numpy(),action_lengths=lengths)
        assert pending is None and counts['physical_actions']==5*planner.iter
        assert counts['action_backward_calls']==100*len(replan_rounds) and counts['planning_predict_calls']==500*len(replan_rounds)
        after=state_digest(model);assert before==after and all(v.grad is None and not v.requires_grad for v in model.parameters())
        verify_plain_tensors(model,out,device);evaluator.close()
        worker=jr(out/'environment/worker_terminal.json');assert worker['simulator_resets']==1 and worker['physical_actions']==counts['physical_actions']
        if arm=='B':
            old=Path(entry['historical_B']);checks={};regression_hashes={}
            for name in ('selected_actions.npz',f'environment/step_{counts["physical_actions"]:03d}.npz'):
                with np.load(old/name,allow_pickle=False) as x,np.load(out/name,allow_pickle=False) as y:
                    checks[name]=set(x.files)==set(y.files) and all(byte_equal(x[k],y[k]) for k in x.files)
                regression_hashes[name]=dict(historical_sha256=sha(old/name),actual_sha256=sha(out/name))
            s0=jr(old/'summary.json');checks['metrics']=s0['success']==bool(planner.is_success[0]) and s0['physical_actions']==counts['physical_actions']
            save(out/'baseline_regression.json',dict(status='passed' if all(checks.values()) else 'failed_retained',checks=checks,file_hashes=regression_hashes,historical_path=str(old),historical_summary_sha256=sha(old/'summary.json')))
            assert all(checks.values()),checks
        verify(a.protocol)
        s=dict(status='complete',arm=arm,draw_index=entry['draw_index'],trajectory_group_id=entry['trajectory_group_id'],task=task,
            success=bool(planner.is_success[0]),physical_actions=counts['physical_actions'],rounds=planner.iter,metrics=evaluator.last,counts=dict(counts),
            planning_total_wall_seconds=sum(planning_times),training_total_wall_seconds=sum(training_times),monitor_total_wall_seconds=sum(monitor_times),
            replan_rounds=len(replan_rounds),reuse_rounds=len(reuse_rounds),replan_round_indices=replan_rounds,reuse_round_indices=reuse_rounds,residual_updates=0 if adapter is None else adapter.optimizer_steps,
            trainable_parameters=0 if adapter is None else sum(v.numel() for v in adapter.parameters()),frozen_state_unchanged=before==after,base_state_before=before,base_state_after=after,
            protocol_sha256=sha(a.protocol),physical_resets=1,new_targets=0,independent_confirmation=False,wall_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        save(out/'summary.json',s);save(out/'terminal.json',dict(status='complete',summary_sha256=sha(out/'summary.json')))
        print(json.dumps(dict(event='complete',run=a.run,success=s['success'],actions=s['physical_actions'],replans=len(replan_rounds),reuse=len(reuse_rounds),updates=s['residual_updates'])),flush=True)
    except BaseException as e:
        save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),phase=phase,round=active_round,counts=dict(counts)))
        save(out/'terminal.json',dict(status='failed_retained',failure_sha256=sha(out/'failure.json')));raise
    finally:
        if evaluator is not None:evaluator.close()

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--protocol',type=Path,required=True);q.add_argument('--run',required=True);q.add_argument('--output',type=Path,required=True);main(q.parse_args())
