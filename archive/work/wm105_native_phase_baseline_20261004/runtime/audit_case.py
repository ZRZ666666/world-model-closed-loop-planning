"""WM105 saved-only native MPC audit. No Torch, encoder, model or environment."""
from pathlib import Path
import argparse,csv,hashlib,json,os,sys,traceback
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
import numpy as np
from runtime_common import verify_protocol
B=Path(__file__).resolve().parent;ROOT=B.parent;W=ROOT.parent
AM=np.array([-.0087,.0068],np.float32);AS=np.array([.2019,.2002],np.float32)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def jr(p):return json.loads(Path(p).read_text('utf-8-sig'))
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def geometry(s,g):
    da=s[4]-g[4]
    r=dict(joint_position_error=float(np.linalg.norm(s[:4]-g[:4])),angle_error=float(abs(np.arctan2(np.sin(da),np.cos(da)))),pusher_error=float(np.linalg.norm(s[:2]-g[:2])),block_error=float(np.linalg.norm(s[2:4]-g[2:4])))
    r['success']=bool(r['joint_position_error']<20 and r['angle_error']<np.pi/9);return r
def main(args):
    p=verify_protocol(args.protocol);matches=[e for e in p['queue'] if e['run_name']==args.run];assert len(matches)==1
    e=matches[0];run=Path(e['output']);out=B/'candidate_audit_v1'/args.run;out.mkdir(parents=True,exist_ok=False)
    pins={};checks=[]
    def check(name,ok):checks.append(dict(name=name,passed=bool(ok)));assert ok,name
    def pin(path,expected=None):
        path=str(Path(path));h=sha(path);pins[path]=h
        if expected is not None:check('hash:'+path,h==expected)
        return h
    def read(path):pin(path);return jr(path)
    def arr(path,fields=None):
        pin(path)
        with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if fields is None else fields)}
    def finite(name,a,shape,dtype=np.float32):check(name,a.shape==shape and a.dtype==dtype and np.isfinite(a).all())
    def stable(name,path,receipt):
        z=arr(path);keys=('cpu','cuda','numpy_keys','numpy_meta','python')
        check(name+':rng_fields',set(z)=={k+s for k in keys for s in ('_before','_after')})
        for k in keys:
            check(name+':rng:'+k,bits(z[k+'_before'],z[k+'_after']))
            for side in ('before','after'):check(name+':rng_hash:'+k+side,hashlib.sha256(z[k+'_'+side].tobytes()).hexdigest()==receipt['rng'][side+'_sha256'][k])
        check(name+':rng_receipt',receipt['rng']['all_equal'] and all(receipt['rng']['per_source_equal'].values()) and receipt['rng']['file_sha256']==sha(path))
        check(name+':model_frozen',receipt['before_model_state_sha256']==receipt['after_model_state_sha256']==p['expected_model_state_sha256'])
    try:
        pin(__file__,p['source_hashes'][str(Path(__file__).resolve())]);pin(args.protocol);pin(B/'root_execution_gate.json')
        s=read(run/'summary.json');t=read(run/'terminal.json')
        check('actual_complete',s['status']==t['status']=='complete' and t['summary_sha256']==sha(run/'summary.json'))
        check('case_identity',s['protocol_sha256']==sha(args.protocol) and s['arm']==e['arm']=='native_frozen' and s['dataset']=='val' and s['draw_index']==e['draw_index'] and s['trajectory_id']==e['trajectory_id'] and s['offset']==e['offset'] and s['planner_seed']==100 and s['env_seed']==e['env_seed'])
        pre=read(run/'pre_run_lock.json');check('prelock',pre['protocol_sha256']==sha(args.protocol) and pre['source_hashes']==p['source_hashes'] and pre['entry']==e)
        R=int(s['rounds']);N=int(s['physical_actions']);check('round_budget',1<=R<=20 and N==5*R and s['physical_resets']==1)
        check('exact_round_inventory',[x.name for x in sorted(run.glob('round_*'))]==[f'round_{r:02d}' for r in range(R)])
        check('exact_feedback_inventory',[x.name for x in sorted(run.glob('feedback_*.json'))]==[f'feedback_{r:02d}.json' for r in range(R)])
        env=run/'environment';worker=read(env/'worker_terminal.json')
        check('worker_budget',worker['status']=='closed' and worker['physical_actions']==N and worker['simulator_resets']==1 and worker['gpu_initialized'] is False and not (env/'failure.json').exists())
        check('all_raw_durable',[x.name for x in sorted(env.glob('step_*.npz'))]==[f'step_{n:03d}.npz' for n in range(1,N+1)])
        small=('state','proprio','action','action_command','action_applied')
        actual=arr(env/f'step_{N:03d}.npz',small);state=actual['state'];q=actual['proprio'];u=actual['action_command']
        finite('state',state,(N+1,7));finite('proprio',q,(N+1,4))
        for k in small[2:]:finite(k,actual[k],(N,2))
        check('commands_applied',bits(u,actual['action_applied']) and bits(u,actual['action']))
        check('proprio_layout',bits(q,state[:,[0,1,5,6]]))
        for n in range(1,N):
            z=arr(env/f'step_{n:03d}.npz',small)
            check(f'raw{n}:cumulative',all(bits(z[k],actual[k][:n+(1 if k in ('state','proprio') else 0)]) for k in small))
        initial=arr(env/'initial.npz');refdir=Path(e['reference_cache'])
        for name,h in e['files'].items():pin(refdir/name,h)
        ref=arr(refdir/'task_realization.npz');obs=arr(refdir/'planner_observations.npz');reset=arr(refdir/'reset_metadata.npz')
        check('raw_reset_contract',reset['initial_state'].shape==(7,) and reset['initial_state'].dtype==np.float32 and np.isfinite(reset['initial_state']).all() and int(reset['env_seed'])==e['env_seed'] and str(reset['task_id'].item())==e['task_id'])
        check('observed_initial_cache',bits(initial['rgb'][0],ref['initial_rgb']) and bits(state[0],initial['state'][0]) and bits(state[0],ref['initial_state']) and bits(q[0],ref['initial_proprio']))
        check('planner_cache_identity',all(bits(obs[k],ref[k]) for k in obs))
        goal=ref['goal_state'];check('goal_proprio_layout',bits(ref['goal_proprio'],goal[[0,1,5,6]]))
        for path,h in worker['reference_hashes'].items():pin(path,h)
        phy=read(env/'physics_receipt.json');pin(env/'physics_receipt.json',worker['physics_receipt_sha256'])
        check('nominal_physics',phy['before']==phy['after'] and phy['before']['k_p']==100 and phy['before']['k_v']==20 and phy['kv_factor']==phy['moment_factor']==phy['action_gain']==1 and phy['post_setter_initial_rgb_state_proprio_bitwise_equal'])
        check('frozen_model',s['base_state_before']==s['base_state_after']==p['expected_model_state_sha256'] and s['base_updates']==0 and s['new_targets_generated_in_run']==0)
        load=read(run/'load_receipt.json');check('loaded_frozen',load['parameters_frozen'] and load['state_unchanged'])
        selected=arr(run/'selected_actions.npz');taken=selected['actions_model'];finite('selected_actions',taken,(1,R,10))
        rows=[dict(actions=0,**geometry(state[0],goal))];alignment=[];previous=None;first=None;goal_codes=None;cached_current=[]
        for r in range(R):
            start=5*r;end=start+5;rd=run/f'round_{r:02d}'
            inp=arr(rd/'input_observations.npz');opt=arr(rd/'optimization.npz',('final_actions_model','goal_visual','goal_proprio'))
            final=opt['final_actions_model'];finite(f'r{r}:returned',final,(1,5,10))
            check(f'r{r}:executed_post100_token',bits(final[0,0].reshape(5,2)*AS+AM,u[start:end]) and bits(taken[:,r:r+1],final[:,:1]))
            expected_memo=np.empty((1,0,10),np.float32) if previous is None else previous[:,1:]
            check(f'r{r}:own_memo',bits(inp['memo_actions'],expected_memo));previous=final
            rgb=initial['rgb'][-1:] if r==0 else arr(env/f'step_{start:03d}.npz',('rgb',))['rgb'][-1:]
            check(f'r{r}:current_obs',bits(inp['current_rgb'],rgb[None]) and bits(inp['current_proprio'],q[start][None,None]))
            check(f'r{r}:unchanged_goal',bits(inp['goal_rgb'],ref['goal_rgb'][None,None]) and bits(inp['goal_proprio'],ref['goal_proprio'][None,None]))
            if goal_codes is None:goal_codes=(opt['goal_visual'],opt['goal_proprio'])
            check(f'r{r}:fixed_goal_codes',bits(opt['goal_visual'],goal_codes[0]) and bits(opt['goal_proprio'],goal_codes[1]))
            f=read(run/f'feedback_{r:02d}.json');g=geometry(state[end],goal);rows.append(dict(actions=end,**g))
            check(f'r{r}:native_geometry',g['success']==f['success'] and all(g[k]==f[k] for k in ('joint_position_error','angle_error','pusher_error','block_error')))
            check(f'r{r}:feedback',f['actions']==end and f['new_actions']==5 and f['simulator_resets']==1 and f['gpu_initialized'] is False and Path(f['path']).resolve()==(env/f'step_{end:03d}.npz').resolve())
            if g['success'] and first is None:first=end
            check(f'r{r}:no_execution_after_success',r==R-1 or not g['success'])
            cc=read(rd/'counts.json')['delta']
            for k,v in dict(planning_predict_calls=500,planning_context_frames=1200,planning_encode_calls=101,planning_encoded_frames=101,planning_action_encode_calls=500,planning_action_encoded_tokens=500,action_backward_calls=100,parameter_updates=0).items():check(f'r{r}:planning_count:'+k,cc[k]==v)
            losses=read(rd/'losses.json');check(f'r{r}:GD100_losses',len(losses)==100 and np.isfinite(losses).all())
            post=arr(rd/'post100_prediction.npz');receipt=read(rd/'post100_receipt.json');stable(f'r{r}:post100',rd/'post100_rng.npz',receipt)
            check(f'r{r}:post100_identity',bits(post['returned_H5_model_actions'],final) and bits(post['current_rgb'],inp['current_rgb']) and bits(post['current_proprio'],inp['current_proprio']) and receipt['round']==r and receipt['step']==r and receipt['returned_actions_byte_equal'] and receipt['current_observation_unchanged'] and receipt['returned_is_post100_not_argmin'])
            finite(f'r{r}:visualforecast',post['full_visual_prediction'],(1,6,196,8));finite(f'r{r}:proprioforecast',post['full_proprio_prediction'],(1,6,10))
            dd=receipt['counts_delta']
            for k,v in dict(post100_rollout_calls=1,post100_predict_calls=5,post100_context_frames=12,post100_encode_calls=1,post100_encoded_frames=1,post100_action_encode_calls=5,post100_action_encoded_tokens=5,diagnostic_state_digest_calls=2).items():check(f'r{r}:post100_count:'+k,dd[k]==v)
            check(f'r{r}:diagnostic_no_control_work',all(dd[k]==0 for k in dd if k.startswith('planning_') or k in ('action_backward_calls','parameter_updates','physical_actions')) and receipt['backward_calls']==0)
            cached_current.append((post['current_rgb'],post['current_proprio']))
            alignment.append(dict(round=r,start_raw=start,end_raw=end,prediction_file=str(rd/'post100_prediction.npz'),predicted_future_slot=1,actual_trajectory=str(env/f'step_{end:03d}.npz'),actual_raw_index=end,actual_code_file=str(run/f'round_{r+1:02d}/post100_prediction.npz') if r+1<R else str(run/'terminal_observation_encoding.npz'),actual_code_kind='next_current_slot0' if r+1<R else 'direct_terminal_encode',no_truth_for_future_slots_2_to_5=True))
        # Endpoints reuse only a next-round CURRENT code, never a future prediction.
        for r in range(R-1):
            endpoint=arr(env/f'step_{5*(r+1):03d}.npz',('rgb',))['rgb'][-1:]
            check(f'r{r}:actual_endpoint_next_current',bits(endpoint[None],cached_current[r+1][0]) and bits(q[5*(r+1)][None,None],cached_current[r+1][1]))
        terminal_code=arr(run/'terminal_observation_encoding.npz');last_rgb=arr(env/f'step_{N:03d}.npz',('rgb',))['rgb'][-1:]
        check('terminal_encoder_raw_identity',bits(terminal_code['raw_rgb'],last_rgb[None]) and bits(terminal_code['raw_proprio'],q[-1][None,None]))
        finite('terminal_visual_code',terminal_code['visual'],(1,1,196,8));finite('terminal_proprio_code',terminal_code['proprio'],(1,1,10))
        tr=read(run/'terminal_encoding_receipt.json');stable('terminal_encoding',run/'terminal_encoding_rng.npz',tr)
        dd=tr['counts_delta'];check('terminal_encoding_counts',dd['terminal_encode_calls']==dd['terminal_encoded_frames']==1 and dd['diagnostic_state_digest_calls']==2 and all(dd[k]==0 for k in dd if k.startswith(('planning_','post100_')) or k in ('action_backward_calls','parameter_updates','physical_actions')))
        check('terminal_encoding_feedback',tr['actual_actions']==N and Path(tr['source_feedback_path']).resolve()==(env/f'step_{N:03d}.npz').resolve())
        expected=dict(model_loads=1,planning_predict_calls=500*R,planning_context_frames=1200*R,planning_encode_calls=101*R,planning_encoded_frames=101*R,planning_action_encode_calls=500*R,planning_action_encoded_tokens=500*R,action_backward_calls=100*R,parameter_updates=0,physical_actions=N,post100_rollout_calls=R,post100_predict_calls=5*R,post100_context_frames=12*R,post100_encode_calls=R,post100_encoded_frames=R,post100_action_encode_calls=5*R,post100_action_encoded_tokens=5*R,terminal_encode_calls=1,terminal_encoded_frames=1,diagnostic_state_digest_calls=2*R+2,base_state_digest_calls=2)
        for k,v in expected.items():check('aggregate_count:'+k,s['counts'][k]==v)
        check('native_stop',s['success']==(first is not None) and (first==N if s['success'] else N==100))
        check('selected_length',selected['action_lengths'].shape==(1,) and (float(selected['action_lengths'][0])==R if s['success'] else np.isinf(selected['action_lengths'][0])))
        check('last_feedback',s['metrics']==read(run/f'feedback_{R-1:02d}.json'))
        # Optional historical alignment is reported, not used to choose or change a run.
        old_comparison=None
        if e['draw_index']<8:
            od=W/'wm106_candidate_attribution_20261004/runtime/execution_v1'/f"draw_{e['draw_index']:03d}"
            if (od/'terminal.json').is_file():
                ot=read(od/'terminal.json');check('WM106_historical_complete',ot['status']=='complete_candidate_attribution_draw')
                op=od/'native_zero/optimization.npz';pin(op,ot['output_hashes'][str(op)])
                of=arr(op,('final_actions_model',))['final_actions_model'];nf=arr(run/'round_00/optimization.npz',('final_actions_model',))['final_actions_model']
                old_comparison=dict(historical_output=str(od),round0_returned_H5_byte_equal=bits(of,nf),descriptive_only_not_algorithm_change_gate=True)
        for name,data in [('native_boundaries.csv',rows),('factual_alignment.csv',alignment)]:
            with (out/name).open('x',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        audit=dict(status='passed',run_name=args.run,draw_index=e['draw_index'],trajectory_id=e['trajectory_id'],offset=e['offset'],initial_group_id=e['initial_group_id'],trajectory_group_id=e['trajectory_group_id'],actual_actions=N,resets=1,simulator_resets=1,rounds=R,success=s['success'],first_native_success_action=first,initial_geometry_success=rows[0]['success'],initial_success_is_descriptive_not_stop=True,final_geometry=rows[-1],counts=s['counts'],factual_blocks=R,post100_future_slot1_only=True,terminal_code_direct_proprio_not_asserted_equal_to_rollout_separate_emb=True,historical_WM106=old_comparison,checks=len(checks),audit_model_calls=0,audit_encoder_calls=0,audit_environment_calls=0)
        save(out/'audit.json',audit);save(out/'checks.json',checks);save(out/'input_hashes.json',pins)
        save(out/'terminal.json',dict(status='complete',audit_sha256=sha(out/'audit.json'),output_hashes={str(x):sha(x) for x in out.iterdir() if x.is_file()}))
        print(json.dumps(dict(status='passed',run=args.run,checks=len(checks))))
    except BaseException as err:
        save(out/'failure.json',dict(error=repr(err),traceback=traceback.format_exc(),completed_checks=len(checks)));save(out/'input_hashes.json',pins)
        save(out/'terminal.json',dict(status='failed_retained',failure_sha256=sha(out/'failure.json'),no_automatic_retry=True));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',required=True);main(ap.parse_args())
