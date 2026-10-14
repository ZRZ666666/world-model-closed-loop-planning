"""WM108 saved-only history/deadline audit, derived from WM105. No Torch/E/environment."""
from pathlib import Path
import argparse,csv,hashlib,json,os,sys,traceback
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
import numpy as np
from wm108_contract import B,BASE_PROTOCOL,verify_protocol
ROOT=B;W=B.parent
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
def saved_scores(visual,proprio,goal_visual,goal_proprio,round_index,deadline):
    # Batch can be100 logged candidates or1 post100 forecast. Native all mode
    # normalizes float32 powers, then takes another temporal mean, including t0.
    frame=np.mean((visual.astype(np.float64)-goal_visual.astype(np.float64))**2,axis=(-2,-1))
    frame+=np.mean((proprio.astype(np.float64)-goal_proprio.astype(np.float64))**2,axis=-1)
    assert frame.shape[-1]==6 and np.isfinite(frame).all()
    if round_index<5:return frame[...,5-round_index if deadline else 5]
    powers=np.array([2**i for i in range(6)],np.float32);weights=powers/np.sum(powers)
    return np.mean(frame*weights,axis=-1)
def main(args):
    p=verify_protocol(args.protocol);matches=[e for e in p['queue'] if e['run_name']==args.run];assert len(matches)==1
    e=matches[0];run=Path(e['output']);out=B/'audit_v1'/args.run;out.mkdir(parents=True,exist_ok=False)
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
        check('case_identity',s['protocol_sha256']==sha(args.protocol) and s['arm']==e['arm'] and e['arm'] in ('history3','deadline','history3_deadline') and s['dataset']=='val' and s['draw_index']==e['draw_index'] and s['trajectory_id']==e['trajectory_id'] and s['offset']==e['offset'] and s['planner_seed']==100 and s['env_seed']==e['env_seed'])
        pre=read(run/'pre_run_lock.json');check('prelock',pre['protocol_sha256']==sha(args.protocol) and pre['source_hashes']==p['source_hashes'] and pre['entry']==e)
        history_frames=int(e['history_frames']);deadline=bool(e['deadline'])
        check('variant_identity',(history_frames,deadline)=={'history3':(3,False),'deadline':(1,True),'history3_deadline':(3,True)}[e['arm']] and s['history_frames']==history_frames and s['deadline']==deadline)
        base=jr(BASE_PROTOCOL);be=base['queue'][e['draw_index']];baseline=Path(e['baseline_output'])
        pin(BASE_PROTOCOL);check('baseline_entry',baseline.resolve()==Path(be['output']).resolve() and all(e[k]==be[k] for k in ('dataset','task_id','trajectory_id','offset','initial_group_id','trajectory_group_id','env_seed','planner_seed','files')))
        bt=read(baseline/'terminal.json');check('baseline_complete',bt['status']=='complete');pin(baseline/'summary.json',bt['summary_sha256'])
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
        old_initial=arr(baseline/'environment/initial.npz')
        check('baseline_same_initial',all(bits(initial[k],old_initial[k]) for k in ('rgb','state','proprio')))
        first5=arr(env/'step_005.npz',('rgb',*small));old_first5=arr(baseline/'environment/step_005.npz',('rgb',*small))
        check('baseline_same_first5',all(bits(first5[k],old_first5[k]) for k in first5))
        ri=read(run/'first_round_identity.json');fi=read(run/'first5_identity.json')
        check('producer_first_identity',ri['status']==fi['status']=='passed' and ri['final_H5_byte_equal'] and ri['planner_inputs_byte_equal'] and ri['baseline_read_only_after_returned_plan_saved'] and all(fi['per_field_byte_equal'].values()))
        check('producer_full_preaction_identity',ri['all_optimization_fields_byte_equal'] and ri['initial_rng_byte_equal'] and ri['losses_json_exact_equal'] and ri['no_numerical_tolerance'] and ri['before_first_actual_action'] and ri['loss_count_baseline']==ri['loss_count_actual']==len(ri['loss_value_exact_equal'])==100 and all(ri['loss_value_exact_equal']))
        for stem in ('actual_plan','baseline_rng','actual_rng','baseline_losses','actual_losses'):pin(ri[stem+'_path'],ri[stem+'_sha256'])
        pin(ri['baseline_plan_path'],ri['baseline_plan_sha256']);pin(ri['baseline_input_path'],ri['baseline_input_sha256'])
        pin(fi['baseline_path'],fi['baseline_sha256']);pin(fi['actual_path'],fi['actual_sha256'])
        old_rng=arr(baseline/'rng_before_plan.npz');new_rng=arr(run/'rng_before_plan.npz')
        check('baseline_same_initial_rng',set(old_rng)==set(new_rng) and all(bits(new_rng[k],old_rng[k]) for k in new_rng))
        rows=[dict(actions=0,**geometry(state[0],goal))];alignment=[];history_rows=[];previous=None;first=None;goal_codes=None;cached_current=[]
        for r in range(R):
            start=5*r;end=start+5;rd=run/f'round_{r:02d}';h=min(history_frames,r+1);context=sum(min(h+i,3) for i in range(5))
            inp=arr(rd/'input_observations.npz');opt=arr(rd/'optimization.npz')
            final=opt['final_actions_model'];finite(f'r{r}:returned',final,(1,5,10))
            check(f'r{r}:executed_post100_token',bits(final[0,0].reshape(5,2)*AS+AM,u[start:end]) and bits(taken[:,r:r+1],final[:,:1]))
            expected_memo=np.empty((1,0,10),np.float32) if previous is None else previous[:,1:]
            check(f'r{r}:own_memo',bits(inp['memo_actions'],expected_memo));previous=final
            rgb=initial['rgb'][-1:] if r==0 else arr(env/f'step_{start:03d}.npz',('rgb',))['rgb'][-1:]
            check(f'r{r}:current_obs',bits(inp['current_rgb'],rgb[None]) and bits(inp['current_proprio'],q[start][None,None]))
            check(f'r{r}:unchanged_goal',bits(inp['goal_rgb'],ref['goal_rgb'][None,None]) and bits(inp['goal_proprio'],ref['goal_proprio'][None,None]))
            hist=arr(rd/'history_inputs.npz');hr=read(rd/'history_receipt.json')
            history_start=r-h+1;indices=np.arange(history_start,r+1,dtype=np.int64)*5
            expected_source=env/'initial.npz' if r==0 else env/f'step_{start:03d}.npz'
            source_history=arr(expected_source,('rgb','proprio'))
            finite(f'r{r}:history_rgb',hist['history_rgb'],(1,h,224,224,3),np.uint8)
            finite(f'r{r}:history_q',hist['history_proprio'],(1,h,4))
            finite(f'r{r}:past_tokens',hist['past_actions_model'],(1,h-1,10))
            finite(f'r{r}:executed_prefix',hist['executed_model_prefix'],(1,r,10))
            check(f'r{r}:actual_history_indices',bits(hist['raw_action_indices'],indices))
            check(f'r{r}:actual_own_history',bits(hist['history_rgb'],source_history['rgb'][indices][None]) and bits(hist['history_proprio'],q[indices][None]) and bits(source_history['proprio'],q[:start+1]))
            check(f'r{r}:history_current_alignment',bits(hist['history_rgb'][:,-1:],inp['current_rgb']) and bits(hist['history_proprio'][:,-1:],inp['current_proprio']))
            check(f'r{r}:past_token_command_alignment',bits(hist['executed_model_prefix'],taken[:,:r]) and bits(hist['past_actions_model'],taken[:,history_start:r]))
            if h>1:check(f'r{r}:past_tokens_are_actual_raw_commands',bits(hist['past_actions_model'].reshape(-1,2)*AS+AM,u[5*history_start:start]))
            check(f'r{r}:history_metadata',hr['history_frames']==h and hr['configured_history_frames']==history_frames and hr['current_raw_action']==start and hr['history_start_macro']==history_start and hr['raw_action_indices']==indices.tolist() and hr['past_token_start']==history_start and hr['past_token_end_exclusive']==r)
            check(f'r{r}:history_permissions',hr['past_tokens_are_commands_not_displacements'] and hr['past_tokens_detached'] and hr['future_optimized_tokens']==5 and hr['no_source_episode_prehistory'])
            check(f'r{r}:history_source',Path(hr['source_feedback_path']).resolve()==expected_source.resolve() and hr['source_feedback_sha256']==sha(expected_source))
            slot=5-r if deadline and r<5 else None
            check(f'r{r}:history_objective_metadata',hr['deadline']==deadline and hr['score_slot']==slot and hr['crop_start']==h-1 and hr['score_sequence_length']==6 and hr['original_staged_step']==r and hr['objective_mode']==('deadline_single_slot' if deadline and r<5 else 'native_staged'))
            if goal_codes is None:goal_codes=(opt['goal_visual'],opt['goal_proprio'])
            check(f'r{r}:fixed_goal_codes',bits(opt['goal_visual'],goal_codes[0]) and bits(opt['goal_proprio'],goal_codes[1]))
            finite(f'r{r}:future_candidate_bank',opt['proposals_before_update'],(100,1,5,10))
            finite(f'r{r}:cropped_visual_bank',opt['visual_predictions'],(100,1,6,196,8))
            finite(f'r{r}:cropped_proprio_bank',opt['proprio_predictions'],(100,1,6,10))
            finite(f'r{r}:goal_visual',opt['goal_visual'],(1,1,196,8));finite(f'r{r}:goal_proprio',opt['goal_proprio'],(1,1,10))
            if r==0:
                old_input=arr(baseline/'round_00/input_observations.npz');old_opt=arr(baseline/'round_00/optimization.npz')
                check('baseline_same_first_inputs',set(inp)==set(old_input) and all(bits(inp[k],old_input[k]) for k in inp))
                check('baseline_same_first100_bank_and_H5',set(opt)==set(old_opt) and all(bits(opt[k],old_opt[k]) for k in opt))
            f=read(run/f'feedback_{r:02d}.json');g=geometry(state[end],goal);rows.append(dict(actions=end,**g))
            check(f'r{r}:native_geometry',g['success']==f['success'] and all(g[k]==f[k] for k in ('joint_position_error','angle_error','pusher_error','block_error')))
            check(f'r{r}:feedback',f['actions']==end and f['new_actions']==5 and f['simulator_resets']==1 and f['gpu_initialized'] is False and Path(f['path']).resolve()==(env/f'step_{end:03d}.npz').resolve())
            if g['success'] and first is None:first=end
            check(f'r{r}:no_execution_after_success',r==R-1 or not g['success'])
            cc=read(rd/'counts.json')['delta']
            for k,v in dict(planning_predict_calls=500,planning_context_frames=100*context,planning_encode_calls=101,planning_encoded_frames=100*h+1,planning_action_encode_calls=500,planning_action_encoded_tokens=100*(h+4),action_backward_calls=100,parameter_updates=0).items():check(f'r{r}:planning_count:'+k,cc[k]==v)
            losses=read(rd/'losses.json');check(f'r{r}:GD100_losses',len(losses)==100 and np.isfinite(losses).all())
            recomputed=saved_scores(opt['visual_predictions'],opt['proprio_predictions'],*goal_codes,r,deadline).reshape(100)
            check(f'r{r}:independent100_objective',np.allclose(np.asarray(losses,np.float64),recomputed,atol=1e-6,rtol=1e-5))
            if r==0:check('baseline_same_first100_losses',losses==read(baseline/'round_00/losses.json'))
            post=arr(rd/'post100_prediction.npz');receipt=read(rd/'post100_receipt.json');stable(f'r{r}:post100',rd/'post100_rng.npz',receipt)
            check(f'r{r}:post100_identity',bits(post['returned_H5_model_actions'],final) and bits(post['current_rgb'],inp['current_rgb']) and bits(post['current_proprio'],inp['current_proprio']) and receipt['round']==r and receipt['step']==r and receipt['returned_actions_byte_equal'] and receipt['current_observation_unchanged'] and receipt['returned_is_post100_not_argmin'])
            finite(f'r{r}:visualforecast',post['full_visual_prediction'],(1,6,196,8));finite(f'r{r}:proprioforecast',post['full_proprio_prediction'],(1,6,10))
            finite(f'r{r}:uncropped_visual',post['uncropped_visual_prediction'],(1,h+5,196,8));finite(f'r{r}:uncropped_proprio',post['uncropped_proprio_prediction'],(1,h+5,10))
            check(f'r{r}:crop_actual_current_plus_future5',bits(post['full_visual_prediction'],post['uncropped_visual_prediction'][:,h-1:]) and bits(post['full_proprio_prediction'],post['uncropped_proprio_prediction'][:,h-1:]))
            check(f'r{r}:post100_same_history',bits(post['history_rgb'],hist['history_rgb']) and bits(post['history_proprio'],hist['history_proprio']) and bits(post['past_actions_model'],hist['past_actions_model']))
            joined=np.concatenate((hist['past_actions_model'],final),axis=1)
            check(f'r{r}:past_plus_future_action_alignment',bits(post['full_H_history_future_model_actions'],joined))
            check(f'r{r}:post100_history_receipt',receipt['past_actions_byte_equal'] and receipt['history_frames']==h and receipt['crop_start']==h-1 and receipt['slot1_only_has_actual_first5_truth'] and receipt['suffix_truth_available'] is False)
            obj=receipt['objective']
            check(f'r{r}:post100_objective',obj==dict(mode='deadline_single_slot' if deadline and r<5 else 'staged',alpha=1,base=2,step=r,score_sequence_length=6,score_slot=slot))
            recalculated=float(saved_scores(post['full_visual_prediction'],post['full_proprio_prediction'],*goal_codes,r,deadline).reshape(-1)[0])
            check(f'r{r}:independent_post100_objective',np.isclose(recalculated,receipt['returned_loss'],atol=1e-6,rtol=1e-5))
            check(f'r{r}:loss_receipt',receipt['initial_logged_loss']==losses[0] and receipt['last_logged_pre_update_loss']==losses[-1] and receipt['minimum_logged_pre_update_loss']==min(losses) and receipt['minimum_logged_index']==int(np.argmin(losses)))
            dd=receipt['counts_delta']
            for k,v in dict(post100_rollout_calls=1,post100_predict_calls=5,post100_context_frames=context,post100_encode_calls=1,post100_encoded_frames=h,post100_action_encode_calls=5,post100_action_encoded_tokens=h+4,diagnostic_state_digest_calls=2).items():check(f'r{r}:post100_count:'+k,dd[k]==v)
            check(f'r{r}:diagnostic_no_control_work',all(dd[k]==0 for k in dd if k.startswith('planning_') or k in ('action_backward_calls','parameter_updates','physical_actions')) and receipt['backward_calls']==0)
            cached_current.append((post['current_rgb'],post['current_proprio']))
            history_rows.append(dict(round=r,current_raw_action=start,history_frames=h,history_start_raw=int(indices[0]),history_raw_indices=json.dumps(indices.tolist()),past_token_count=h-1,future_token_count=5,crop_start=h-1,scoring_mode='deadline_single_slot' if deadline and r<5 else ('native_last' if r<5 else 'native_all'),scored_slot=slot if slot is not None else (5 if r<5 else None),planning_predict_calls=500,planning_context_frames=100*context,planning_encoded_frames=100*h+1,planning_action_encoded_tokens=100*(h+4),post100_context_frames=context,post100_encoded_frames=h,post100_action_encoded_tokens=h+4,max_abs_logged_loss_recompute_error=float(np.max(np.abs(recomputed-np.asarray(losses)))),post100_loss_recompute_error=recalculated-receipt['returned_loss']))
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
        expected=dict(model_loads=1,planning_predict_calls=500*R,planning_context_frames=sum(100*sum(min(min(history_frames,r+1)+i,3) for i in range(5)) for r in range(R)),planning_encode_calls=101*R,planning_encoded_frames=sum(100*min(history_frames,r+1)+1 for r in range(R)),planning_action_encode_calls=500*R,planning_action_encoded_tokens=sum(100*(min(history_frames,r+1)+4) for r in range(R)),action_backward_calls=100*R,parameter_updates=0,physical_actions=N,post100_rollout_calls=R,post100_predict_calls=5*R,post100_context_frames=sum(sum(min(min(history_frames,r+1)+i,3) for i in range(5)) for r in range(R)),post100_encode_calls=R,post100_encoded_frames=sum(min(history_frames,r+1) for r in range(R)),post100_action_encode_calls=5*R,post100_action_encoded_tokens=sum(min(history_frames,r+1)+4 for r in range(R)),terminal_encode_calls=1,terminal_encoded_frames=1,diagnostic_state_digest_calls=2*R+2,base_state_digest_calls=2)
        for k,v in expected.items():check('aggregate_count:'+k,s['counts'][k]==v)
        check('native_stop',s['success']==(first is not None) and (first==N if s['success'] else N==100))
        check('selected_length',selected['action_lengths'].shape==(1,) and (float(selected['action_lengths'][0])==R if s['success'] else np.isinf(selected['action_lengths'][0])))
        check('last_feedback',s['metrics']==read(run/f'feedback_{R-1:02d}.json'))
        for name,data in [('native_boundaries.csv',rows),('factual_alignment.csv',alignment),('history_alignment.csv',history_rows)]:
            with (out/name).open('x',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        audit=dict(status='passed',run_name=args.run,draw_index=e['draw_index'],arm=e['arm'],history_frames=history_frames,deadline=deadline,trajectory_id=e['trajectory_id'],offset=e['offset'],initial_group_id=e['initial_group_id'],trajectory_group_id=e['trajectory_group_id'],actual_actions=N,resets=1,simulator_resets=1,rounds=R,success=s['success'],first_native_success_action=first,initial_geometry_success=rows[0]['success'],initial_success_is_descriptive_not_stop=True,final_geometry=rows[-1],counts=s['counts'],factual_blocks=R,post100_future_slot1_only=True,terminal_code_direct_proprio_not_asserted_equal_to_rollout_separate_emb=True,same_initial_and_first5_bytes=True,same_original_first_H5=True,checks=len(checks),audit_model_calls=0,audit_encoder_calls=0,audit_environment_calls=0)
        save(out/'audit.json',audit);save(out/'checks.json',checks);save(out/'input_hashes.json',pins)
        save(out/'terminal.json',dict(status='complete',audit_sha256=sha(out/'audit.json'),output_hashes={str(x):sha(x) for x in out.iterdir() if x.is_file()}))
        print(json.dumps(dict(status='passed',run=args.run,checks=len(checks))))
    except BaseException as err:
        save(out/'failure.json',dict(error=repr(err),traceback=traceback.format_exc(),completed_checks=len(checks),terminal_already_written=(out/'terminal.json').exists()))
        if not (out/'input_hashes.json').exists():save(out/'input_hashes.json',pins)
        if not (out/'terminal.json').exists():save(out/'terminal.json',dict(status='failed_retained',failure_sha256=sha(out/'failure.json'),no_automatic_retry=True))
        raise
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',required=True);main(ap.parse_args())
