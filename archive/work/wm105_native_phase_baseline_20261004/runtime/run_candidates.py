"""WM105 fixed50 serial queue; complete case+audit boundaries only, never automatic retries."""
from runtime_common import *
import argparse,subprocess,sys,traceback

CONTROLLER=B/'candidate_actual_controller_v1';AUDITS=B/'candidate_audit_v1'

def write_once(path,value):
    # The inherited runtime save function overwrites; controller receipts must not.
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)

def completed(entry):
    run=Path(entry['output']);audit=AUDITS/entry['run_name']
    at=jr(run/'terminal.json');au=jr(audit/'terminal.json')
    assert at['status']==au['status']=='complete'
    assert sha(run/'summary.json')==at['summary_sha256'] and sha(audit/'audit.json')==au['audit_sha256']
    for terminal in (at,au):
        for path,h in terminal.get('output_hashes',{}).items():assert sha(path)==h,path
    actual=jr(run/'summary.json');review=jr(audit/'audit.json')
    assert actual['status']=='complete' and review['status']=='passed' and actual['draw_index']==entry['draw_index']
    assert actual['protocol_sha256']==sha(PROTOCOL)
    assert actual['physical_resets']==review['resets']==1
    actions=actual['physical_actions'];rounds=actual['rounds']
    assert isinstance(actions,int) and isinstance(rounds,int) and 1<=rounds<=20 and actions==5*rounds==review['actual_actions']
    assert bool(actual['success'])==bool(review['success'])
    first=review['first_native_success_action']
    assert (first==actions) if actual['success'] else (first is None and actions==100)
    return dict(actual_summary=dict(path=str(run/'summary.json'),sha256=sha(run/'summary.json')),
        actual_terminal=dict(path=str(run/'terminal.json'),sha256=sha(run/'terminal.json')),
        audit=dict(path=str(audit/'audit.json'),sha256=sha(audit/'audit.json')),
        audit_terminal=dict(path=str(audit/'terminal.json'),sha256=sha(audit/'terminal.json')),
        physical_actions=actions,physical_resets=1,rounds=rounds,success=bool(actual['success']))

def main(args):
    protocol=verify_protocol(args.protocol)
    if args.resume:
        assert CONTROLLER.is_dir() and not (CONTROLLER/'terminal.json').exists(), 'Failed/completed controller cannot be automatically resumed'
    else:CONTROLLER.mkdir(exist_ok=False)
    prefix=[]
    for entry in protocol['queue']:
        done=CONTROLLER/(entry['run_name']+'.complete.json')
        if not done.exists():break
        assert args.resume, 'Existing completion requires explicit --resume'
        old=jr(done);live=completed(entry)
        assert old['draw_index']==entry['draw_index'] and old['actual_exit_code']==old['audit_exit_code']==0
        assert old['receipts']==live and old['protocol_sha256']==sha(args.protocol)
        prefix.append((entry,live))
    for entry in protocol['queue'][len(prefix):]:
        assert not Path(entry['output']).exists() and not (AUDITS/entry['run_name']).exists(), 'Partial case/audit retained; no automatic replay'
        assert not list(CONTROLLER.glob(entry['run_name']+'.*')), 'Existing dispatch/log retained; no automatic relaunch'
    invocation=len(list(CONTROLLER.glob('invocation_*.json')))
    write_once(CONTROLLER/f'invocation_{invocation:03d}.json',dict(command=[sys.executable,*sys.argv],protocol_sha256=sha(args.protocol),
        completed_prefix=len(prefix),remaining=[e['draw_index'] for e in protocol['queue'][len(prefix):]],skip_complete_not_replayed=True))
    child=None;current=None;codes={}
    def launch(cmd,name,phase):
        nonlocal child
        with (CONTROLLER/f'{name}.{phase}.stdout.log').open('x',encoding='utf-8') as stdout,(CONTROLLER/f'{name}.{phase}.stderr.log').open('x',encoding='utf-8') as stderr:
            child=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            write_once(CONTROLLER/f'{name}.{phase}.process.json',dict(pid=child.pid,command=cmd))
            code=child.wait();codes[phase]=code
            write_once(CONTROLLER/f'{name}.{phase}.exit.json',dict(pid=child.pid,exit_code=code,captured_from_wait=True))
            child=None;assert code==0,(phase,code)
    try:
        for entry in protocol['queue'][len(prefix):]:
            if (ROOT/'STOP_AFTER_CURRENT_DRAW').exists():
                write_once(CONTROLLER/f'paused_{invocation:03d}.json',dict(status='stopped_at_complete_case_audit_boundary',completed_prefix=len(prefix),
                    next_draw=entry['draw_index'],protocol_sha256=sha(args.protocol),physical_actions_completed=sum(r['physical_actions'] for _,r in prefix)))
                print(dict(status='paused',complete_cases=len(prefix)),flush=True);return
            verify_protocol(args.protocol);current=entry['draw_index'];codes={};name=entry['run_name']
            actual=[protocol['python'],str(B/'run_frozen.py'),'--protocol',str(args.protocol),'--run',name,'--output',entry['output']]
            audit=[protocol['python'],str(B/'audit_case.py'),'--protocol',str(args.protocol),'--run',name]
            write_once(CONTROLLER/(name+'.dispatch.json'),dict(draw_index=current,actual_command=actual,audit_command=audit,protocol_sha256=sha(args.protocol)))
            launch(actual,name,'actual');launch(audit,name,'audit');receipts=completed(entry)
            write_once(CONTROLLER/(name+'.complete.json'),dict(draw_index=current,protocol_sha256=sha(args.protocol),actual_exit_code=codes['actual'],audit_exit_code=codes['audit'],receipts=receipts))
            prefix.append((entry,receipts));print(dict(event='case_actual_and_audit_complete',draw_index=current,completed=len(prefix)),flush=True)
        assert len(prefix)==50;verify_protocol(args.protocol)
        actions=sum(r['physical_actions'] for _,r in prefix);resets=sum(r['physical_resets'] for _,r in prefix)
        assert resets==50 and 250<=actions<=5000
        write_once(CONTROLLER/'terminal.json',dict(status='complete',draws=50,registered_cells=50,physical_resets=resets,physical_actions=actions,
            successes=sum(r['success'] for _,r in prefix),rounds=sum(r['rounds'] for _,r in prefix),protocol_sha256=sha(args.protocol),
            actual_summaries={e['run_name']:r['actual_summary'] for e,r in prefix},actual_terminals={e['run_name']:r['actual_terminal'] for e,r in prefix},
            actual_audits={e['run_name']:r['audit'] for e,r in prefix},audit_terminals={e['run_name']:r['audit_terminal'] for e,r in prefix},
            per_case_accounting={e['run_name']:{k:r[k] for k in ('physical_actions','physical_resets','rounds','success')} for e,r in prefix},
            historical_goal_resets=50,historical_goal_raw_actions=1250,new_goal_resets=0,new_goal_raw_actions=0))
        print(dict(status='complete',draws=50,physical_actions=actions,physical_resets=resets),flush=True)
    except BaseException as error:
        pid=None;parent_exit=None
        if child is not None:
            pid=child.pid
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
            parent_exit=child.returncode
        write_once(CONTROLLER/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),draw_index=current,complete_prefix=len(prefix),
            completed_prefix_physical_actions=sum(r['physical_actions'] for _,r in prefix),current_case_budget_requires_saved_worker_review=True,
            captured_exit_codes=codes,cleanup_parent_pid=pid,cleanup_parent_exit=parent_exit,descendant_liveness_not_verified=True,no_automatic_retry=True))
        write_once(CONTROLLER/'terminal.json',dict(status='failed_retained',failure_sha256=sha(CONTROLLER/'failure.json')));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,default=PROTOCOL);parser.add_argument('--resume',action='store_true');main(parser.parse_args())
