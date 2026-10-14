"""Serial existing-development experiment queue. Stops and retains every failure."""
from pathlib import Path
import argparse,datetime,json,subprocess,sys,traceback
from run_case import jr,save,sha,verify,B

def main(a):
    p=verify(a.protocol);out=a.output.resolve();out.mkdir(exist_ok=False)
    current=None;completed=[]
    try:
        save(out/'started.json',dict(time=datetime.datetime.now().astimezone().isoformat(),protocol_sha256=sha(a.protocol),command=[sys.executable,*sys.argv]))
        for e in p['queue']:
            current=e['run_name'];case=Path(e['output'])
            if case.exists():
                assert a.resume_completed,'Existing case requires explicit completed-prefix resume'
                t=jr(case/'terminal.json');s=jr(case/'summary.json')
                assert t['status']=='complete' and t['summary_sha256']==sha(case/'summary.json')
                assert s['protocol_sha256']==sha(a.protocol) and s['status']=='complete'
                completed.append(current);continue
            if (B/'STOP_AFTER_CURRENT_CASE').exists():
                save(out/'terminal.json',dict(status='paused_at_case_boundary',completed=completed));return
            command=[p['python'],'-B',str(B/'run_case.py'),'--protocol',str(a.protocol),'--run',current,'--output',str(case)]
            save(out/f'{current}_dispatch.json',dict(command=command))
            with (out/f'{current}.stdout.log').open('w',encoding='utf-8') as stdout,(out/f'{current}.stderr.log').open('w',encoding='utf-8') as stderr:
                child=subprocess.Popen(command,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                save(out/f'{current}_process.json',dict(pid=child.pid,command=command))
                code=child.wait()
            save(out/f'{current}_exit.json',dict(exit_code=code))
            assert code==0,(current,code)
            t=jr(case/'terminal.json');s=jr(case/'summary.json')
            assert t['status']=='complete' and t['summary_sha256']==sha(case/'summary.json') and s['protocol_sha256']==sha(a.protocol)
            assert s['frozen_state_unchanged'] and s['physical_actions']<=100 and s['physical_resets']==1
            if e['arm']=='B':assert jr(case/'baseline_regression.json')['status']=='passed'
            completed.append(current)
            save(out/f'{current}_verified.json',dict(summary_sha256=sha(case/'summary.json'),completed=len(completed)))
            print(json.dumps(dict(completed=len(completed),total=len(p['queue']),run=current,success=s['success'],actions=s['physical_actions'])),flush=True)
        save(out/'terminal.json',dict(status='complete',cases=len(completed),protocol_sha256=sha(a.protocol)))
        command=[p['python'],'-B',str(B/'saved_analysis.py'),'--protocol',str(a.protocol),'--output',str(out/'final_analysis')]
        with (out/'analysis.stdout.log').open('w',encoding='utf-8') as so,(out/'analysis.stderr.log').open('w',encoding='utf-8') as se:
            r=subprocess.run(command,stdout=so,stderr=se,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        save(out/'analysis_exit.json',dict(command=command,exit_code=r.returncode))
        assert r.returncode==0,'Saved analysis failed; experiments retained'
    except BaseException as e:
        save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),current=current,completed=completed))
        if not (out/'terminal.json').exists():save(out/'terminal.json',dict(status='failed_retained',failure_sha256=sha(out/'failure.json')))
        raise

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--protocol',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--resume-completed',action='store_true');main(q.parse_args())
