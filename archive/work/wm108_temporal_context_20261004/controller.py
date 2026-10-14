"""One serial WM108 queue. Audit every actual case; retain failures, no retries."""
from pathlib import Path
import argparse
import subprocess
import sys
import traceback
from wm108_contract import B, PROTOCOL, ARMS, sha, jr, save, verify_protocol

C = B/'controller_v1'
A = B/'audit_v1'

def completed(e, protocol_path):
    out, audit = Path(e['output']), A/e['run_name']
    terminal, at = jr(out/'terminal.json'), jr(audit/'terminal.json')
    assert terminal['status'] == at['status'] == 'complete'
    assert sha(out/'summary.json') == terminal['summary_sha256']
    assert sha(audit/'audit.json') == at['audit_sha256']
    for t in (terminal, at):
        for path, h in t.get('output_hashes', {}).items():
            assert sha(path) == h, path
    actual, review = jr(out/'summary.json'), jr(audit/'audit.json')
    assert actual['status'] == 'complete' and review['status'] == 'passed'
    assert actual['arm'] == review['arm'] == e['arm']
    assert actual['draw_index'] == review['draw_index'] == e['draw_index']
    assert actual['protocol_sha256'] == sha(protocol_path)
    assert actual['physical_resets'] == review['resets'] == 1
    count, rounds = actual['physical_actions'], actual['rounds']
    assert isinstance(count, int) and 1 <= rounds <= 20 and count == 5*rounds == review['actual_actions']
    assert actual['success'] == review['success']
    first = review['first_native_success_action']
    assert (first == count) if actual['success'] else (first is None and count == 100)
    pairs = [('actual_summary', out/'summary.json'), ('actual_terminal', out/'terminal.json'),
             ('audit', audit/'audit.json'), ('audit_terminal', audit/'terminal.json')]
    return {**{key: dict(path=str(path), sha256=sha(path)) for key, path in pairs},
            'physical_actions': count, 'physical_resets': 1, 'rounds': rounds, 'success': actual['success']}

def main(args):
    p = verify_protocol(args.protocol)
    if args.resume:
        assert C.is_dir() and not (C/'terminal.json').exists(), 'Terminal queues cannot resume'
    else:
        C.mkdir(exist_ok=False)
    prefix = []
    for e in p['queue']:
        receipt = C/(e['run_name']+'.complete.json')
        if not receipt.exists():
            break
        assert args.resume, 'Explicit --resume required for completed prefix'
        old, live = jr(receipt), completed(e, args.protocol)
        assert old['protocol_sha256'] == sha(args.protocol) and old['receipts'] == live
        assert old['actual_exit_code'] == old['audit_exit_code'] == 0
        assert old['draw_index'] == e['draw_index'] and old['arm'] == e['arm']
        prefix.append((e, live))
    for e in p['queue'][len(prefix):]:
        assert not Path(e['output']).exists() and not (A/e['run_name']).exists(), 'Partial execution retained; no replay'
        assert not list(C.glob(e['run_name']+'.*')), 'Existing dispatch retained; no automatic relaunch'
    n = len(list(C.glob('invocation_*.json')))
    save(C/f'invocation_{n:03d}.json', dict(command=[sys.executable,*sys.argv], protocol_sha256=sha(args.protocol),
         completed_prefix=len(prefix), remaining=[e['run_name'] for e in p['queue'][len(prefix):]]))
    child = None
    current = None
    codes = {}
    def launch(cmd, name, phase):
        nonlocal child
        with (C/f'{name}.{phase}.stdout.log').open('x', encoding='utf-8') as stdout, (C/f'{name}.{phase}.stderr.log').open('x', encoding='utf-8') as stderr:
            child = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, text=True, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            save(C/f'{name}.{phase}.process.json', dict(pid=child.pid, command=cmd))
            code = child.wait()
            codes[phase] = code
            save(C/f'{name}.{phase}.exit.json', dict(pid=child.pid, exit_code=code, captured_from_wait=True))
            child = None
            assert code == 0, (name, phase, code)
    try:
        for e in p['queue'][len(prefix):]:
            if (B/'STOP_AFTER_CURRENT_CASE').exists():
                save(C/f'paused_{n:03d}.json', dict(status='stopped_at_complete_case_audit_boundary', completed_prefix=len(prefix), next_run=e['run_name'], protocol_sha256=sha(args.protocol)))
                print(dict(status='paused', complete_cases=len(prefix)), flush=True)
                return
            verify_protocol(args.protocol)
            current, codes = e['run_name'], {}
            cmd = [p['python'], '-B', str(B/'run_intervention.py'), '--protocol', str(args.protocol), '--run', current, '--output', e['output']]
            audit_cmd = [p['python'], '-B', str(B/'audit_case.py'), '--protocol', str(args.protocol), '--run', current]
            save(C/(current+'.dispatch.json'), dict(draw_index=e['draw_index'], arm=e['arm'], actual_command=cmd, audit_command=audit_cmd, protocol_sha256=sha(args.protocol)))
            launch(cmd, current, 'actual')
            launch(audit_cmd, current, 'audit')
            receipt = completed(e, args.protocol)
            save(C/(current+'.complete.json'), dict(draw_index=e['draw_index'], arm=e['arm'], protocol_sha256=sha(args.protocol), actual_exit_code=codes['actual'], audit_exit_code=codes['audit'], receipts=receipt))
            prefix.append((e, receipt))
            print(dict(event='case_actual_and_audit_complete', run=current, completed=len(prefix)), flush=True)
        assert len(prefix) == 150
        verify_protocol(args.protocol)
        total = sum(r['physical_actions'] for _, r in prefix)
        assert 750 <= total <= 15000
        by_arm = {arm: dict(cases=sum(e['arm']==arm for e,_ in prefix), successes=sum(r['success'] for e,r in prefix if e['arm']==arm), physical_actions=sum(r['physical_actions'] for e,r in prefix if e['arm']==arm)) for arm in ARMS}
        save(C/'terminal.json', dict(status='complete', draws=50, cases=150, physical_actions=total, physical_resets=150,
             protocol_sha256=sha(args.protocol), by_arm=by_arm,
             actual_summaries={e['run_name']:r['actual_summary'] for e,r in prefix}, actual_terminals={e['run_name']:r['actual_terminal'] for e,r in prefix},
             actual_audits={e['run_name']:r['audit'] for e,r in prefix}, audit_terminals={e['run_name']:r['audit_terminal'] for e,r in prefix},
             per_case_accounting={e['run_name']:{k:r[k] for k in ('physical_actions','physical_resets','rounds','success')} for e,r in prefix}, new_targets=0, baseline_replayed=False))
        print(dict(status='complete', cases=150, physical_actions=total, physical_resets=150), flush=True)
    except BaseException as error:
        pid, code = None, None
        if child is not None:
            pid = child.pid
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(timeout=10)
            code = child.returncode
        save(C/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), current_run=current,
             complete_prefix=len(prefix), completed_prefix_physical_actions=sum(r['physical_actions'] for _,r in prefix),
             current_case_budget_requires_saved_worker_review=True, captured_exit_codes=codes,
             cleanup_parent_pid=pid, cleanup_parent_exit=code, descendant_liveness_not_verified=True, no_automatic_retry=True))
        save(C/'terminal.json', dict(status='failed_retained', failure_sha256=sha(C/'failure.json')))
        raise

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=Path, default=PROTOCOL)
    ap.add_argument('--resume', action='store_true')
    main(ap.parse_args())
