#!/usr/bin/env bash
# Finish a pod that has completed its run: pull results, verify the expected number of
# final runs per method in its journal, DELETE the pod via the RunPod API and drop it from
# pods.conf. Requires RUNPOD_API_KEY.
#   bash pod_finish.sh <tag> <journal stem> '<json {"method": n_finals, ...}>'
#   bash pod_finish.sh <tag> k0_<scale>_<method> '{"k0": 5}'     # LR x decay grid (K0)
#   bash pod_finish.sh <tag> dc '{"XS_D10M_E16_AdamW": 3, ...}'  # data-constrained study: results/dc/<stem>.json, n finals each
set -uo pipefail
tag=$1; stem=$2; exp=$3
line=$(grep "^$tag " pods.conf | tail -1); ip=$(awk '{print $2}' <<<"$line"); port=$(awk '{print $3}' <<<"$line")
[ -n "$ip" ] || { echo "$tag not in pods.conf"; exit 1; }
state=$(ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=ERROR -p "$port" "root@$ip" 'tmux has-session -t run 2>/dev/null && echo RUN || echo END' 2>/dev/null)
[ "$state" = "END" ] || { echo "$tag: still running (or unreachable: '$state')"; exit 2; }
bash pod_status.sh --pull "$tag" 2>&1 | grep "↓"
python3 - "$tag" "$stem" "$exp" <<'PY'
import json, os, re, io, sys, urllib.request
tag, stem, exp = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
conf = io.open('pods.conf', encoding='utf-8').read()
pid = re.search(rf'^# {tag} id=(\S+)', conf, re.M).group(1)
import math
if stem.startswith('k0_'):
    res = f'results/{stem}.json'
    ok = os.path.exists(res) and os.path.exists(f'results/journal/{stem}.{tag}.jsonl')
    if ok:
        fin = json.load(open(res))['finals']
        ok = len(fin) >= exp['k0'] and all(math.isfinite(m['test_ppl']) for m in fin.values())
        print(tag, {'finals': len(fin)}, 'OK' if ok else 'INCOMPLETE')
    else:
        print(tag, 'result or journal missing locally: INCOMPLETE')
elif stem == 'dc':
    ok = True
    for rs, n in exp.items():
        res = f'results/dc/{rs}.json'
        if not os.path.exists(res):
            print(tag, rs, 'missing locally: INCOMPLETE'); ok = False; continue
        d = json.load(open(res)); fin = d['finals']
        good = (len(fin) >= n and all(math.isfinite(m['test_loss']) for m in fin.values())
                and all(math.isfinite(g['val_loss']) for g in d['grid']))
        print(tag, rs, {'grid': len(d['grid']), 'finals': len(fin)}, 'OK' if good else 'INCOMPLETE')
        ok = ok and good
else:
    counts = {}
    for line in open(f'results/journal/{stem}_standardized_results.{tag}.jsonl'):
        try: r = json.loads(line)
        except Exception: continue
        if r.get('kind') == 'eval': counts[r['method']] = counts.get(r['method'], 0) + 1
    ok = all(counts.get(m, 0) >= n for m, n in exp.items()) and os.path.exists(f'new results/incoming/{tag}/{stem}_standardized_results.csv')
    print(tag, {m: counts.get(m, 0) for m in exp}, 'OK' if ok else 'INCOMPLETE')
if not ok: sys.exit(3)
req = urllib.request.Request(f'https://rest.runpod.io/v1/pods/{pid}', method='DELETE', headers={'Authorization': f'Bearer {os.environ["RUNPOD_API_KEY"]}',
                                      'User-Agent': 'regdecay-pod-create/1.0'})
try:
    with urllib.request.urlopen(req, timeout=60) as r: st = r.status
except urllib.error.HTTPError as e: st = e.code
print(f'DELETE {tag} {pid} -> {st}')
if st in (200, 204):
    keep = [l for l in conf.splitlines() if not (l.startswith(f'{tag} ') or l.startswith(f'# {tag} id='))]
    keep.append(f'# DONE + pulled + deleted: {tag}={pid}')
    io.open('pods.conf', 'w', encoding='utf-8').write('\n'.join(keep) + '\n')
PY
