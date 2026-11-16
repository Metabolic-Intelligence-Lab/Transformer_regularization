#!/usr/bin/env python3
"""Stop, start or show a pod recorded in pods.conf, through the RunPod REST API.
A STOPPED pod keeps its /workspace volume (the corpus and the journals) and bills only
storage, so it is the way to pause between protocol steps without paying idle GPU time.

    RUNPOD_API_KEY=... python pod_power.py status P1_pilot
    RUNPOD_API_KEY=... python pod_power.py stop   P1_pilot
    RUNPOD_API_KEY=... python pod_power.py start  P1_pilot     # then update pods.conf ip/port from the output
"""
import json
import os
import re
import sys
import urllib.error
import urllib.request

API = 'https://rest.runpod.io/v1'


def pod_id(tag, conf='pods.conf'):
    m = re.search(rf'^# {re.escape(tag)} id=(\S+)', open(conf, encoding='utf-8').read(), re.M)
    if not m:
        sys.exit(f'{tag}: no "# {tag} id=..." line in {conf}')
    return m.group(1)


def req(method, path, key, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(API + path, data=data, method=method,
                               headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json',
                                        'User-Agent': 'regdecay-pod-create/1.0'})
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {'raw': raw[:300].decode('utf-8', 'replace')}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in ('status', 'stop', 'start'):
        sys.exit(__doc__)
    action, tag = argv
    key = os.environ.get('RUNPOD_API_KEY') or sys.exit('set RUNPOD_API_KEY')
    pid = pod_id(tag)
    if action == 'status':
        st, p = req('GET', f'/pods/{pid}', key)
    else:
        st, p = req('POST', f'/pods/{pid}/{action}', key)
    print(f'{action} {tag} ({pid}) -> HTTP {st}')
    if isinstance(p, dict):
        keep = {k: p.get(k) for k in ('desiredStatus', 'publicIp', 'portMappings', 'costPerHr', 'lastStatusChange') if k in p}
        print(json.dumps(keep or p, indent=1)[:800])
    return st


if __name__ == '__main__':
    sys.exit(0 if main() in (200, 201, 204) else 1)
