#!/usr/bin/env python3
"""RunPod pods over HTTPS only (GraphQL API + the pod's Jupyter proxy), for networks that block ssh.

    RUNPOD_API_KEY=... python pod_https.py create <tag>                  # pod + Jupyter ready
    RUNPOD_API_KEY=... python pod_https.py launch <tag> <launch script> <args...>
    RUNPOD_API_KEY=... python pod_https.py sh <tag> '<shell command>'    # run a command, print its output
    RUNPOD_API_KEY=... python pod_https.py pull <tag> <glob tag>         # results + log -> pull_<tag>/
    RUNPOD_API_KEY=... python pod_https.py status|stop|start|delete <tag>

State (pod id, Jupyter token) lives in pods_https.json (git-ignored, mode 600). `launch` uploads a
tarball of the committed tree (code, tests, the journals the launch script replays) and the API key,
builds the pinned corpus on the pod from HuggingFace (dcstudy.prepare_data; pod_launch_dc.sh then
checks its SHA-256 against dcstudy/data_meta.json and refuses to run on a mismatch), and starts the
launch script in tmux. The launch script stops the pod itself when it ends.
"""
import base64
import io
import json
import os
import ssl
import subprocess
import sys
import tarfile
import time
import secrets
import urllib.error
import urllib.request

GQL = 'https://api.runpod.io/graphql'
IMAGE = 'runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04'
GPUS = ['NVIDIA GeForce RTX 3090', 'NVIDIA RTX A4500', 'NVIDIA RTX A5000', 'NVIDIA GeForce RTX 4090',
        'NVIDIA RTX A6000', 'NVIDIA GeForce RTX 3090 Ti', 'NVIDIA L4', 'NVIDIA A40', 'NVIDIA L40S', 'NVIDIA L40', 'NVIDIA RTX 6000 Ada Generation',
        'NVIDIA RTX 4000 Ada Generation', 'NVIDIA RTX 5000 Ada Generation', 'NVIDIA A100 80GB PCIe',
        'NVIDIA A100-SXM4-80GB', 'NVIDIA H100 PCIe', 'NVIDIA H100 80GB HBM3']
CLOUDS = os.environ.get('CLOUD', 'COMMUNITY,SECURE').split(',')
STATE = 'pods_https.json'
REVISION = '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
UA = {'User-Agent': 'regdecay-pod-https/1.0'}


def key():
    k = os.environ.get('RUNPOD_API_KEY')
    if not k:
        sys.exit('set RUNPOD_API_KEY')
    return k


def gql(query, variables=None):
    body = json.dumps({'query': query, 'variables': variables or {}}).encode()
    req = urllib.request.Request(GQL, data=body, headers={'Content-Type': 'application/json',
                                                          'Authorization': f'Bearer {key()}', **UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    if out.get('errors'):
        raise RuntimeError(json.dumps(out['errors'])[:400])
    return out['data']


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save_state(st):
    with open(STATE, 'w') as fh:
        json.dump(st, fh, indent=1)
    os.chmod(STATE, 0o600)


def pod(tag):
    st = load_state()
    if tag not in st:
        sys.exit(f'no pod {tag} in {STATE}')
    return st[tag]


# ----------------------------------------------------------------------------- Jupyter
def jurl(p, path):
    return f"https://{p['id']}-8888.proxy.runpod.net{path}"


def jreq(p, method, path, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(jurl(p, path), data=data, method=method,
                                 headers={'Authorization': f"token {p['token']}",
                                          'Content-Type': 'application/json', **UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def jupyter_ready(p):
    try:
        jreq(p, 'GET', '/api/status', timeout=20)
        return True
    except Exception:
        return False


def upload(p, data, name):
    """Upload bytes to <jupyter root>/<name> in 8 MB chunks (contents API)."""
    step = 8 << 20
    n = max(1, (len(data) + step - 1) // step)
    for i in range(n):
        part = data[i * step:(i + 1) * step]
        chunk = -1 if i == n - 1 else i + 1
        body = {'type': 'file', 'format': 'base64', 'name': name, 'path': name,
                'content': base64.b64encode(part).decode()}
        if n > 1:
            body['chunk'] = chunk
        jreq(p, 'PUT', f'/api/contents/{name}', body, timeout=600)


def download(p, name):
    out = jreq(p, 'GET', f'/api/contents/{name}?content=1&format=base64&type=file', timeout=600)
    return base64.b64decode(out['content'])


def sh(p, cmd, wait=20, quiet=False):
    """Run a shell command in a Jupyter terminal and return what it printed within `wait` seconds."""
    import websocket
    term = jreq(p, 'POST', '/api/terminals')['name']
    ws = websocket.create_connection(
        f"wss://{p['id']}-8888.proxy.runpod.net/terminals/websocket/{term}",
        header=[f"Authorization: token {p['token']}"], sslopt={'cert_reqs': ssl.CERT_REQUIRED}, timeout=wait)
    mark = f'__END_{secrets.token_hex(4)}__'
    ws.send(json.dumps(['stdin', f"{cmd}; echo {mark}\r"]))
    buf, t0 = '', time.time()
    try:
        while time.time() - t0 < wait:
            msg = json.loads(ws.recv())
            if msg[0] == 'stdout':
                buf += msg[1]
                if buf.count(mark) >= 2:          # the echoed command line, then the real marker
                    break
    except Exception:
        pass
    ws.close()
    try:
        jreq(p, 'DELETE', f'/api/terminals/{term}')
    except Exception:
        pass
    text = buf.split(mark)[-2] if buf.count(mark) >= 2 else buf
    if not quiet:
        print(text.strip())
    return text


# ----------------------------------------------------------------------------- commands
def create(tag):
    st = load_state()
    if tag in st:
        print(f'[CREATE] {tag} exists: {st[tag]["id"]}')
        return st[tag]
    token = secrets.token_urlsafe(24)
    q = """mutation($input: PodFindAndDeployOnDemandInput) {
      podFindAndDeployOnDemand(input: $input) { id costPerHr machine { gpuDisplayName } } }"""
    for cloud, gpu in [(c, g) for c in CLOUDS for g in GPUS]:
        inp = {'cloudType': cloud, 'gpuCount': 1, 'volumeInGb': 30, 'containerDiskInGb': 40,
               'minVcpuCount': 4, 'minMemoryInGb': 16, 'gpuTypeId': gpu, 'name': f'regdecay-{tag}',
               'imageName': IMAGE, 'ports': '8888/http,22/tcp', 'volumeMountPath': '/workspace',
               'startJupyter': True, 'startSsh': True,
               # no public IP or CUDA filter: they emptied the offer on 2026-11-17; pod_provision
               # refuses a host without CUDA, and the watcher deletes such a pod
               'env': [{'key': 'JUPYTER_PASSWORD', 'value': token}]}
        try:
            d = gql(q, {'input': inp})['podFindAndDeployOnDemand']
        except RuntimeError as e:
            print(f'[CREATE] {cloud} {gpu}: {str(e)[:80]}')
            continue
        if d and d.get('id'):
            st[tag] = {'id': d['id'], 'token': token, 'gpu': gpu, 'cloud': cloud, 'costPerHr': d.get('costPerHr'),
                       'created': time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}
            save_state(st)
            print(f"[CREATE] {tag} -> {d['id']} {cloud} {gpu} {d.get('costPerHr')} $/h")
            break
    else:
        sys.exit('[CREATE] no GPU available')
    p = st[tag]
    for i in range(60):
        if jupyter_ready(p):
            print(f'[CREATE] Jupyter ready after {i * 15} s')
            return p
        time.sleep(15)
    sys.exit(f'[CREATE] Jupyter not ready on {p["id"]} after 15 min: delete it')


def repo_tarball(extra_globs):
    files = subprocess.run(['git', 'ls-files', 'dcstudy', 'analysis', 'tests', 'dc_grid.py', 'experiment_utils.py',
                            'pod_common.sh', 'pod_launch_dc.sh', 'pod_launch_P3.sh', 'pytest.ini',
                            'requirements.txt'], capture_output=True, text=True, check=True).stdout.split()
    import glob
    for g in extra_globs:
        files += glob.glob(g)
    if subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True).stdout.strip():
        sys.exit('[LAUNCH] uncommitted changes: commit first')
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tf:
        for f in sorted(set(files)):
            # read through git for tracked files: OneDrive may hold a dataless placeholder
            blob = subprocess.run(['git', 'show', f'HEAD:{f}'], capture_output=True).stdout
            info = tarfile.TarInfo(f'Regularization_Decay/{f}')
            info.size = len(blob)
            info.mode = 0o755 if f.endswith('.sh') else 0o644
            tf.addfile(info, io.BytesIO(blob))
    return buf.getvalue(), len(files)


def launch(tag, script, args):
    p = pod(tag)
    journals = ['results/journal/dc_XS_D159.384M_E1_AdamW*.jsonl']
    tgz, n = repo_tarball(journals)
    print(f'[LAUNCH] tarball {len(tgz) / 1e6:.2f} MB, {n} files')
    upload(p, tgz, 'regdecay_repo.tgz')
    upload(p, open(os.path.expanduser('~/.runpod_key'), 'rb').read().strip(), 'regdecay_key')
    q = ' '.join(args)
    cmd = (
        "J=$PWD; R=$J/regdecay_repo.tgz; ls -la $R; "
        "install -m 600 $J/regdecay_key /root/.runpod_key && rm -f $J/regdecay_key; "
        f"echo {p['id']} > /root/.runpod_pod_id; "
        "cd /workspace && rm -rf Regularization_Decay && tar xzf $R && rm -f $R && cd Regularization_Decay && mkdir -p logs; "
        "(command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null)); "
        f"tmux new-session -d -s run \"bash -c '"
        f"pip install -q \\\"transformers==4.46.3\\\" datasets scipy pytest 2>&1 | tail -2; "
        f"[ -f /workspace/data/dc/meta.json ] || python -m dcstudy.prepare_data --out /workspace/data/dc "
        f"--train-tokens 160000000 --revision {REVISION}; "
        f"bash {script} {tag} {q}' 2>&1 | tee -a logs/pod_{tag}_full.log\"; "
        "sleep 2; tmux ls; ls /root/.runpod_key /root/.runpod_pod_id")
    sh(p, cmd, wait=120)


def pull(tag, gtag):
    p = pod(tag)
    name = f'regdecay_pull_{tag}.tgz'
    sh(p, f"J=$PWD; cd /workspace/Regularization_Decay && "
          f"tar czf $J/{name} results/dc/*_{gtag}.json results/journal/*_{gtag}.jsonl logs/pod_{tag}_full.log && ls -la $J/{name}", wait=60)
    data = download(p, name)
    sh(p, f"rm -f $PWD/{name}", wait=15, quiet=True)
    out = f'pull_{tag}'
    os.makedirs(out, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        tf.extractall(out)
    print(f'[PULL] {len(data) / 1e6:.2f} MB -> {out}/')


def status(tag):
    p = pod(tag)
    d = gql('query($id: String!) { pod(input: {podId: $id}) { id desiredStatus lastStatusChange '
            'runtime { uptimeInSeconds gpus { gpuUtilPercent } } } }', {'id': p['id']})['pod']
    print(json.dumps(d))
    return d


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, tag = sys.argv[1], sys.argv[2]
    if cmd == 'create':
        create(tag)
    elif cmd == 'launch':
        launch(tag, sys.argv[3], sys.argv[4:])
    elif cmd == 'sh':
        sh(pod(tag), sys.argv[3], wait=int(os.environ.get('WAIT', '30')))
    elif cmd == 'pull':
        pull(tag, sys.argv[3])
    elif cmd == 'status':
        status(tag)
    elif cmd in ('stop', 'start', 'delete'):
        pid = pod(tag)['id']
        m = {'stop': 'podStop(input: {podId: $id}) { id desiredStatus }',
             'start': 'podResume(input: {podId: $id, gpuCount: 1}) { id desiredStatus }',
             'delete': 'podTerminate(input: {podId: $id})'}[cmd]
        print(json.dumps(gql(f'mutation($id: String!) {{ {m} }}', {'id': pid})))
        if cmd == 'delete':
            st = load_state(); st[tag]['deleted'] = time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())
            st[f"{tag}#{pid}"] = st.pop(tag); save_state(st)
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
