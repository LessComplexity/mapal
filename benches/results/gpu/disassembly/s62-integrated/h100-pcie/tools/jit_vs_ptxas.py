"""Mapal kernels: opcode sequence of the driver-JIT image (observer) vs offline ptxas 12.8.93 -arch=sm_90a."""
import glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from sass_stats import functions, split
static, ls = sys.argv[1], json.load(open(sys.argv[2]))
local = {}  # name -> every offline variant (an MLP site keeps its name across base/integrated builds)
for f in glob.glob(os.path.join(static, '*.sass')):
    for n, body in functions(open(f, errors='replace').read()).items():
        if n.startswith('mapal_'):
            local.setdefault(n, []).append([split(i)[1] for _, i in body])
same = diff = 0
for cap in ls:
    for k in cap['kernels']:
        if not k['name'].startswith('mapal_tile') or not k.get('image'):
            continue
        txt = open(os.path.join(cap['capture'], k['image'] + '.nvdisasm'), errors='replace').read()
        jit = [split(i)[1] for _, i in functions(txt).get(k['name'], [])]
        ok = jit in local.get(k['name'], [])
        same += ok; diff += not ok
        if not ok:
            print('DIFF', k['name'], len(jit))
print(f'mapal kernels: {same} identical opcode sequences, {diff} differ')
