"""disasm-rtx-5090: one harness process under the repaired counter-free observer
(.benchmark-artifacts/gpu/library-images/preparation/library-image-capture-v2: capture.cpp
+ admit.py, unchanged), then the exact launched entries decoded from the saved images.
usage: python3 obs.py <out_dir> <ptx_for_MAPAL_CAPTURE_PTX> <observer.so> -- <argv...>
Writes <out>/stdout, stderr, admission.json (or admission-error.txt), sass/<n>-<symbol>.sass,
launches.tsv (every launch in the selected region: symbol grid block smem join)."""
import json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'observer'))
from admit import admit

out, ptx, so = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
argv = sys.argv[sys.argv.index('--') + 1:]
out.mkdir(parents=True)
env = {**os.environ, 'CUDA_INJECTION64_PATH': so, 'MAPAL_CAPTURE_OUT': str(out / 'capture'),
       'MAPAL_CAPTURE_PTX': ptx, 'DIS_PROFILE': '1', 'NVIDIA_TF32_OVERRIDE': '0'}
r = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=180)
(out / 'stdout').write_text(r.stdout)
(out / 'stderr').write_text(r.stderr + f'\nexit={r.returncode}\n')
nvdisasm = '/usr/local/cuda/bin/nvdisasm'
(out / 'sass').mkdir()
rows = []
try:
    rep = admit(out / 'capture')
    (out / 'admission.json').write_text(json.dumps(rep, indent=1, default=str))
    seen = {}
    for x in rep['selected_launches']:
        img = next(i for i in rep['images'] if i['event'] == x['image_event'])
        key = (x['image_event'], x['symbol'])
        if key not in seen:
            seen[key] = len(seen)
            s = subprocess.run([nvdisasm, '--print-code', '--cuda-function-index', str(x['elf_symbol_index']),
                                str(out / 'capture' / img['path'])], capture_output=True, text=True, timeout=60)
            (out / 'sass' / f"{seen[key]:02d}-{x['symbol'][:120]}.sass").write_text(s.stdout + s.stderr)
        rows.append([x['symbol'], 'x'.join(map(str, x['grid'])), 'x'.join(map(str, x['block'])), str(x['smem']), x['join'],
                     f"{seen[key]:02d}"])
    status = f"admitted launches={len(rep['selected_launches'])} images={len(rep['images'])}"
except Exception as e:  # keep the raw capture; decode the launched symbols from every saved image
    (out / 'admission-error.txt').write_text(f'{type(e).__name__}: {e}\n')
    status = f'NOT ADMITTED: {type(e).__name__}: {e}'
    ev = (out / 'capture' / 'events.tsv').read_text().splitlines() if (out / 'capture' / 'events.tsv').exists() else []
    syms = []
    for l in ev:
        f = l.split('\t')
        if f[0] == 'launch' and f[3] == '0' and f[13] == '1' and f[14] == '0':  # exit site, in the region, ok
            sym = bytes.fromhex(f[15]).decode()
            rows.append([sym, f[10].replace(',', 'x'), f[11].replace(',', 'x'), f[12], 'fallback-cuobjdump', ''])
            if sym not in syms: syms.append(sym)
    for i, sym in enumerate(syms):
        for cub in sorted((out / 'capture').glob('module-*.cubin')):
            s = subprocess.run(['/usr/local/cuda/bin/cuobjdump', '-sass', '-fun', sym, str(cub)], capture_output=True, text=True, timeout=60)
            if 'Function :' in s.stdout:
                (out / 'sass' / f"{i:02d}-{sym[:120]}.sass").write_text(f'# fallback: cuobjdump -sass -fun from {cub.name}\n' + s.stdout)
                break
(out / 'launches.tsv').write_text(''.join('\t'.join(r) + '\n' for r in rows))
print(f'{out.name}: exit={r.returncode} {status}')
