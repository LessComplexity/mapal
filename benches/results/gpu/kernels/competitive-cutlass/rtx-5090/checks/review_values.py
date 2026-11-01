from pathlib import Path
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import re
import struct

packet = Path('/private/tmp/mapal-s61-competitive-reference/packet-v2')
root = Path('/private/tmp/mapal-s61-gpu/attempt-09/retrieved/trace/competitive-ordinary')
out = Path('/private/tmp/mapal-competitive-timing-review')
summary = json.loads((root / 'summary.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

@lru_cache(maxsize=None)
def value(bits):
    number = struct.unpack('<f', struct.pack('<I', bits))[0]
    assert math.isfinite(number)
    return Fraction.from_float(number)

@lru_cache(maxsize=None)
def oracle(name):
    path = packet / 'oracles' / name
    document = json.loads(path.read_text())
    assert document['budget_rule'] == 'exact-rational-binary64-ceiling-v1'
    cells = {}
    for cell in document['cells']:
        ordinal = cell['ordinal']
        assert ordinal not in cells
        refs = [int.from_bytes(bytes.fromhex(cell[k]), 'little')
                for k in ('operationwise_bits_le', 'fmaf_bits_le')]
        bound = Fraction.from_float(cell['budget'])
        exact_bound = Fraction(int(cell['budget_exact_numerator']), int(cell['budget_exact_denominator']))
        assert bound >= exact_bound
        assert Fraction.from_float(math.nextafter(cell['budget'], -math.inf)) < exact_bound
        cells[ordinal] = (cell['row'], cell['col'], refs, bound)
    return cells, sha(path)

records, words = [], 0
for record in summary['gates'] + summary['tuning'] + summary['confirmations']:
    args = record['process']['argv']
    filename = Path(args[args.index('--oracle-file') + 1]).with_suffix('.json').name
    cells, oracle_hash = oracle(filename)
    assert oracle_hash == record['oracle_json_sha256']
    path = root / record['path'] / 'stdout'
    assert sha(path) == record['process']['stdout_sha256'] == summary['files'][str(path.relative_to(root))]
    found = {arm: {} for arm in ('mapal', 'cublas', 'cutlass')}
    face_index = 1 if '--contract' in args else 0
    matches = {arm: [0, 0] for arm in found}
    for line in path.read_text().splitlines():
        if not line.startswith('checked_value '):
            continue
        match = re.fullmatch(r'checked_value leg=(mapal|cublas|cutlass) ordinal=(\d+) row=(\d+) col=(\d+) bits=([0-9a-f]{8})', line)
        assert match, line
        arm, ordinal, row, col, bits = match.groups()
        ordinal, row, col, bits = int(ordinal), int(row), int(col), int(bits, 16)
        assert ordinal not in found[arm]
        erow, ecol, refs, bound = cells[ordinal]
        assert (row, col) == (erow, ecol)
        actual = value(bits)
        for index, ref in enumerate(refs):
            assert abs(actual - value(ref)) <= bound, (record['path'], arm, ordinal)
            matches[arm][index] += bits == ref
        if arm == 'mapal':
            assert bits == refs[face_index], (record['path'], ordinal, 'ordered bits')
        found[arm][ordinal] = bits
        words += 1
    for arm in found:
        assert set(found[arm]) == set(cells)
    records.append(dict(path=record['path'], cells_per_arm=len(cells),
                        matches_operationwise_fmaf=matches, stdout_sha256=sha(path)))

assert len(records) == 68 and words == 254976
result = dict(pass_=True, processes=len(records), checked_value_words=words,
              implementation='Independent Fraction/IEEE-word reader; imports no harness checker code',
              source_scope='Pre-timing exported words against frozen dual scalar references and independently audited outward rational bounds',
              oracle_files=oracle.cache_info().currsize, records=records)
(out / 'numerical-result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'records'}))
