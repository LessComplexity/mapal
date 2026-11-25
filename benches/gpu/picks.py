#!/usr/bin/env python3
"""The site picks an arm launched, and the arm a rung A/B compares an OFF arm with at the same picks.

box_run.sh's traced repetitions print each site's decision (`probe ... sites=[<key>:<pick>:<src> ...]`, one entry a
site in launch order). A tuned box runs `on` from its wisdom (the tune pass trials the ON text only); an
`off-<rung>` arm is another text, whose site keys the wisdom does not hold, so it launches the rules' candidate 0.
Its time against `on` is then two changes — the rung and the pick — not the rung's effect (gapfix-rows-operands:
the RTX 3090's shfl-tree row, `on` candidate 3 from the wisdom against the OFF arm's candidate 0: 0.743; at
candidate 0 on both arms 0.969, inside the session's null). A rung's ratio is read against the ON text's arm that
launched the same picks: `on` itself, else `rules` (candidate 0 forced), else `pick<i>` (candidate i at every site);
none of them: the pair is pick-confounded and judges nothing.
"""
import re

SITES = re.compile(r' sites=\[([^\]]*)\]')


def launched(line):
    """The pick indexes a traced probe line's sites launched, in launch order; None when the line carries none."""
    m = SITES.search(line)
    if not m:
        return None
    return tuple(int(e.split(':')[1]) for e in m[1].split() if e.count(':') == 2)


def same(a, b):
    """Two arms launched the same picks: equal sequences, or one candidate at every site of each (a forced pick
    prints one entry, `0000000000000000:0:forced`)."""
    return a == b or (bool(a) and bool(b) and len(set(a)) == 1 and set(a) == set(b))


def partner(picks, arm, base='on'):
    """The ON-text arm to compare `arm` with: `base` when the two launched the same picks (or either was not
    traced: nothing to check), else `rules`, else `pick<i>` with the same picks; None when no timed arm matches.
    picks: {arm: launched tuple or None} of one (program, face) in one session."""
    a, b = picks.get(arm), picks.get(base)
    if a is None or b is None or same(a, b):
        return base
    for alt in ['rules'] + sorted(k for k in picks if re.fullmatch(r'pick\d+', k)):
        if picks.get(alt) is not None and same(picks[alt], a):
            return alt
    return None
