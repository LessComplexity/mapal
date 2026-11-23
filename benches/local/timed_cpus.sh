# Sourced by the timed local scripts (model_scale_run.sh, model_scale_once.sh, comp_run.sh): the CPUs a timed run is
# pinned to ($CPUS, a taskset list such as 0-7 or 8-15,40-47) — how busy they are, and what they can run at.
#
# Why the set is part of a result: the 9970X's cores do not reach the same clock. ACPI CPPC's `highest_perf` (the
# firmware's per-core ceiling, /sys/devices/system/cpu/cpuN/acpi_cppc/highest_perf) is 211-241 on CCD 0 (cores 0-7),
# 166 on CCD 1 (8-15), 171-206 on CCD 2 and 166 on CCD 3. The same binaries pinned to 0-7 against 8-15 (integration2
# ABBA, 8 reps): lsm_1_131072 0.81x, pj_m1_n4096_k4096 0.89x, ew_4096 0.94x, mlp_512 / indep_512 0.95x, attn_1024_64
# 1.00x — the host-bound share of a 15-90 us program. So two runs compare only on sets with the same ceilings;
# conditions.txt records them (the 3a8e654f and sweep-merge runs timed on 0-7, integration 1's on 8-15).

# The CPU numbers of a taskset list, one a line.
cpus_of() {
    local IFS=, r
    for r in $1; do
        if [[ $r == *-* ]]; then seq "${r%-*}" "${r#*-}"; else echo "$r"; fi
    done
}

# % busy of $CPUS over 2 s (/proc/stat).
busy() {
    local set a b
    set=$(cpus_of "$CPUS" | sed 's/^/cpu/' | tr '\n' ' ')
    a=$(awk -v s="$set" 'BEGIN { n = split(s, x, " "); for (i = 1; i <= n; i++) w[x[i]] = 1 }
        ($1 in w) { t += $2+$3+$4+$5+$6+$7+$8+$9; i += $5+$6 } END { print t, i }' /proc/stat)
    sleep 2
    b=$(awk -v s="$set" 'BEGIN { n = split(s, x, " "); for (i = 1; i <= n; i++) w[x[i]] = 1 }
        ($1 in w) { t += $2+$3+$4+$5+$6+$7+$8+$9; i += $5+$6 } END { print t, i }' /proc/stat)
    awk -v a="$a" -v b="$b" 'BEGIN { split(a, x, " "); split(b, y, " "); dt = y[1] - x[1]; di = y[2] - x[2];
        print (dt > 0) ? int(100 * (dt - di) / dt) : 100 }'
}

# min-max of the CPPC per-core ceiling over $CPUS ("na" where the firmware does not expose it).
cppc() {
    local c v lo="" hi=""
    for c in $(cpus_of "$CPUS"); do
        v=$(cat "/sys/devices/system/cpu/cpu$c/acpi_cppc/highest_perf" 2>/dev/null) || continue
        [ -z "$lo" ] || [ "$v" -lt "$lo" ] && lo=$v
        [ -z "$hi" ] || [ "$v" -gt "$hi" ] && hi=$v
    done
    [ -n "$lo" ] && echo "$lo-$hi" || echo na
}
