import json,sys
d=json.load(open(sys.argv[1]))
print({k:d[k] for k in ("clock_sm_mhz_median_under_load","clock_mem_mhz_median_under_load","peak_fp32_tflops_at_clock","peak_dram_gbs")})
for c,v in d["cells"].items():
    print("==",c,"null",v["null_variation_per_pass"])
    for n,r in v["rows"].items():
        print(f"  {n:22s} {r['pass_medians_ms']} all={r['median_ms_all']:.4f} spread={r['pass_spread']} GB/s={r['gbs']} %bw={r['pct_dram_peak']} TF={r['tflops']} %fp={r['pct_fp32_peak']} AI={r['ai_flop_per_byte']} tbw={r['t_dram_floor_ms']} tfp={r['t_fp32_floor_ms']}")
