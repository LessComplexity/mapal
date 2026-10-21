# Report presentation QA

2026-10-21. Author: `/root/audit_s54_worktrees`. Read-only review of the three presented Markdown reports against already independently checked CSVs, their check receipts, and preserved detailed reports. No exporter/validator rerun, source audit, compilation, benchmark/GPU action or source/HEAD change. Only this note was written.

## Findings

**No outstanding numerical presentation mismatch or scope blocker in the final hashes below.** One minor rounding mismatch was reported and corrected by root: `s54-gpu-fusion.md:156` originally printed the RTX4090 primary normalized-null minimum as0.98073. The checked Exact1024 CSV value is0.9807248299710614, so direct five-decimal rounding is0.98072. The final Markdown now uses0.98072; the interpretation is unchanged. No other numerical edit was needed.

## Checked scope

| Presentation | Cross-check and outcome |
|---|---|
| s54 integrated kernel tables |90 rows: four primary hosts ×18 rows plus independent4090 ×18. All450 displayed timing/ratio cells match the relevant shipped CSV rows at the displayed precision. Exact maps to `_exact`, FMA to Contract `_c`, TF32 to requested `_tf`. Ratios are library/Mapal; values above1 favor Mapal. |
| s54 composed-program comparison |20 Contract4096 rows match program medians, selected library medians and leg names. The text correctly labels ordinary Mapal iteration versus library device events and supplies no end-to-end speed ratio between those different clocks. |
| s54 decode |Four shipped/control wall rows, rounded ratios and allocation pairs match the program CSVs. The 578/627/564/450ms values are ordinary process-wall medians. The control disables base residency together with retain-delta; prose explicitly refuses an isolated-retention attribution. The stated trace byte/allocation counts agree with those rows. |
| s54 selection/profiling and controls |The four affected Contract4096 families' ON/OFF medians match both accepted selection hosts. The MLP control and allocation/API chronology agree with the preserved placement triage. Eight actual timed allocations are correctly distinguished from ten static allocation call sites;14 is process-wide. Zero observed cross-stream tile overlap is limited to the two profiled MLP2 ON runs, with no SM-utilization claim. API elapsed time is not added to kernels or called entirely removable allocator cost. |
| s54 confirmation and instability |Independent4090 has its own machine/host, power setting and table. The favorable FMA2048/4096 medians repeat; the8192 CUTLASS win does not. Broad ranges and the1.667% maximum normalized-null deviation are retained. The unstable5090 Exact8192 normalized null0.70363 is explicitly excluded from small-optimization support. Other null bounds match the checked CSVs after the minor correction above. |
| Large/rectangular timing and route tables |All48 displayed base rows match cases.csv, including actual routes.47 rows have matching numeric medians; M1/N2048/K7168 TF32 correctly shows gate failure and no timing. The four diagnostic rows,16K repeat medians and5.25GiB/3GiB requested-storage labels match preserved data. These are kernel CUDA-event times, not process times or measured peak VRAM. |
| Primitive inventory and failures |The96 emission cells comprise32 ordered separate mul/add,32 requested Contract FMA,25 TF32 MMA and7 requested-TF32 FMA fallbacks. The fallback set and all M16 Exact/Contract scalar geometries match summary.csv. cases.csv and primitive-verification.json agree on98/100 passing invocations and two failed base/null invocations; integrity_pass=true is kept separate from overall pass=false. |
| PTX/SASS summary |All four displayed timing rows match the30-row comparison CSV, including supplement rows on another5090 host. The M16 dense6.07× time ratio is Mapal/cuBLAS, while the surrounding win ratios use library/Mapal and are labeled accordingly. The4090 primary/confirmation distinction and8192 non-repeat are correct. Instruction/resource claims remain scoped to the preserved detailed report; offline Mapal assembly is not promoted to the driver's actual JIT image. |

The four primary independent-check receipts sum to30,621 numeric fields and15,660 raw kernel samples (42+42+42+48 kernel rows, three legs,30 samples), matching the presentation. The4090 confirmation receipt reports7,494 fields,42 kernel rows and620 loss rows, matching its section. These receipt checks are not another complete exporter audit. The shown kernel tables are selected shipped summaries; full arms, distributions, nulls and losses remain linked.

## Numerical and evidence limits retained correctly

* Extracted-kernel gates sample256 outputs. Exact/Contract use the ordered bitwise reference; requested TF32 gates Mapal against four times cuBLAS's clamped diagnostic. The diagnostic denominator is max(abs(f64 reference),0.001), not an unconditional relative-error or percentage measure. There is no fixed absolute application-accuracy ceiling and no independent cuBLAS/CUTLASS accuracy gate in that instrument. The reports distinguish separate composed/decode library gates.
* The passing M16 expert TF32 diagnostics3.302/3.302/3.248 are disclosed. The failed M1 requested-TF32 fallback retains its diagnostic, exit and absent timing. Byte-identical renamed fallback PTX with a passing Contract gate does not itself establish a wrong-value compiler defect; the Boolean stderr sentinel is not treated as a cell count.
* Library configurations are pinned tested choices, not an exhaustive tuning search. Separate precision contracts, different hosts and differing time domains are not silently combined into speedup claims. Medians/ranges are not confidence intervals.
* No executed Mapal JIT SASS, cuBLAS SASS or useful hardware counters is claimed by the PTX audit. Static conversion, register, local-byte and pipeline facts are not assigned a causal share of time. Candidate experiments remain candidates; rounding changes remain semantic decisions.
* At the reviewed snapshot only four final primaries plus the independent4090 confirmation are accepted. H100 primary and independent5090 confirmation remain pending/active. Neither is admitted by this QA. New composed GPU/PyTorch values/timings remain pending, and recurrent checkpoint failure is disclosed; emitted placement and scalar/native certification do not replace those missing gates.
* The later added bounded replay paragraph was reviewed as prose. It correctly labels its source-level exit-payload explanation as a candidate, leaves exact failed-object identity unproved and selects no fix. Host call-effects paragraphs also keep existing vectorization separate from unproved joint scheduling/performance gains.

## Reviewed versions

Numerical tables did not change during review. Root added a bounded replay-candidate paragraph to the large report and corrected the one null rounding digit in s54. The final hashes below bind the completed QA; later additions require their own review and are not implicitly covered.

| Markdown | First-read SHA256 | Final reviewed SHA256 |
|---|---|---|
| `docs/performance/s54-gpu-fusion.md` | `a5f5d12e1edbda8be43ab69c957a2f42563b7c3a25a1609bcf808c32cafdd3f8` | `4e4c90a22463adc1eba3b4fb651dbe6690e9abf49f682d511d918c65ee5883ec` |
| `docs/performance/s57-large-composed-evidence.md` | `e44f73b9893ebe58b3dbdaf2dfabc3b89cff2bbf868699a5664608a5caf5d383` | `911ee8446542cc928b57403839f5228e3f665bf2277ae0a6b09bdb6528e2ef49` |
| `docs/performance/s57-ptx-sass-gap.md` | `867b8410f0891a4ea91db6a6062bc0738a2c446596d260ec502989f322a89068` | `867b8410f0891a4ea91db6a6062bc0738a2c446596d260ec502989f322a89068` |

Checked presentation-input file digests (no regeneration):

```json
{
  "benches/results-s54-gpu/s57-integrated-rtx-3090-primary-retry2-kernels.csv": "0808c9c2a58b9ba8bb4cb4523b24de8bb283fb33452aeaa20a1c5e3ac2a93e1a",
  "benches/results-s54-gpu/s57-integrated-rtx-3090-primary-retry2-programs.csv": "490d6e4dd7b9d15e5c4b013adfa9ea63b464a2b93f407625201dcc913e638f8e",
  "benches/results-s54-gpu/s57-integrated-rtx-3090-primary-retry2-nulls.csv": "ca81574c71fb6bdce48459c08ae34800388a559c7ee778ce0c800d85856f0d42",
  "benches/results-s54-gpu/s57-integrated-rtx-3090-primary-retry2-independent-csv-check.json": "2de3f0597df9d4de7864e8332cbd39d09b37b2bc0f7d825f65a1d9b663eeaa30",
  "benches/results-s54-gpu/s57-integrated-a100-primary-kernels.csv": "5c75cada709ab221d166b87917412f8071d710cd701d5c9e2f63b9b9fd94ec25",
  "benches/results-s54-gpu/s57-integrated-a100-primary-programs.csv": "4a6118399ce7edee2de600bc2bef80d45572ba7f6cc6a743b0145105baf1082f",
  "benches/results-s54-gpu/s57-integrated-a100-primary-nulls.csv": "e84738fc1ed50b6ed7b79256f2976256915b729de9e0bf95448d720053446ea0",
  "benches/results-s54-gpu/s57-integrated-a100-primary-independent-csv-check.json": "c8b5746e717e9d0d6cb9649d6ada99a00cb8d154b6c7a09926128e4c7f35d718",
  "benches/results-s54-gpu/s57-integrated-rtx-4090-primary-kernels.csv": "6d2b1cb5491d52a8b6ef50420962199034c7e853bcddb5ee1a8c8a1f5210238c",
  "benches/results-s54-gpu/s57-integrated-rtx-4090-primary-programs.csv": "3e3bb914eb3fabf95599d252b052316120f79d8528d0d11af490ffe4b609b8fd",
  "benches/results-s54-gpu/s57-integrated-rtx-4090-primary-nulls.csv": "8661cc6c37d692633c80bdb0293e4781ae6ac1ddb57a35998a334c017cfd6ca1",
  "benches/results-s54-gpu/s57-integrated-rtx-4090-primary-independent-csv-check.json": "cf27aff5ea02de023d240c383d3b208f1abc7d6911cbc7139ecbc48e7312fad8",
  "benches/results-s54-gpu/s57-integrated-rtx-5090-primary-retry4-kernels.csv": "aad2f6d369b11e3420ac31da14d586b51b0e42a055e9135a8df32987ba0b5cce",
  "benches/results-s54-gpu/s57-integrated-rtx-5090-primary-retry4-programs.csv": "dab37498b8ce41b2a01a7313bff0aa9ad962530b4f875e2bdf9c56cdf93bbe6f",
  "benches/results-s54-gpu/s57-integrated-rtx-5090-primary-retry4-nulls.csv": "b0e071544d8fabb6d513e592eb94bce22b2e1f5e5283a639b8f56c4ac9619453",
  "benches/results-s54-gpu/s57-integrated-rtx-5090-primary-retry4-independent-csv-check.json": "00d6e4656847f4fda015a2e972a53c9a1421cc667cc6c7bd1854e1a78eb84bb9",
  "benches/results-s54-gpu/s57-integrated-rtx-4090-confirm-retry3-kernels.csv": "16133292dc6de20610bec581f7255cd17d28c19dc17b0d5f3478dc97a443f488",
  "benches/results-s54-gpu/s57-integrated-rtx-4090-confirm-retry3-programs.csv": "86a16f9c9546eb179936edfd7bca250e459010fe5f2f4281619d4c0ac875ae7f",
  "benches/results-s54-gpu/s57-integrated-rtx-4090-confirm-retry3-nulls.csv": "bca87fa79d48b7edcf582d3735f902cacc221f4e3440dfb15324964050deff86",
  "benches/results-s54-gpu/s57-integrated-rtx-4090-confirm-retry3-independent-csv-check.json": "e3bbd6505461720321522f00cde2f4402a338291f261ceab177b1abddd299678",
  "benches/results-s54-gpu/s56-selection-rtx-5090-streams.csv": "9adcbe111162d4b03b198196931c98286142a25c349fa6811c5b5e981706e08c",
  "benches/results-s54-gpu/s57-large-rtx-5090/cases.csv": "3e6d2f4a5a35f47fadacc0af734cf57dea1f24b6fb86263178d54c733b1e8a56",
  "benches/results-s54-gpu/s57-large-rtx-5090/primitive-verification.json": "43bb3366f2203247721a934ee1d9053dfa62c0139211a1307a8d8c87856dd17d",
  "benches/results-s54-gpu/s57-large-composed-preparation/primitive/summary.csv": "8403cba0e957f5a5c16b03d95ef73721f221413577ea163ddfcf3673698d01a7",
  "benches/results-s54-gpu/s57-ptx-sass-gap/comparison.csv": "f6948f22f511f6bbd50eccc68e08fc59d8864b1465bd029d6dcec27c97d859c7",
  "benches/results-s54-gpu/s57-ptx-sass-gap/REPORT.md": "7dc41e0cb12d2dd5cb9979c8f2fed3ec83ccc711a449572fbbb285d5b874f846"
}
```
