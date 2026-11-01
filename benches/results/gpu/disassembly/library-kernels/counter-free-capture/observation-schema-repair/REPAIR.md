# Callback invocation and ELF compatibility repair

The actual pilot admitted **0/3 executed-image legs**. Its Mapal harness exited0 and passed pre/post numeric gates; the attribution parser rejected before decoding. cuBLAS/CUTLASS selected-call legs were skipped. No SASS or ordinary timing comes from this repair.

`diagnosis.json` binds the immutable raw events/images and original rejection. The original parser has92 duplicate `(domain,correlation,site)` keys across6 reused driver-correlation groups, including repeated same-CBID calls nested under runtime calls. These are distinct callback invocations. Pinned CUDA12.8.90 `cupti_callbacks.h`376–383 documents `correlationData` as shared64-bit state between one invocation's ENTER/EXIT; correlationId separately links activity records. The repair records an observer-owned monotonically assigned token in that storage, then checks its active storage/domain/CBID/correlation at EXIT. Sequential storage reuse is allowed after completion; active reuse, missing storage, zero/corrupt/orphan tokens and unfinished invocations reject. No row is deduplicated.

The token is carried independently on API, launch, loader, direct-GetFunction and stream-metadata records. Runtime parent identity includes its invocation token and original activity correlation. The parser checks exact callback intervals/CBID/correlation/name/context where applicable and requires complete supported launch/loader callback inventories. Activity correlation stays authoritative for kernel/API joins. Repeated or ambiguous activity/dispatch correlations still reject; this repair does not invent an activity invocation ID or infer execution from delivery order.

Static ELF compatibility: actual module-2759.cubin contains defined executable STT_FUNC index19 `$__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath`, offset576/size1856, within `.text._Z8decode_qPfiif`. A helper can share its containing entry's text section. The repair bounds/validates every defined function, requires a unique exact section-owning entry, and inventories contained helper subranges under it. Only exact `.text.<own-symbol>` entry symbols enter the launch resolver. Helper names cannot become launch entries or decoder ordinals. ELF table uniqueness, executable/type/range checks and active-image collision checks remain. All5 real saved images now parse statically; this does not admit execution joins.

Context compatibility: raw callbacks239/240 carry a nonzero pointer with UID0 before authoritative CONTEXT_CREATED241. Pinned callback documentation says context is the current context and may change during runtime initialization; UID is assigned as contexts are created. The repair does **not** treat UID0 as a proven initialized context. It permits only exact driver SetCurrent303 or PrimaryCtxRetain386 provisional records in the prefix before the same pointer's observed CONTEXT_CREATED. A provisional record after creation/destruction, missing creation, positive-UID precreation use, arbitrary UID0 API, module/image use or dispatch before creation still rejects. Every executed context/stream must retain its successful CUPTI getter and activity-ID joins. The exception supplies no module or context identity.

Only production `capture.cpp` and `admit.py` change. `source-diff.patch` compares them with immutable packet-v1. Runner/request/harness/numerical checker/selection/PTX/tool prerequisites/CLI/resource caps remain unchanged. Callback TSV schema changes internally to `library-image-callback-v2`; old traces cannot silently acquire new invocation tokens and are not retrospectively admitted.

A fourth issue was found by the requested full replay: six warmup dispatches have identical numeric driver and runtime-parent correlations, with only matching RUNTIME activity. The old numeric-priority branch incorrectly required DRIVER activity. The repair enumerates the actual enabled domain/CBID candidates, requires a unique callback invocation (exact parent token/containment for runtime), requires matching successful callback/activity status, and admits exactly one domain. Both valid domains, duplicate eligible invocations, wrong parents and contradictory statuses reject.

Local proof:

- `cpu-admission-reviewed/results.json`:53 production-C++ callback/parser cases, actual pinned CUPTI types, stubbed CUPTI functions. Includes repeated/nested same-CBID/correlation, repeated loaders, missing/active-reused/zero tokens, ambiguous launch correlation, helper containment/nonexec/out-of-bounds/selected-helper rejection, drop/graph/CDP/unknown dispatch and lifetime negatives.
- `cpu-binding-reviewed/results.json`:12 adversarial mutations of callback-produced output, including swapping loader tokens between actual same-CBID/correlation invocations, cross-domain token reuse, missing launch metadata and precise provisional-prefix rejection.
- `actual-callback-replay-final/replay-result.json` (initial generic check):3,118 generic API callback rows,1,559 invocations,6 reused-correlation groups replayed through production source. The CPU harness supplies synthetic documented invocation storage. It excludes38 API rows needing launch/loader/GetFunction parameter payloads or profiler region control, covered by the preceding actual-source fixtures. This replay is not admission of v1 raw execution and does not prove the GPU implementation will furnish the expected storage behavior.
- `full-synthetic-replay-reviewed/results.json`:all3,156 API callback rows,1,799 activity records,5 byte-identical images and13 kernel records replayed through the actual repaired callback/activity code. The recorded single API-activity process/thread pair, exactly balanced serialized callback stack and contiguous completion block are explicitly verified. Fresh CPU-owned invocation storage is supplied. Stripping only new invocation fields and schema reproduces every original event field exactly, including IDs/parent correlations/launch resources; the strict parser accepts this **synthetic fixture**. No raw v1 row was changed or retrospectively admitted. Generated trace/cubins remain local and are excluded from the next remote payload.
- `cpu-runner-reviewed/results.json`:10 unchanged request/decoder integration cases rerun with the repaired parser imported.
- `actual-elf-inventory.json` and `diagnosis.json`:static parsing of all5 immutable callback images; source and image SHA256 identities retained.

Compile/reproduce with the package's pinned include trees (or the preparation root's identical header-proof):

```sh
c++ -std=c++17 -O0 -Wall -Wextra -Werror \
 -I ../header-proof/crt/usr/local/cuda-12.8/targets/x86_64-linux/include \
 -I ../header-proof/cudart/usr/local/cuda-12.8/targets/x86_64-linux/include \
 -I ../header-proof/cupti/usr/local/cuda-12.8/targets/x86_64-linux/include \
 test_callback.cpp -o test_callback
python3 -B test_admission.py --out /private/tmp/FRESH-callback-cases
python3 -B test_invocation_binding.py --base /private/tmp/FRESH-callback-cases \
 --out /private/tmp/FRESH-binding-cases
```

Review and fresh root-approved live execution remain required. No host control, driver changes, counter permission changes, rental, upload or retry were performed here. V1 archive/packet and failed trace are untouched.
