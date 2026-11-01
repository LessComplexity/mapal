# RTX 5090 profiler permission denial

The separate Nsight Compute 2025.1.1 diagnostic returned `ERR_NVGPUCTRPERM` on device 0 and exit 1. It created no `.ncu-rep`; no SASS or dynamic-counter result is admitted. Subsequent library legs and the dynamic phase were correctly skipped. The independently admitted ordinary timing remains valid.

[Independent analysis](analysis.json) verifies the selected binary, request and all 15 diagnostic output seals. [Original request](request.json), [raw diagnostic archive](raw/diagnostic.tar.gz), and [recovery hash](recovery.json) preserve the failure. The profiler error split one pre-gate value line; that gate is rejected as malformed rather than reconstructed. All three post gates pass their 256 sampled values, but cannot override the missing report and nonzero profiler exit.

The host permission layer was not inventoried, so its exact configuration is unknown. No host controls were changed. Callback image capture is the next separately reviewed diagnostic for actual library SASS; it will not establish dynamic counter behavior.
