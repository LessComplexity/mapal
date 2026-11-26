# Roadmap

Where Mapal goes next. Back to the [README](../README.md); what is built today: [STATUS](STATUS.md).

0. **Now: the machine model** ([ADR-0050](decisions/ADR-0050-the-machine-model-placement-by-one-cost-model.md)):
   placement by one cost model over the dataflow's and the hardware's facts, matrix products first, then attention,
   cells and folds, and the whole graph's co-schedule. **Format generics**
   ([ADR-0051](decisions/ADR-0051-format-generics-to-and-widen.md)): `x -> to<bf16>`, `x -> widen<f32>`, and
   functions generic over their number formats.
1. **An AMD backend.** The same graph and facts on a second GPU vendor: its compute units, wavefront width, LDS
   and matrix instructions are one more translation into the machine model.
2. **A PyTorch importer**, to bring existing models into Mapal.
3. **Then the ecosystem around the language:** development → deployment → verification → synthesis, where the same
   search also chooses an ASIC's own parameters (array sizes, memories) within a budget.

Done so far, and how each step turned out, including what was tried and did not work:
[`docs/sessions/`](sessions/).
