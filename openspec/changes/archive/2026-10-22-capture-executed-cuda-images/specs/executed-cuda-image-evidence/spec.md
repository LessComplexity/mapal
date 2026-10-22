## ADDED Requirements

### Requirement: Driver-PTX observation packets require complete identities

An observation packet SHALL preserve the exact authorized PTX and copied loaded cubin, and SHALL establish unambiguous context/module/function/launch identity joins and completed existing-harness status before admitting the packet as observation evidence.

#### Scenario: Complete driver-PTX observation
- **WHEN** exact authorized PTX, nonempty valid CUDAELF code for the expected kernel, complete chronological identity joins and successful harness completion are present
- **THEN** the packet is admitted as observation evidence with exact hashes and explicit accuracy limits

#### Scenario: Missing or ambiguous evidence
- **WHEN** initialization, loaded bytes, module/function/launch identity, successful completion or artifact integrity is missing, corrupt or ambiguous
- **THEN** the packet is rejected and production-JIT machine-code evidence remains absent

### Requirement: Observation is bounded and distinct from timing

Observation collection SHALL use finite explicit byte/event/time caps and exclusive output destinations, SHALL persist only authorized benchmark module code and metadata, and SHALL remain distinct from ordinary timing, library-fatbin and offline-assembly claims.

#### Scenario: Bounded isolated collection
- **WHEN** observation runs within its configured limits into a new destination
- **THEN** exact raw code and identity evidence are retained without runtime/compiler mutation or added synchronization and its timing is labeled inadmissible

#### Scenario: Overwrite, cap breach or wrong variant
- **WHEN** the output exists, a collection limit is exceeded, or offline cubin evidence is substituted for production driver-PTX observation
- **THEN** collection or admission fails and no ordinary timing or production-JIT claim is produced

### Requirement: Selected library calls retain complete dispatch attribution

A library observation SHALL bind an exact retained executable, selected implementation,
numerically checked completion and selected-call interval to every observed internal
dispatch. Each admitted entry SHALL have an unambiguous active context/module/function
association, callback-lifetime image copy and exact executable-entry identity. Callback
invocation identity SHALL remain distinct from API activity correlation and domain.

#### Scenario: Complete multi-kernel library call
- **WHEN** every selected dispatch has a supported direct loader/function association or unique active ELF entry plus authoritative function metadata, and all numerical and bounded-collection gates pass
- **THEN** all selected entries are retained with exact hashes, launch geometry, join provenance and the numerical guarantee actually checked

#### Scenario: Ambiguous invocation or incomplete selected inventory
- **WHEN** invocation tokens, API domains, context lifetimes, active module associations or selected dispatches are missing, contradictory, reused or ambiguous
- **THEN** the affected observation is rejected without deduplication, guessed identity or retrospective repair of a sealed failed trace

### Requirement: Decoder evidence preserves interpretation limits

A decoded entry SHALL be bound to the captured module and exact ELF entry, decoder
identity, command and output. Unknown annotations and unavailable resource fields SHALL
remain explicit. Static instructions SHALL NOT be represented as dynamic traffic,
achieved occupancy, executed operation counts or a causal performance explanation.

#### Scenario: Qualified successful disassembly
- **WHEN** exact-entry disassembly succeeds but contains unresolved annotations
- **THEN** supported instruction observations may be reported with those limits, while complete control encoding and dynamic performance remain unproved
