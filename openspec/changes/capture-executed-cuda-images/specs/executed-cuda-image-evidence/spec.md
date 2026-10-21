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
