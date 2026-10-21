## Purpose

Make each backend's developer emission command independently addressable so building another backend cannot replace the executable a benchmark invokes.

## ADDED Requirements

### Requirement: Backend emitter examples have distinct executable names
The developer interface SHALL expose the LLVM emitter as `emit` and the CUDA emitter as `emit_cuda`, preserving each emitter's existing output format and command arguments.

#### Scenario: Both backends are built
- **WHEN** a developer builds all workspace examples
- **THEN** `emit` produces LLVM text and `emit_cuda` produces CUDA source from the same supported source program
- **AND** neither build overwrites the other backend's executable

#### Scenario: Legacy CUDA example name is requested
- **WHEN** a developer asks the CUDA package to run the old `emit` example name
- **THEN** the build tool rejects the nonexistent example rather than running an executable from another backend

#### Scenario: Missing source argument
- **WHEN** either emitter is invoked without a source argument
- **THEN** it reports usage for its own executable and exits unsuccessfully
