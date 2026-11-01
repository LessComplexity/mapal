# gpu-validation Specification

## Purpose
Make a requested GPU correctness smoke run produce an unambiguous success or failure result before performance work begins.

## Requirements

### Requirement: GPU smoke checks its requested correctness scope
A differential-only GPU run SHALL execute all bundled value and trap cases, SHALL require nonempty value coverage, and SHALL succeed only when every case passes. It SHALL skip performance benchmarks and benchmark-harness compilation.

#### Scenario: Complete passing smoke
- **WHEN** a nonempty bundle produces the expected value or expected trap for every case
- **THEN** the smoke returns success and records completion with the pass counts

#### Scenario: Wrong value or wrong trap outcome
- **WHEN** any bundled case produces a wrong value, unexpected exit status, or missing expected trap
- **THEN** the smoke returns failure and does not record successful completion

#### Scenario: Empty correctness scope
- **WHEN** the bundle contains no value-comparison cases
- **THEN** the smoke returns failure

### Requirement: Rental execution preserves remote failure and cleanup
The rental runner SHALL retain available result logs and report remote execution failure to its caller, while preserving its normal instance cleanup behavior.

#### Scenario: Successful remote run
- **WHEN** remote execution succeeds and results are retrieved
- **THEN** the runner returns success and cleans up its temporary instance

#### Scenario: Failed remote run
- **WHEN** remote execution fails
- **THEN** the runner attempts result retrieval, returns failure and cleans up its temporary instance

### Requirement: Timed programs preserve execution failures
The full GPU runner SHALL record the process status of both ordinary and traced probe,
end-to-end and decode executions, and SHALL fail when either execution returns nonzero.

#### Scenario: Plausible output followed by failure
- **WHEN** a timed program prints plausible values or timing and then exits unsuccessfully
- **THEN** its record includes the failed status and the runner stops without successful completion

#### Scenario: Traced execution fails
- **WHEN** the ordinary execution succeeds but the traced execution fails
- **THEN** the record preserves both statuses and the runner returns failure
