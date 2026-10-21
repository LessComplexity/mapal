## MODIFIED Requirements

### Requirement: Optimization preserves evaluation outcome
Optimization SHALL preserve successful values and outputs, including observations inside loops and computed loop-exit values, and SHALL preserve whether evaluation traps, when both evaluations have sufficient execution budget. Optimizing an accepted loop with observations or a computed exit value SHALL NOT fail with an internal reconstruction error.

#### Scenario: Simplified guard leaves a trap unselected
- **WHEN** a program succeeds because a potentially trapping computation belongs only to an unselected guard arm and surrounding expressions simplify
- **THEN** the optimized program succeeds with the same value and output

#### Scenario: Selected trap remains observable
- **WHEN** evaluation reaches an out-of-bounds access in a selected guard arm
- **THEN** the optimized program also traps

#### Scenario: Unconditional trap remains observable
- **WHEN** a trapping computation is unconditional even though its value is unused
- **THEN** optimization preserves that trap

#### Scenario: Observed loop exits with a value
- **WHEN** an accepted terminating loop prints during its iterations and exits with a value
- **THEN** optimization completes without an internal error and preserves every printed output and the exit value

#### Scenario: Loop exits without advancing
- **WHEN** an accepted loop exits on its initial condition with a computed value and its advance path contains an observation or trap
- **THEN** optimization preserves the exit value and does not introduce the unexecuted observation or trap
