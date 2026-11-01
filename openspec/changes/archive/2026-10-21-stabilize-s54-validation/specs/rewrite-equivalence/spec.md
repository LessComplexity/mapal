## Purpose

Keep optimized Mapal programs observationally equivalent to their original evaluation, including guard-controlled traps.

## ADDED Requirements

### Requirement: Optimization preserves evaluation outcome
Optimization SHALL preserve successful values and outputs and SHALL preserve whether evaluation traps, when both evaluations have sufficient execution budget.

#### Scenario: Simplified guard leaves a trap unselected
- **WHEN** a program succeeds because a potentially trapping computation belongs only to an unselected guard arm and surrounding expressions simplify
- **THEN** the optimized program succeeds with the same value and output

#### Scenario: Selected trap remains observable
- **WHEN** evaluation reaches an out-of-bounds access in a selected guard arm
- **THEN** the optimized program also traps

#### Scenario: Unconditional trap remains observable
- **WHEN** a trapping computation is unconditional even though its value is unused
- **THEN** optimization preserves that trap
