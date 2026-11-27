//! The rejection matrix (DESIGN §14.4): ≥1 test per L-code of §4 except L1901
//! (only a lower bug can raise it). Each program is parse-clean — lower owns the
//! rejection. Includes the named gotcha regressions from §14.4.

mod common;
use common::{assert_rejects, lower_err_codes, lower_ok};

#[path = "rejection/chains_guards.rs"]
mod chains_guards;
#[path = "rejection/loops_folds.rs"]
mod loops_folds;
#[path = "rejection/seq_regressions.rs"]
mod seq_regressions;
#[path = "rejection/structure.rs"]
mod structure;
#[path = "rejection/updates_widening.rs"]
mod updates_widening;
