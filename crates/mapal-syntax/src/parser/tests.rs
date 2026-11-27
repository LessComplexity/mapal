//! Parser unit tests (DESIGN §20 item 4): the §3.6 precedence examples verbatim,
//! the precedence/associativity table, hole-expression shapes, every §17 ledger
//! row, the §14.4 classification table, ADR-0011 scan cases, the termination-rule
//! matrix, trailing commas, `ret.0` targets, the guard-payload F-matrix,
//! `category` recovery, and the pinned design-review regressions.

use super::*;
use crate::loc::LineIndex;

mod blocks;
mod helpers;
mod ledger;
mod precedence;
mod recovery;
mod stages;

use self::helpers::*;
