//! [`GpuRungs`]: the GPU tile kernel's rungs as a set (one `bool` per rung).
//!
//! One documented `bool` per rung makes a struct longer than a file, so its
//! fields are declared in two parts, in declaration order: `kernel_rounds`
//! (S52 through the `calls` rounds) and `device_rounds` (onto-device on).
//! Each part is a `macro_rules!` that appends its fields to the list it is
//! handed and passes the list to the next macro named in `[..]`;
//! `gpu_rungs_struct!` declares the struct from the whole list.

#[macro_use]
mod kernel_rounds;
#[macro_use]
mod device_rounds;

/// Declares [`GpuRungs`] from the field list the parts hand on (the chain's
/// last link).
macro_rules! gpu_rungs_struct {
    ([] $($fields:tt)*) => {
        /// The GPU tile kernel's rungs (S52), as a set: `true` = the rung is in it.
        ///
        /// **What ships is the rung rules over the part's physical properties**
        /// (hardware-properties, [`TargetProfile::gpu_rungs`], `rules.rs`): a site's
        /// fill against the part's SMs, its face, and properties such as FFMA per
        /// DRAM byte, registers per resident thread, L2 per SM and measured launch
        /// costs — never a part's name or version. [`GpuArms`] forces a rung on or off
        /// so its effect can be measured — an A/B needs both arms of the same
        /// compiler, the `MovePanel::Off` precedent; an arm is a measurement flag,
        /// never a per-part default. Values are identical either way: every rung
        /// moves operands, never the order a cell folds them.
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct GpuRungs {
            $($fields)*
        }
    };
}

kernel_rounds!([device_rounds, gpu_rungs_struct]);
