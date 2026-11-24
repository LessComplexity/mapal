//! Module frame (DESIGN §2–§4, §8): the `.cu` prelude (includes + `mapal-rt`
//! `extern "C"` decls + the trap-flag/error machinery), the `FlowProd_*`
//! struct definitions, the `Str` host globals, and the `main` wrapper.
//!
//! **Harness build recipe (DESIGN §4/§6):**
//! `nvcc -std=c++17 -fmad=false -arch=sm_89 prog.cu libmapal_rt.a -lpthread -ldl -lm -o prog`
//! `-fmad=false` pins device float parity with the interpreter oracle (no FMA
//! contraction); `--use_fast_math` and host `-march=native`/`-mfma` are
//! forbidden (§4). The link tail is the pinned Linux default (§6.6).
//!
//! Trap design (§3), host side: one `cudaMalloc`'d `unsigned int` device flag,
//! zeroed once per process by `cudaMemset` at `main` start. WP3's launch sites
//! pass it to trap-capable kernels and call `trap_check_after_launch()` after
//! every launch **that can trap** (#14's `TrapCaps` trim: provably trap-free
//! launches pass no flag and skip the readback) — `cudaGetLastError` (the
//! exit-102 infra protocol) plus a host-synchronizing D→H `cudaMemcpy` of the
//! flag; nonzero ⇒ `mapal_trap(kind - 1)` on the host (exit 101). The flag
//! stores the mapal-rt kind **plus one**: 0 must stay the quiescent value (the
//! memset zeroing), so a device div_zero guard stores `1u` and an index_oob
//! guard stores `2u`; the readback decodes to the mapal-rt encoding (0 =
//! div_zero, 1 = index_oob). A bare-kind store would collide — div_zero's 0
//! would read back as "no trap", crossing the R1 classes. Host-side scalar
//! traps (Div/Mod guards) call `mapal_trap` directly. Every
//! `cudaMalloc`/`cudaMemcpy` return is asserted by `cu_check`; on error the
//! process prints to stderr and exits **102** — the harness-visible
//! infra-failure class, never an R1 data point.

use std::collections::HashSet;

use mapal_ir::{CategoryIr, ObjectId, ObjectKind, Ty, Value};
use slotmap::SecondaryMap;

use crate::ty::{lower_ty, prod_shape};

/// The `.cu` prelude: includes, the `mapal-rt` `extern "C"` block (exact
/// signatures from `mapal-rt/src/lib.rs` — `usize` → `size_t`, `*const u8` →
/// `const uint8_t*`), and the trap-flag + exit-102 machinery (DESIGN §3).
/// `trap_check_after_launch` is WP3's launch-site hook; it stays
/// `[[maybe_unused]]` because scalar-only modules have no launches.
/// ADR-0042 / ADR-0046: the math builtins' C ([`math_helpers`]) is emitted
/// after the prelude when the program has one.
/// ADR-0043: the headers a program with narrow storage includes.
pub(crate) fn narrow_headers() -> &'static str {
    "\n#include <cuda_bf16.h>\n#include <cuda_fp16.h>\n"
}

/// ADR-0043: the rounding conversion into a narrow ty (round to nearest even).
pub(crate) fn narrow_fn(ty: &mapal_ir::Ty) -> &'static str {
    match ty {
        mapal_ir::Ty::Narrow(mapal_ir::NarrowFloat::Bf16) => "__float2bfloat16_rn",
        mapal_ir::Ty::Narrow(mapal_ir::NarrowFloat::F16) => "__float2half_rn",
        // ADR-0043 slice 3: the constructors round to nearest even and saturate.
        mapal_ir::Ty::Narrow(mapal_ir::NarrowFloat::F8e4m3) => "__nv_fp8_e4m3",
        mapal_ir::Ty::Narrow(mapal_ir::NarrowFloat::F8e5m2) => "__nv_fp8_e5m2",
        _ => unreachable!("Narrow into a non-narrow ty passed validation"),
    }
}

/// ADR-0046 D5/D7: what the CUDA backend's C declares — the correctly rounded
/// `sqrt` on both sides (`__fsqrt_rn` / `__dsqrt_rn` on the device, `sqrtf` /
/// `sqrt` on the host: IEEE `squareRoot`), so the exact faces take it; every
/// other builtin is its reference.
const CUDA_UNITS: [mapal_ir::math::Unit; 2] = [
    mapal_ir::math::Unit {
        op: mapal_ir::math::UnitOp::Sqrt,
        bits: 32,
        sym: "MAPAL_SQRT32",
        err: mapal_ir::math::Err::Exact,
        lo: 0.0,
        hi: f64::INFINITY,
        cost: 4,
        specials: true,
        evidence: "IEEE 754 squareRoot: __fsqrt_rn / sqrtf",
    },
    mapal_ir::math::Unit {
        op: mapal_ir::math::UnitOp::Sqrt,
        bits: 64,
        sym: "MAPAL_SQRT64",
        err: mapal_ir::math::Err::Exact,
        lo: 0.0,
        hi: f64::INFINITY,
        cost: 4,
        specials: true,
        evidence: "IEEE 754 squareRoot: __dsqrt_rn / sqrt",
    },
];
pub(crate) const CUDA_MATH: mapal_ir::math::Target = mapal_ir::math::Target {
    units: &CUDA_UNITS,
    defs: &[],
};

/// ADR-0042 / ADR-0046: `mapal_<builtin>_f32` / `_f64` for every builtin and
/// width the program uses — the exact face's realization
/// ([`mapal_ir::math::choose`] over [`CUDA_MATH`]) printed from its trace: on
/// the device each float step an explicit round-to-nearest intrinsic, so
/// `-fmad=true` cannot contract it and the device's bits are the
/// interpreter's.
pub(crate) fn math_helpers(used: &std::collections::BTreeSet<(mapal_ir::MathFn, u32)>) -> String {
    use mapal_ir::math::{self, B2, Cvt, F2, FPred, Face, I2, IPred, Op, VTy, W};
    let mut s = String::from(
        "\n#include <cstring>\n\
         __host__ __device__ inline float mapal_f32_of(uint32_t b) { float f; memcpy(&f, &b, sizeof f); return f; }\n\
         __host__ __device__ inline double mapal_f64_of(uint64_t b) { double f; memcpy(&f, &b, sizeof f); return f; }\n\
         __host__ __device__ inline uint32_t mapal_bits32(float f) { uint32_t b; memcpy(&b, &f, sizeof b); return b; }\n\
         __host__ __device__ inline uint64_t mapal_bits64(double f) { uint64_t b; memcpy(&b, &f, sizeof b); return b; }\n\
         #ifdef __CUDA_ARCH__\n\
         #define MAPAL_MUL32(a, b) __fmul_rn(a, b)\n#define MAPAL_ADD32(a, b) __fadd_rn(a, b)\n#define MAPAL_SUB32(a, b) __fsub_rn(a, b)\n\
         #define MAPAL_MUL64(a, b) __dmul_rn(a, b)\n#define MAPAL_ADD64(a, b) __dadd_rn(a, b)\n#define MAPAL_SUB64(a, b) __dsub_rn(a, b)\n\
         #define MAPAL_FMA32(a, b, c) __fmaf_rn(a, b, c)\n#define MAPAL_FMA64(a, b, c) __fma_rn(a, b, c)\n\
         #define MAPAL_SQRT32(a) __fsqrt_rn(a)\n#define MAPAL_SQRT64(a) __dsqrt_rn(a)\n\
         #define MAPAL_MULHI64(a, b) __umul64hi(a, b)\n\
         #define MAPAL_CLZ32(a) ((uint32_t)__clz((int)(a)))\n#define MAPAL_CLZ64(a) ((uint64_t)__clzll((long long)(a)))\n\
         #else\n\
         #define MAPAL_MUL32(a, b) ((a) * (b))\n#define MAPAL_ADD32(a, b) ((a) + (b))\n#define MAPAL_SUB32(a, b) ((a) - (b))\n\
         #define MAPAL_MUL64(a, b) ((a) * (b))\n#define MAPAL_ADD64(a, b) ((a) + (b))\n#define MAPAL_SUB64(a, b) ((a) - (b))\n\
         #define MAPAL_FMA32(a, b, c) fmaf(a, b, c)\n#define MAPAL_FMA64(a, b, c) fma(a, b, c)\n\
         #define MAPAL_SQRT32(a) sqrtf(a)\n#define MAPAL_SQRT64(a) sqrt(a)\n\
         #define MAPAL_MULHI64(a, b) ((uint64_t)(((unsigned __int128)(a) * (b)) >> 64))\n\
         #define MAPAL_CLZ32(a) ((uint32_t)((a) ? __builtin_clz(a) : 32))\n#define MAPAL_CLZ64(a) ((uint64_t)((a) ? __builtin_clzll(a) : 64))\n\
         #endif\n",
    );
    for &(f, bits) in used {
        let c = math::choose(f, bits, Face::Exact, false, true, &CUDA_MATH);
        let def = math::trace(&c, &CUDA_MATH);
        let w = bits;
        let (t, ut, of, bitsf) = if bits == 64 {
            ("double", "uint64_t", "mapal_f64_of", "mapal_bits64")
        } else {
            ("float", "uint32_t", "mapal_f32_of", "mapal_bits32")
        };
        let cty = |v: VTy| match v {
            VTy::F => t,
            VTy::I32 => "uint32_t",
            VTy::I64 => "uint64_t",
            VTy::B => "bool",
        };
        let names: Vec<String> = def
            .steps
            .iter()
            .enumerate()
            .map(|(i, st)| match st.op {
                Op::Arg => "x".to_string(),
                Op::CF(b) => format!("{of}(0x{b:x}{})", if bits == 64 { "ull" } else { "u" }),
                Op::CI(W::I32, v) => format!("((uint32_t)0x{v:x}u)"),
                Op::CI(W::I64, v) => format!("((uint64_t)0x{v:x}ull)"),
                _ => format!("v{i}"),
            })
            .collect();
        s.push_str(&format!(
            "__host__ __device__ inline {t} mapal_{}_f{w}({t} x) {{\n",
            f.name()
        ));
        for (i, st) in def.steps.iter().enumerate() {
            let a = |k: usize| names[st.args[k].0 as usize].as_str();
            let wn = |w: W| if w == W::I64 { 64 } else { 32 };
            let e = match st.op {
                Op::Arg | Op::CF(_) | Op::CI(..) => continue,
                Op::F2(op) => {
                    let o = match op {
                        F2::Add => "ADD",
                        F2::Sub => "SUB",
                        F2::Mul => "MUL",
                    };
                    format!("MAPAL_{o}{w}({}, {})", a(0), a(1))
                }
                Op::FNeg => format!("(-({}))", a(0)),
                Op::Fma => format!("MAPAL_FMA{w}({}, {}, {})", a(0), a(1), a(2)),
                Op::FCmp(p) => match p {
                    FPred::Ogt => format!("({} > {})", a(0), a(1)),
                    FPred::Olt => format!("({} < {})", a(0), a(1)),
                    FPred::Oeq => format!("({} == {})", a(0), a(1)),
                    FPred::Uno => format!("({0} != {0} || {1} != {1})", a(0), a(1)),
                },
                Op::I2(iw, op) => {
                    let it = if iw == W::I64 { "uint64_t" } else { "uint32_t" };
                    let st_ = if iw == W::I64 { "int64_t" } else { "int32_t" };
                    match op {
                        I2::Add => format!("(({it})({} + {}))", a(0), a(1)),
                        I2::Sub => format!("(({it})({} - {}))", a(0), a(1)),
                        I2::Mul => format!("(({it})({} * {}))", a(0), a(1)),
                        I2::MulHi if iw == W::I64 => format!("MAPAL_MULHI64({}, {})", a(0), a(1)),
                        I2::MulHi => format!("((uint32_t)(((uint64_t){} * {}) >> 32))", a(0), a(1)),
                        I2::And => format!("(({it})({} & {}))", a(0), a(1)),
                        I2::Or => format!("(({it})({} | {}))", a(0), a(1)),
                        I2::Xor => format!("(({it})({} ^ {}))", a(0), a(1)),
                        I2::Shl => format!("(({it})({} << {}))", a(0), a(1)),
                        I2::LShr => format!("(({it})({} >> {}))", a(0), a(1)),
                        I2::AShr => format!("(({it})(({st_}){} >> {}))", a(0), a(1)),
                    }
                }
                Op::Clz(iw) => format!("MAPAL_CLZ{}({})", wn(iw), a(0)),
                Op::ICmp(iw, p) => {
                    let st_ = if iw == W::I64 { "int64_t" } else { "int32_t" };
                    match p {
                        IPred::Eq => format!("({} == {})", a(0), a(1)),
                        IPred::Ne => format!("({} != {})", a(0), a(1)),
                        IPred::Ult => format!("({} < {})", a(0), a(1)),
                        IPred::Ugt => format!("({} > {})", a(0), a(1)),
                        IPred::Slt => format!("(({st_}){} < ({st_}){})", a(0), a(1)),
                        IPred::Sgt => format!("(({st_}){} > ({st_}){})", a(0), a(1)),
                    }
                }
                Op::B2(op) => match op {
                    B2::And => format!("({} && {})", a(0), a(1)),
                    B2::Or => format!("({} || {})", a(0), a(1)),
                    B2::Xor => format!("({} != {})", a(0), a(1)),
                },
                Op::Sel => format!("({} ? {} : {})", a(0), a(1), a(2)),
                Op::Cvt(cv) => match cv {
                    Cvt::Bits => format!("{bitsf}({})", a(0)),
                    Cvt::FromBits => format!("{of}(({ut}){})", a(0)),
                    Cvt::FToSi(W::I32) => format!("((uint32_t)(int32_t){})", a(0)),
                    Cvt::FToSi(W::I64) => format!("((uint64_t)(int64_t){})", a(0)),
                    Cvt::SiToF(W::I32) => format!("(({t})(int32_t){})", a(0)),
                    Cvt::SiToF(W::I64) => format!("(({t})(int64_t){})", a(0)),
                    Cvt::UiToF(W::I32) => format!("(({t})(uint32_t){})", a(0)),
                    Cvt::UiToF(W::I64) => format!("(({t})(uint64_t){})", a(0)),
                    Cvt::ZExt => format!("((uint64_t){})", a(0)),
                    Cvt::Trunc => format!("((uint32_t){})", a(0)),
                    Cvt::BoolToI(iw) => format!("((uint{}_t)({} ? 1 : 0))", wn(iw), a(0)),
                },
                Op::Unit(_, sym) => format!("{sym}({})", a(0)),
            };
            s.push_str(&format!("  const {} v{i} = {e};\n", cty(st.ty)));
        }
        s.push_str(&format!("  return {};\n}}\n", names[def.out.0 as usize]));
    }
    s
}

pub(crate) const PRELUDE: &str = r#"#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cmath>
#include <cuda_runtime.h>

extern "C" {
void mapal_print_i32(int32_t v, bool newline);
void mapal_print_i64(int64_t v, bool newline);
void mapal_print_u8(uint8_t v, bool newline);
void mapal_print_bool(bool v, bool newline);
void mapal_print_f32(float v, bool newline);
void mapal_print_f64(double v, bool newline);
void mapal_print_str(const uint8_t* ptr, size_t len, bool newline);
// Matches the Rust `-> !` (mapal-rt/src/lib.rs): lets the compiler drop the
// dead fall-through after host guards. C++11 attribute, legal on an
// extern "C" declaration; host-only, so no device-pass concern.
[[noreturn]] void mapal_trap(uint32_t kind);
}

// --- trap flag + CUDA error protocol (DESIGN §3) ---------------------------
static unsigned int* d_trap = nullptr;

// Assert one CUDA API return; on error print to stderr and exit 102 (the
// harness-visible infra-failure class — never an R1 data point).
static void cu_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "flow-cuda infra error at %s: %s\n", what, cudaGetErrorString(err));
        exit(102);
    }
}

// Zero the trap flag — once per process, called at main start (H→D, §2 item 3).
static void trap_init() {
    cu_check(cudaMalloc((void**)&d_trap, sizeof(unsigned int)), "cudaMalloc(d_trap)");
    cu_check(cudaMemset(d_trap, 0, sizeof(unsigned int)), "cudaMemset(d_trap)");
}

// Trap-capable launch sites call this after EVERY such kernel launch (#14:
// a provably trap-free kernel takes no trap argument and skips this
// readback): cudaGetLastError
// (exit-102 protocol), then a host-synchronizing D→H read of the flag (the
// memcpy is the sync point); nonzero kind ⇒ mapal_trap(kind - 1) on the host
// (exit 101) — the flag stores kind + 1 (0 = quiescent after trap_init's
// memset; 1 = div_zero, 2 = index_oob), decoded here to the mapal-rt kinds.
[[maybe_unused]] static void trap_check_after_launch() {
    cu_check(cudaGetLastError(), "kernel launch");
    unsigned int kind = 0;
    cu_check(cudaMemcpy(&kind, d_trap, sizeof(unsigned int), cudaMemcpyDeviceToHost),
             "cudaMemcpy(d_trap)");
    if (kind != 0) {
        mapal_trap(kind - 1);
    }
}
"#;

/// A private `Str` constant global: its symbol name and byte length. `Print`
/// of a `Str` passes the pointer + explicit `len` to `mapal_print_str` (DESIGN
/// §1; the C array NUL-terminates, but mapal-rt reads exactly `len` bytes).
pub(crate) struct StrGlobal {
    pub name: String,
    pub bytes: Vec<u8>,
}

/// A residual ≥ 2 product shape to define: the `FlowProd_*` name and the
/// lowered component texts in surviving (erased-index) order — plus its
/// C-layout size (kernel::abi_sizeof), emitted as a `static_assert` guard
/// against ABI drift between the arena's offset arithmetic and nvcc's
/// `sizeof` (plan-smart-arenas §8's belt-and-suspenders, checked on the box
/// leg where the TU compiles).
pub(crate) struct ProdStruct {
    pub name: String,
    pub fields: Vec<String>,
    pub size: u64,
}

/// Collect every `Str` constant object → a private global (DESIGN §2). One
/// global per object, named `strN` by deterministic object order (the llvm
/// scheme, re-spelled for C++).
pub(crate) fn collect_str_globals(ir: &CategoryIr) -> SecondaryMap<ObjectId, StrGlobal> {
    let mut out: SecondaryMap<ObjectId, StrGlobal> = SecondaryMap::new();
    let mut n = 0usize;
    for (id, obj) in ir.objects() {
        if obj.kind == ObjectKind::Constant
            && obj.ty == Ty::Str
            && let Some(Value::Str(s)) = &obj.value
        {
            out.insert(
                id,
                StrGlobal {
                    name: format!("str{n}"),
                    bytes: s.clone().into_bytes(),
                },
            );
            n += 1;
        }
    }
    out
}

/// Emit the `Str` globals block (deterministic object order via the
/// SecondaryMap): `static const char strN[] = "…";`.
pub(crate) fn emit_str_globals(globals: &SecondaryMap<ObjectId, StrGlobal>) -> String {
    let mut out = String::new();
    for (_, g) in globals.iter() {
        out.push_str(&format!(
            "static const char {}[] = \"{}\";\n",
            g.name,
            escape_bytes(&g.bytes),
        ));
    }
    out
}

/// C string-literal escaping: printable ASCII (except `"` and `\`) verbatim,
/// everything else as a 3-digit octal escape. Octal, not `\xHH`: C hex
/// escapes greedily consume following hex digits (`"\x41" "B"` ≠ `"\x41B"`),
/// octal stops at 3 digits. (Trigraphs don't exist in C++17 — `?` is safe.)
fn escape_bytes(bytes: &[u8]) -> String {
    let mut s = String::new();
    for &b in bytes {
        match b {
            b'"' => s.push_str("\\\""),
            b'\\' => s.push_str("\\\\"),
            0x20..=0x7e => s.push(b as char),
            _ => s.push_str(&format!("\\{b:03o}")),
        }
    }
    s
}

/// Collect every residual ≥ 2 product shape appearing in the IR's object tys
/// (fn input/output tys are objects, so one pass covers both), deduped by
/// lowered name, in deterministic first-appearance order with **inner shapes
/// before outer** (C++ needs a field's struct defined before its use). Arrays
/// recurse to their element (an array of products is `FlowProd_*` AoS).
pub(crate) fn collect_prod_structs(ir: &CategoryIr) -> Vec<ProdStruct> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<ProdStruct> = Vec::new();
    for (_, obj) in ir.objects() {
        collect_ty(&obj.ty, &mut seen, &mut out);
    }
    out
}

/// Post-order shape walk: surviving components first (so nested products
/// define before their parents), then `ty` itself if it names a struct.
fn collect_ty(ty: &Ty, seen: &mut HashSet<String>, out: &mut Vec<ProdStruct>) {
    match ty {
        Ty::Array { elem, .. } => collect_ty(elem, seen, out),
        Ty::Tuple(ts) => {
            for c in ts {
                if lower_ty(c).is_some() {
                    collect_ty(c, seen, out);
                }
            }
            collect_self(ty, seen, out);
        }
        Ty::Struct { fields, .. } => {
            for (_, c) in fields {
                if lower_ty(c).is_some() {
                    collect_ty(c, seen, out);
                }
            }
            collect_self(ty, seen, out);
        }
        _ => {}
    }
}

fn collect_self(ty: &Ty, seen: &mut HashSet<String>, out: &mut Vec<ProdStruct>) {
    if let Some((name, fields)) = prod_shape(ty)
        && seen.insert(name.clone())
    {
        let size = crate::kernel::abi_sizeof(ty).expect("residual ≥ 2 product has a size");
        out.push(ProdStruct { name, fields, size });
    }
}

/// Emit the `FlowProd_*` definitions: `struct Name { t0 f0; t1 f1; … };` —
/// fields named by erased slot index (the ty.rs contract) — each followed by
/// a `static_assert` pinning the arena's ABI size model against the target
/// compiler's `sizeof` (plan-smart-arenas §8).
pub(crate) fn emit_prod_structs(prods: &[ProdStruct]) -> String {
    let mut out = String::new();
    for p in prods {
        out.push_str(&format!("struct {} {{\n", p.name));
        for (i, f) in p.fields.iter().enumerate() {
            out.push_str(&format!("    {f} f{i};\n"));
        }
        out.push_str("};\n");
        out.push_str(&format!(
            "static_assert(sizeof({0}) == {1}, \"{0}: abi_sizeof drift (plan-smart-arenas)\");\n",
            p.name, p.size
        ));
    }
    out
}

/// The `main` wrapper (DESIGN §4, llvm BL8 port). Calls `mapal_main` after
/// `trap_init`, prints a non-erased scalar return through `mapal-rt` with
/// newline = true (the `Unit → i32` closed shape, so the differential
/// observes it), frees the trap flag at exit, returns 0. Open entries get a
/// value-initialized argument so emission stays total (llvm's zeroinitializer
/// rule); an array-typed input gets `nullptr`.
pub(crate) fn emit_main_wrapper(ir: &CategoryIr) -> String {
    let entry = ir.entry();
    let fd = ir.func(entry).expect("sealed graph: entry resolves");
    let input_ty = &ir.object(fd.input).expect("input resolves").ty;
    let output_ty = &ir.object(fd.output).expect("output resolves").ty;

    let mut out = String::from("int main() {\n");
    out.push_str("  trap_init();\n");

    let arg = match lower_ty(input_ty) {
        None => String::new(),
        Some(t) if t.ends_with('*') => "nullptr".to_string(),
        Some(t) => format!("{t}{{}}"),
    };

    match lower_ty(output_ty) {
        None => {
            out.push_str(&format!("  mapal_main({arg});\n"));
        }
        Some(rty) => {
            out.push_str(&format!("  {rty} r = mapal_main({arg});\n"));
            if let Some(call) = print_call(output_ty, "r") {
                out.push_str(&format!("  {call}\n"));
            }
        }
    }
    // The flag is freed explicitly; a leak would be reclaimed by context
    // teardown anyway (DESIGN §2's main-return rule — recorded, not policed).
    out.push_str("  cu_check(cudaFree(d_trap), \"cudaFree(d_trap)\");\n");
    out.push_str("  return 0;\n}\n");
    out
}

/// The `mapal-rt` print statement for a scalar return value operand (BL8
/// result print). `None` for a type mapal-rt cannot print through this path
/// (aggregates, arrays — L1207).
fn print_call(ty: &Ty, operand: &str) -> Option<String> {
    let func = print_dispatch(ty)?;
    Some(format!("{func}({operand}, true);"))
}

/// The `mapal-rt` print function for a printable scalar, or `None` for
/// non-printables (aggregates/arrays — the wrapper simply doesn't print).
/// u8 routes to `mapal_print_u8`; no `zeroext` ceremony — the C++ ABI passes
/// `uint8_t`/`bool` natively.
pub(crate) fn print_dispatch(ty: &Ty) -> Option<&'static str> {
    match ty {
        Ty::Int { bits: 32, .. } => Some("mapal_print_i32"),
        Ty::Int { bits: 64, .. } => Some("mapal_print_i64"),
        Ty::Int { bits: 8, .. } => Some("mapal_print_u8"),
        Ty::Bool => Some("mapal_print_bool"),
        Ty::Float { bits: 32 } => Some("mapal_print_f32"),
        Ty::Float { bits: 64 } => Some("mapal_print_f64"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mapal_ir::{Dest, FuncKind, IrBuilder, SourceLoc};

    const L: SourceLoc = SourceLoc { start: 0, end: 0 };

    fn tup(ts: Vec<Ty>) -> Ty {
        Ty::Tuple(ts)
    }

    fn arr(elem: Ty, size: u64) -> Ty {
        Ty::Array {
            elem: Box::new(elem),
            size,
        }
    }

    fn collect_one(ty: &Ty) -> Vec<ProdStruct> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        collect_ty(ty, &mut seen, &mut out);
        out
    }

    #[test]
    fn prod_structs_dedupe_and_first_appearance_order() {
        let a = tup(vec![Ty::i32(), Ty::Bool]);
        let b = tup(vec![Ty::f64(), Ty::i64()]);
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        collect_ty(&a, &mut seen, &mut out);
        collect_ty(&b, &mut seen, &mut out);
        collect_ty(&a, &mut seen, &mut out); // duplicate shape: deduped
        let names: Vec<&str> = out.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["FlowProd_int32_t_bool", "FlowProd_double_int64_t"]);
        assert_eq!(out[0].fields, ["int32_t", "bool"]);
        assert_eq!(out[1].fields, ["double", "int64_t"]);
    }

    #[test]
    fn prod_structs_nested_define_inner_first() {
        // Tuple[Tuple[i32,i32], bool] — the inner struct must precede the
        // outer (C++ definitional order).
        let inner = tup(vec![Ty::i32(), Ty::i32()]);
        let outer = tup(vec![inner, Ty::Bool]);
        let out = collect_one(&outer);
        let names: Vec<&str> = out.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "FlowProd_int32_t_int32_t",
                "FlowProd_FlowProd_int32_t_int32_t_bool"
            ]
        );
        assert_eq!(out[1].fields, ["FlowProd_int32_t_int32_t", "bool"]);
    }

    #[test]
    fn prod_structs_recurse_through_arrays_and_residual_one() {
        // Array of products → the element shape is collected (AoS handle).
        let out = collect_one(&arr(tup(vec![Ty::i32(), Ty::Bool]), 4));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "FlowProd_int32_t_bool");
        // A residual-1 wrapper still recurses into its surviving product
        // component: Tuple[Tuple[i32,i32], Unit] is bare, but the inner
        // struct is needed.
        let bare = tup(vec![tup(vec![Ty::i32(), Ty::i32()]), Ty::Unit]);
        let out = collect_one(&bare);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "FlowProd_int32_t_int32_t");
        // Fully erased / residual-1 products name no struct at all.
        assert!(collect_one(&tup(vec![Ty::IoToken, Ty::i32()])).is_empty());
        assert!(collect_one(&Ty::Unit).is_empty());
    }

    #[test]
    fn prod_struct_emission_shape() {
        let out = collect_one(&tup(vec![Ty::i32(), Ty::Bool]));
        let text = emit_prod_structs(&out);
        assert_eq!(
            text,
            "struct FlowProd_int32_t_bool {\n    int32_t f0;\n    bool f1;\n};\n\
             static_assert(sizeof(FlowProd_int32_t_bool) == 8, \
             \"FlowProd_int32_t_bool: abi_sizeof drift (plan-smart-arenas)\");\n"
        );
    }

    #[test]
    fn str_escaping_quotes_backslashes_and_non_printables() {
        // Printable ASCII verbatim; `"` and `\` escaped; everything else a
        // 3-digit octal escape.
        assert_eq!(escape_bytes(b"hi there"), "hi there");
        assert_eq!(escape_bytes(b"say \"hi\""), "say \\\"hi\\\"");
        assert_eq!(escape_bytes(b"c:\\tmp"), "c:\\\\tmp");
        assert_eq!(escape_bytes(b"a\nb\t"), "a\\012b\\011");
        // The greedy-hex trap: byte 0x01 followed by 'A' (a hex digit) —
        // "\x01A" would parse as ONE escape; octal "\001A" cannot.
        assert_eq!(escape_bytes(&[0x01, b'A']), "\\001A");
        // High bytes (UTF-8 continuation) octal-escape too.
        assert_eq!(escape_bytes(&[0xC3, 0xA9]), "\\303\\251");
        assert_eq!(escape_bytes(b""), "");
    }

    fn lower_src(src: &str) -> CategoryIr {
        let po = mapal_syntax::parse(src);
        assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
        mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"))
    }

    #[test]
    fn str_globals_collect_and_emit() {
        let ir = lower_src("fn main() {\n    \"hi\" -> print;\n}\n");
        let globals = collect_str_globals(&ir);
        let text = emit_str_globals(&globals);
        assert_eq!(text, "static const char str0[] = \"hi\";\n");
        // The explicit byte length rides along for mapal_print_str.
        let g = globals.iter().next().unwrap().1;
        assert_eq!(g.bytes.len(), 2);
    }

    /// A hand-built `main` with the given in/out tys; `val = Some(c)` returns
    /// the constant, `None` returns the input (Unit → Unit shape).
    fn build_main(in_ty: Ty, out_ty: Ty, val: Option<Value>) -> CategoryIr {
        let mut b = IrBuilder::new();
        let f = b
            .declare(FuncKind::Named, "main", in_ty, out_ty, L)
            .unwrap();
        {
            let mut fb = b.build_fn(f).unwrap();
            match val {
                Some(v) => {
                    let c = fb.constant(v, L).unwrap();
                    fb.output(c, None, L).unwrap();
                }
                None => {
                    let i = fb.input();
                    fb.output(i, None, L).unwrap();
                }
            }
            fb.finish().unwrap();
        }
        b.seal(f).unwrap()
    }

    #[test]
    fn main_wrapper_prints_scalar_return() {
        let ir = build_main(Ty::Unit, Ty::i32(), Some(Value::I32(42)));
        let w = emit_main_wrapper(&ir);
        assert_eq!(
            w,
            "int main() {\n  trap_init();\n  int32_t r = mapal_main();\n  \
             mapal_print_i32(r, true);\n  \
             cu_check(cudaFree(d_trap), \"cudaFree(d_trap)\");\n  return 0;\n}\n"
        );
    }

    #[test]
    fn main_wrapper_erased_return_just_calls() {
        let ir = build_main(Ty::Unit, Ty::Unit, None);
        let w = emit_main_wrapper(&ir);
        assert!(w.contains("  mapal_main();\n"), "{w}");
        assert!(!w.contains("mapal_print"), "{w}");
    }

    #[test]
    fn main_wrapper_open_input_value_initializes() {
        let ir = build_main(Ty::i32(), Ty::i32(), None);
        let w = emit_main_wrapper(&ir);
        assert!(w.contains("mapal_main(int32_t{})"), "{w}");
        // An array input is a handle: nullptr, not a braced value.
        let ir = build_main(arr(Ty::i32(), 4), arr(Ty::i32(), 4), None);
        let w = emit_main_wrapper(&ir);
        assert!(w.contains("mapal_main(nullptr)"), "{w}");
    }

    #[test]
    fn main_wrapper_does_not_free_returned_array() {
        // §2: an array returned from main is reclaimed by context teardown
        // (recorded, not leak-policed) — the wrapper must NOT cudaFree the
        // returned handle (mapal_main already transferred the duty). (main
        // cannot declare a return in surface syntax — L1002 — so IrBuilder.)
        let mut b = IrBuilder::new();
        let f = b
            .declare(FuncKind::Named, "main", Ty::Unit, arr(Ty::i32(), 2), L)
            .unwrap();
        {
            let mut fb = b.build_fn(f).unwrap();
            let c1 = fb.constant(Value::I32(1), L).unwrap();
            let c2 = fb.constant(Value::I32(2), L).unwrap();
            let a = fb.pack_array(&[c1, c2], Dest::Fresh(None), L).unwrap();
            fb.output(a, None, L).unwrap();
            fb.finish().unwrap();
        }
        let ir = b.seal(f).unwrap();
        let w = emit_main_wrapper(&ir);
        assert!(w.contains("int32_t* r = mapal_main();"), "{w}");
        assert!(!w.contains("cudaFree(r)"), "{w}");
        // And mapal_main's own epilogue must not free it either (the escape).
        let cu = crate::emit(&ir).unwrap();
        let main_start = cu
            .find("static int32_t* mapal_main() {")
            .expect("mapal_main def");
        let main_end = cu[main_start..].find("\n}\n").unwrap() + main_start;
        let def = &cu[main_start..main_end];
        for line in def.lines().filter(|l| l.contains("cudaFree")) {
            assert!(
                !line.contains("cu_check(cudaFree(o"),
                "the returned buffer is never freed by its allocator:\n{def}"
            );
        }
    }

    #[test]
    fn prelude_has_rt_decls_and_exit_102_protocol() {
        // Exact mapal-rt signatures (mapal-rt/src/lib.rs); mapal_trap is
        // [[noreturn]] — the Rust `-> !` (F9).
        for decl in [
            "void mapal_print_i32(int32_t v, bool newline);",
            "void mapal_print_i64(int64_t v, bool newline);",
            "void mapal_print_u8(uint8_t v, bool newline);",
            "void mapal_print_bool(bool v, bool newline);",
            "void mapal_print_f32(float v, bool newline);",
            "void mapal_print_f64(double v, bool newline);",
            "void mapal_print_str(const uint8_t* ptr, size_t len, bool newline);",
            "[[noreturn]] void mapal_trap(uint32_t kind);",
        ] {
            assert!(PRELUDE.contains(decl), "missing: {decl}");
        }
        // §3: flag malloc+memset once per process; launch check; exit 102.
        assert!(PRELUDE.contains("cudaMalloc((void**)&d_trap"));
        assert!(PRELUDE.contains("cudaMemset(d_trap, 0, sizeof(unsigned int))"));
        assert!(PRELUDE.contains("cudaGetLastError()"));
        assert!(PRELUDE.contains("exit(102)"));
    }

    #[test]
    fn prelude_trap_check_decodes_kind_plus_one() {
        // §3's flag encoding: the flag stores kind + 1 (0 = quiescent after
        // the memset; 1 = div_zero, 2 = index_oob); the host readback decodes
        // to the mapal-rt kinds 0/1 — the exact decode text, pinned (a bare
        // `mapal_trap(kind)` would misread a device div_zero's 0-store as
        // "no trap" and index_oob's 1 as div_zero).
        assert!(PRELUDE.contains("mapal_trap(kind - 1);"), "{PRELUDE}");
        // trap_init is unchanged: memset 0 stays the quiescent value.
        assert!(PRELUDE.contains("cudaMemset(d_trap, 0, sizeof(unsigned int))"));
    }
}
