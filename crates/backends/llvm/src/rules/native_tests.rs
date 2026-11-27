//! Tests of the native GPU profile: the calibration line and device query.

use crate::profile::Face;
use crate::profile::{
    CostOver, DeviceCost, DeviceQuery, NATIVE_GPU, RTX_5080, calib_line, cuda_query,
};

/// The local box's calibration line (committed beside its evidence).
const CALIB: &str =
    include_str!("../../../../../benches/results/evidence/local-rtx-5080/calib/rtx-5080.calib");

/// The RTX 5080's device query, as `cuDeviceGetAttribute` reports it
/// (`calib.cu` header and `devq.txt`, 2026-11-19 / 21).
fn rtx_5080_query() -> DeviceQuery {
    DeviceQuery::from_attrs(&[
        (1, 1024),
        (10, 32),
        (13, 2_640_000),
        (16, 84),
        (36, 15_001_000),
        (37, 256),
        (38, 67_108_864),
        (39, 1536),
        (75, 12),
        (76, 0),
        (81, 102_400),
        (82, 65536),
        (97, 101_376),
        (106, 24),
        (111, 1024),
    ])
}

/// A part's placement facts without their provenance's wording.
fn place_values(f: &crate::profile::PlaceFacts) -> crate::profile::PlaceFacts {
    crate::profile::PlaceFacts {
        prov: &[],
        cite: "",
        lat_l2_cite: "",
        start: f.start.map(|(s, _)| (s, "")),
        ..*f
    }
}

/// **hardware-properties: the native GPU profile is the device query and
/// the calibration file** — on the 5080's facts it is the hand spelling's
/// physical properties, so it takes the same rungs at every fill and
/// face; a line keyed by other facts never prices it.
#[test]
fn native_gpu_is_the_query_and_the_calibration() {
    let q = rtx_5080_query();
    assert_eq!(q.calib_key(), "cc=120 sms=84 l2=67108864 mem=15001000x256");
    let p = q.profile(Some(CALIB));
    assert_eq!(p.name, NATIVE_GPU);
    let (g, h) = (p.gpu().expect("gpu"), RTX_5080.gpu().expect("gpu"));
    // The physical properties and the placement probes' facts (ADR-0050
    // `PlaceFacts`: start timing, keeps, hit curve, latencies, efficiency
    // points, sustained clock — the calibration line's `place.` names)
    // are the hand profile's; only the provenance's wording differs (the
    // calibration file against the hand profile's cites).
    assert_eq!(
        place_values(g.place),
        place_values(h.place),
        "the calibration line carries the placement facts"
    );
    assert_eq!(
        crate::profile::Gpu {
            place: h.place,
            ..*g
        },
        *h,
        "the query + calibration spell the hand profile"
    );
    for wps in [0, 1, 2, 4, 64] {
        for face in [Face::Exact, Face::Contract, Face::Tf32] {
            assert_eq!(p.gpu_rungs_on(wps, face), RTX_5080.gpu_rungs_on(wps, face));
        }
    }
    // Another part's facts: no line, no measured costs, the query's
    // clock — and the rules still decide from what is there.
    let other = DeviceQuery { sms: 170, ..q };
    let o = other.profile(Some(CALIB));
    let og = o.gpu().expect("gpu");
    assert_eq!(
        (og.cost, og.clock_khz, og.fp32_lanes),
        (None, 2_640_000, 128)
    );
    assert!(!o.gpu_rungs(0, true).pdl && !o.gpu_rungs(0, true).pinned);
    // A line's names are the `--cost=` names; an unknown one rejects it.
    let over: CostOver = calib_line("k\tpdl_launch_ns=1 fp32_lanes=64", "k").expect("line");
    let at = |n: &str| over[DeviceCost::NAMES.iter().position(|x| *x == n).unwrap()];
    assert_eq!((at("pdl_launch_ns"), at("fp32_lanes")), (Some(1), Some(64)));
    assert!(calib_line("k\tbogus=1", "k").is_none());
}

/// On a machine with a CUDA device: device 0's query builds a profile
/// whose facts are the driver's; on the local RTX 5080 it is the hand
/// spelling's. Without a driver, nothing to check.
#[test]
fn native_gpu_reads_the_driver() {
    let Some(a) = cuda_query::attrs(0) else {
        return;
    };
    let q = DeviceQuery::from_attrs(&a);
    assert!(q.sms > 0 && q.l2_bytes > 0 && q.regs_per_sm > 0 && q.mem_bus_bits > 0);
    if q == rtx_5080_query() {
        let p = q.profile(Some(CALIB));
        let h = RTX_5080.gpu().expect("gpu");
        assert_eq!(
            place_values(p.gpu().expect("gpu").place),
            place_values(h.place)
        );
        assert_eq!(
            crate::profile::Gpu {
                place: h.place,
                ..*p.gpu().expect("gpu")
            },
            *h
        );
    }
}
