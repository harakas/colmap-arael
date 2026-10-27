# arael bundle adjustment backend

A bundle adjustment backend for COLMAP on [arael](https://crates.io/crates/arael),
a Levenberg-Marquardt solver with a Schur reduction and a supernodal
sparse Cholesky factorization, selected with
`--BundleAdjustment.backend ARAEL`. It solves the incremental mapper's
local and global bundle adjustments and the `bundle_adjuster` command
on the same problem the Ceres backend solves: the 2-row reprojection
residual under COLMAP's loss functions, the camera models below, rigs
with their sensor-from-rig poses, COLMAP's constant-parameter and
gauge-fixing settings. A registration's pose refinement
(`RefineAbsolutePose`) and the pose-prior bundle adjuster stay with
Ceres; the structure-less registration's bundle adjustment follows the
backend. The Ceres backend remains the default.

The backend is a model crate (`src/lib.rs`), from which arael's build
tool generates a C library and a C++ interface (`capi/`, `cxx/`, both
committed), and an adjuster over that interface
(`../bundle_adjustment_arael.cc`). The model has one camera type for
every camera model: the camera model is data, the parameters of every
model are fields, and a camera's Hessian blocks are stored at the size
of its free parameters. The model reproduces the Ceres backend's cost:
a parity test solves the same problems with both and compares the
optima (`../bundle_adjustment_arael_test.cc`).

## Building

```
cmake -S . -B build -DARAEL_ENABLED=ON ...
```

The build needs a Rust toolchain (`cargo`) and builds the model crate's
C library, which the COLMAP libraries link. `ARAEL_ENABLED` is off by
default. The model needs arael 0.9.3 or later.

After a change to the model (`src/lib.rs`), regenerate the interface
with [cargo-arael](https://crates.io/crates/cargo-arael) and commit it:

```
cd src/colmap/estimators/arael && cargo arael export
```

The crate's `Cargo.toml` limits the export to the C++ interface
(`interfaces = ["cxx"]`: `cxx/` and the `capi/` crate it builds on;
cargo-arael 0.9.3 or later).

## Using it

Command line, with the mapper or the `bundle_adjuster` command:

```
colmap mapper ... --BundleAdjustment.backend ARAEL
```

The backend's options:

| option | default | meaning |
|---|---|---|
| `BundleAdjustmentArael.preset` | `WELL_CONDITIONED` | arael's damping and termination preset (`DEFAULTS`, `CONSERVATIVE`, `WELL_CONDITIONED`, `ILL_CONDITIONED`) |
| `BundleAdjustmentArael.driver` | `NIELSEN` | the damping schedule: the gain-ratio driver, or the preset's own fixed ladder (`PRESET`) |
| `BundleAdjustmentArael.initial_lambda` | preset's | initial damping |
| `BundleAdjustmentArael.max_num_iterations` | 100 | iteration cap; the mapper passes its own local and global caps |
| `BundleAdjustmentArael.rel_precision` | 1e-6 | relative cost decrease below which an accepted step counts as small |
| `BundleAdjustmentArael.min_diagonal` | 1e-6 | floor under the damping scale of a parameter without curvature |
| `BundleAdjustmentArael.num_threads` | -1 | threads for every stage of the solve; -1 is every core, and the mapper passes its own thread count |
| `BundleAdjustmentArael.schur_policy` | `FORCE` | eliminate the points before factorizing (`AUTO`, `FORCE`, `NEVER`) |
| `BundleAdjustmentArael.schur_solve` | `FACTORIZE` | how the reduced camera system is solved (`FACTORIZE`, `ITERATIVE`, `ITERATIVE_IMPLICIT`) |
| `BundleAdjustmentArael.max_cg_iterations` | 200 | iteration cap of the conjugate-gradient solves |
| `BundleAdjustmentArael.ordering` | `AUTO` | elimination ordering (`AUTO`, `AMD`, `MARGINALIZE_FIRST`, `NATURAL`, `NESTED_DISSECTION`) |
| `BundleAdjustmentArael.block_supernodal`, `.envelope` | `AUTO` | the reduced system's factorization route (`AUTO`, `ALWAYS`, `NEVER`) |

In pycolmap the same options are `BundleAdjustmentOptions.backend`
(`pycolmap.BundleAdjustmentBackend.ARAEL`) and
`BundleAdjustmentOptions.arael` (`pycolmap.AraelBundleAdjustmentOptions`);
for the mapper, `IncrementalPipelineOptions.ba_backend` and
`IncrementalPipelineOptions.ba_arael`, which it passes to every solve
with its own iteration caps and thread count.

**Records.** `--BundleAdjustment.report_path <file>` appends one JSON
line per bundle adjustment of either backend: call site, backend,
problem size, iterations, costs, termination, threads, time.
The reconstruction benchmark (`benchmark/reconstruction/evaluate.py`)
takes `--ba_backend arael` and `--ba_report`.

**Camera models:** SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL,
OPENCV, FULL_OPENCV, OPENCV_FISHEYE, THIN_PRISM_FISHEYE and
RAD_TAN_THIN_PRISM_FISHEYE. Creating an adjuster over another model
fails with a message naming the camera.

**Where the two backends differ.** arael's stop criterion is a
relative cost decrease (`rel_precision`, 1e-6); the mapper runs Ceres
with a gradient tolerance of 10 for the local and 1 for the global
adjustment and the function and parameter tolerances at 0. arael runs
every stage of a solve on the mapper's thread count; COLMAP runs Ceres
on one thread below its residual threshold.

## Benchmark

One machine, one image, two tables with two different claims.

**Setup.** AMD Ryzen 9 7950X3D (16 cores), 93 GB, Linux 6.8; an
Ubuntu 24.04 image with COLMAP upstream `fa089062` plus this tree's
changes as of 2026-10-26, built with the backend on; Ceres 2.2.0 and
SuiteSparse 7.6 from Ubuntu's packages, OpenBLAS 0.3.26, arael from
source at `f7875d6f`, past 0.9.2. The mapper and every solve
run on at most 8 threads, one run at a time. The scenes are from the
reconstruction benchmark's datasets (ETH3D, IMC 2023 and 2024,
BlendedMVS, TartanAir v2), extracted and matched on the machine itself;
every column of a scene starts from the same matches. Columns: `ceres`
and `arael` at 8 threads, `ceres-1` and `arael-1` at 1 thread, each
backend on its defaults. The image also builds the model's previous
layout, one camera type per camera model, on the same arael; it runs
as `arael-per-model` and `arael-per-model-1` in the solver table and
on the four tier-3 scenes.

### Solver table: the same problem for every column

The problem is the mapper's own global bundle adjustment on a mapper
snapshot: three per tier-2 scene (early, middle, end), 81 problems.
Each column is timed around building the adjuster and solving, at the
fewest iterations that bring its cost within 1e-5 relative of its own
converged cost, median of 5 rounds. On 76 problems every column reaches
the same result: the RMS reprojection errors agree to 5e-6 relative.
The figures below are over those 76. On the other five:

| problem | Ceres: iterations to its result, cost | arael: iterations to its result, cost |
|---|---|---|
| imc2024/all/transp_obj_glass_cup, early | 34, 200.546 | 529, 193.393 |
| imc2024/all/transp_obj_glass_cup, middle | 726, 112.829 | 374, 112.831 |
| imc2024/all/transp_obj_glass_cup, late | 1001 (at the cap), 469.239 | 403, 479.065 |
| imc2024/all/dioscuri, early | 5, 1792.66 | 16, 1792.49 |
| imc2023/phototourism/sagrada_familia, early | 1001 (at the cap), 54706.8 | 489, 54849.6 |

| | ceres | ceres-1 | arael | arael-1 |
|---|---|---|---|---|
| time to convergence, 76 problems [s] | 58.8 | 86.4 | 10.2 | 21.7 |
| problems converged in 1 or 2 iterations | 51 | 51 | 69 | 69 |

How many times longer Ceres took than arael on the same problem:

| | geometric mean | min | max |
|---|---|---|---|
| both at 8 threads | 5.4 | 1.9 | 34.5 |
| both at 1 thread | 4.0 | 1.4 | 15.3 |
| Ceres at 8 threads, arael at 1 | 3.0 | 0.8 | 18.8 |

The model's previous layout against this one, on all 81 problems: the
same iteration count on 80 at 8 threads and 79 at 1 thread; the
converged costs differ by 6e-16 (8 threads) and 7e-16 (1 thread)
relative in the median, 8e-6 at most. This layout's time over the
previous one's:

| | 8 threads | 1 thread |
|---|---|---|
| total over the 81 problems | 1.016 | 1.022 |
| per problem, geometric mean (min to max) | 1.005 (0.72 to 1.30) | 1.031 (0.88 to 1.19) |
| the 24 problems of 200k observations or more | 1.028 (0.96 to 1.09) | 1.021 (0.94 to 1.08) |

Three of the problems, each the last snapshot of its scene, time to
convergence in seconds:

| problem | frames | observations | ceres | ceres-1 | arael | arael-1 |
|---|---|---|---|---|---|---|
| imc2023/phototourism/sagrada_familia | 90 | 599745 | 2.87 | 3.62 | 0.43 | 0.95 |
| imc2023/phototourism/british_museum | 170 | 750309 | 4.67 | 6.26 | 0.69 | 1.52 |
| tartanair-v2-perspective/infrastructure/Hospital-hard-P000 | 50 | 99627 | 0.60 | 1.36 | 0.07 | 0.13 |

### Pipeline table: as shipped

The mapper run end to end on every scene, each backend on its
defaults: tier 1 (four form-scenes, one seed), tier 2 (36 form-scenes,
two seeds), tier 3 (four large scenes, one seed, the 8-thread columns
only), 312 cells. It reports what a user of each backend gets:
registered images, pose accuracy (AUC, the area under the pose-error
curve, at 0.5, 1, 5 and 10 degrees), reprojection quantiles, mapper
wall time, bundle adjustment time from the records, peak memory.

The table is labelled as shipped because its columns do not solve the
same problems: the mapper's path diverges after the first solve.

Totals over the 76 cells both thread counts share:

| | ceres | ceres-1 | arael | arael-1 |
|---|---|---|---|---|
| mapper [s] | 7248 | 12296 | 2282 | 3738 |
| bundle adjustment, global [s] | 4026 | 8347 | 576 | 1512 |
| bundle adjustment, local [s] | 1184 | 1618 | 317 | 694 |
| bundle adjustment share of the mapper | 72% | 81% | 39% | 59% |

How many times longer Ceres took than arael on the same cell at the
same thread count, geometric mean over the cells of each camera form:

| form | mapper, 8 threads | mapper, 1 thread | global bundle adjustment, 8 threads | global bundle adjustment, 1 thread |
|---|---|---|---|---|
| one camera per folder (TartanAir, ETH3D rig) | 3.0 | 3.0 | 6.6 | 6.0 |
| one camera per image | 2.7 | 2.5 | 5.5 | 4.3 |
| one shared camera | 2.4 | 1.9 | 4.7 | 3.6 |

Three of the scenes, one camera per image, mapper wall time in seconds
(means over the two seeds):

| scene | images | ceres | ceres-1 | arael | arael-1 |
|---|---|---|---|---|---|
| imc2023/phototourism/sagrada_familia | 90 | 499 | 1146 | 120 | 300 |
| imc2023/phototourism/british_museum | 176 | 525 | 847 | 144 | 215 |
| eth3d/rig/terrains | 660 | 395 | 800 | 87 | 145 |

arael's peak memory divided by Ceres's, per cell: 0.95 (geometric
mean), 0.40 to 1.14.

Quality: the median difference in AUC at 5 degrees between the two
8-thread columns is 0.00 over the 80 cells, and both register the same
number of images on 73 of them. The cells that differ go both ways; the
one where arael is clearly worse is `imc2024/all/lizard` with one
camera per image (tier 3, one seed): AUC5 36.4 against 56.1. On that
cell 26 of arael's 200 global solves ran to the iteration cap. The
model's previous layout, from the same matches and seed, ended that
cell at AUC5 55.5 (150 global solves, 16 at the cap). At 1 thread the
columns register the same number of images on 69 of the 76 cells; the
one where arael is clearly worse is `imc2023/haiper/bike` with one
camera per image, seed 0: AUC5 1.9 against 51.0, 13 of 15 images
registered against 15.

The ETH3D rig scenes run without their rig configuration in both
forms: four cameras, no rig constraint.

## What the fork changes

- The backend: the model crate and its generated interface, the
  adjuster, the options, the pycolmap bindings, the tests.
- Bundle adjustment records for both backends
  (`BundleAdjustment.report_path`).
- The mapper passes its thread count, the record path and the backend
  options to every solve; `automatic_reconstructor` accepts every
  `Mapper` option.
- The reconstruction benchmark: the backend flag, the records and
  their summary, pass-through COLMAP arguments, peak memory,
  reprojection quantiles, bounded library thread pools, matching
  skipped on a matched database; panorama scenes take the mapper
  flags and render on the given thread count.
- Fixes: the automatic reconstructor dropped the log level; the
  structure-less registration's pose refinement ignored the backend.
