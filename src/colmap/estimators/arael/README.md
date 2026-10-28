# arael bundle adjustment backend

A bundle adjustment backend for COLMAP on [arael](https://crates.io/crates/arael),
a Levenberg-Marquardt solver with a Schur reduction and a supernodal
sparse Cholesky factorization, and the default backend of a build with
`ARAEL_ENABLED`. It solves the incremental mapper's
local and global bundle adjustments and the `bundle_adjuster` command
on the same problem the Ceres backend solves: the 2-row reprojection
residual under COLMAP's loss functions, every COLMAP camera model, rigs
with their sensor-from-rig poses, COLMAP's constant-parameter and
gauge-fixing settings, and position priors (`colmap pose_prior_mapper`,
`--Mapper.use_prior_position`) with the Ceres backend's alignment,
weighting and robust loss. A registration's pose refinement
(`RefineAbsolutePose`) stays with Ceres; the structure-less
registration's bundle adjustment follows the backend.

The backend is a model crate (`src/lib.rs`), from which arael's build
tool generates a C library and a C++ interface (`capi/`, `cxx/`, both
committed), and an adjuster over that interface
(`../bundle_adjustment_arael.cc`). The model has one camera type for
every camera model: the camera model is data, the parameters of every
model are fields, and a camera's Hessian blocks are stored at the size
of its free parameters. The model reproduces the Ceres backend's cost:
the tests compare the two backends' costs on every camera model,
including the observations a model cannot project, and solve the same
problems with both to compare the optima
(`../bundle_adjustment_arael_test.cc`).

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

With `ARAEL_ENABLED`, every backend option defaults to `ARAEL`
(`DefaultBundleAdjustmentBackend()`). Ceres on the command line:

```
colmap mapper ... --Mapper.ba_local_backend CERES --Mapper.ba_global_backend CERES
colmap bundle_adjuster ... --BundleAdjustment.backend CERES
colmap automatic_reconstructor ... --Mapper.ba_backend ceres
colmap global_mapper ... --GlobalMapper.ba_backend CERES
```

In the GUI, the backend selectors (automatic reconstruction, bundle
adjustment, the mapper's local and global backend) list the backends
built in.

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
for the mapper, `IncrementalPipelineOptions.ba_local_backend`,
`ba_global_backend` and `ba_arael`, which it passes to every solve
with its own iteration caps and thread count. An arael solve returns a
`pycolmap.AraelBundleAdjustmentSummary`: iterations, costs, the
termination, the linear solver's route, wall time and, with a record
or at log level 3, the per-phase times.

**Records.** `--BundleAdjustment.report_path <file>` appends one JSON
line per bundle adjustment of either backend: call site, backend,
problem size, iterations, costs, termination, threads, time.
The reconstruction benchmark (`benchmark/reconstruction/evaluate.py`)
takes `--ba_backend arael` and `--ba_report`.

**Camera models:** all of COLMAP's, EQUIRECTANGULAR included (its
image width and height stay constant, as in the Ceres backend). An
observation a model cannot project has a zero residual in both
backends: for most models a point behind the camera; the division
models project such points and fail only on a negative discriminant;
EQUIRECTANGULAR fails only for a point at the camera centre.

**Where the two backends differ.** arael's stop criterion is a
relative cost decrease (`rel_precision`, 1e-6); the mapper runs Ceres
with a gradient tolerance of 10 for the local and 1 for the global
adjustment and the function and parameter tolerances at 0. arael runs
every stage of a solve on the mapper's thread count; COLMAP runs Ceres
on one thread below its residual threshold.

## Benchmark

Two tables with different claims: the solver table times both
backends on the same bundle adjustment problems; the pipeline table
runs the mapper end to end with each backend as shipped.

**Setup.** One machine: AMD Ryzen 9 7950X3D (16 cores), 93 GB, Linux
6.8; an Ubuntu 24.04 container with COLMAP upstream `fa089062` and
this branch up to `84f59d05` (the backend and the benchmark changes),
built with `ARAEL_ENABLED=ON`; Ceres 2.2.0 and SuiteSparse 7.6 from
Ubuntu's packages, OpenBLAS 0.3.26, arael 0.9.3. Columns: `ceres` and
`arael` at 8 threads, `ceres-1` and `arael-1` at 1 thread, each backend
on its defaults; one run at a time.

**Scenes.** 34 scenes from the datasets of COLMAP's reconstruction
benchmark (`benchmark/reconstruction/`), named `dataset/category/scene`
as it names them. Each scene was extracted and matched once on the
machine, and every column of a scene starts from the same matches. A
scene runs in one or two camera setups: one camera per image, one
camera shared by all images, or one camera per image folder.

| dataset | scenes | camera setups |
|---|---|---|
| ETH3D DSLR | pipes, courtyard, delivery_area, facade, bridge | per image, shared |
| ETH3D rig | terrains | per folder, per image |
| ETH3D rig | delivery_area | per image |
| IMC 2023 | bike, kyiv-puppet-theater, cyprus, piazza_san_marco, sagrada_familia, dioscuri, british_museum, lincoln_memorial_statue, brandenburg_gate | per image |
| IMC 2024 | transp_obj_glass_cup, dioscuri, multi-temporal-temple-baalshamin, church, lizard | per image |
| BlendedMVS | 5b908d3dc6ab78485f3d24a9, 5ba19a8a360c7c30c1c169df, 5bf3a82cd439231948877aed, 5bb8a49aea1cfa39f1aa7f75 | per image, shared |
| BlendedMVS | 5b558a928bbfb62204e77ba2 | per image |
| TartanAir v2, panoramas rendered as perspective views | UrbanConstruction-easy-P000, OldTownWinter-hard-P001, House-easy-P000, Office-hard-P006, AbandonedFactory-easy-P000, Hospital-hard-P000, Gascola-easy-P003, Ruins-hard-P000 | per folder |

The ETH3D rig scenes run without their rig configuration: no rig
constraint between their cameras. Four of the scenes are large:
`imc2023/phototourism/brandenburg_gate`, `imc2024/all/lizard`,
`eth3d/rig/delivery_area` and `blended-mvs/BlendedMVS/5b558a928bbfb62204e77ba2`.

### Solver table: the same problem for every column

The problems are 81 global bundle adjustments of the mapper, three per
scene from 27 scenes: every scene except pipes, bike,
UrbanConstruction-easy-P000 and the four large ones. They come from
one Ceres mapper run per scene (one camera per folder for TartanAir
and `eth3d/rig/terrains`, one camera per image otherwise), which
writes snapshots of its reconstruction (`--Mapper.snapshot_path`)
after a registration and its local bundle adjustment: the state the
next global bundle adjustment receives. Per scene, the first snapshot
with at least 10 frames, the middle one and the last one ("early",
"middle", "late").

Each column solves the mapper's global bundle adjustment on the
snapshot (trivial loss, focal length and extra parameters refined,
principal point fixed), timed around building the adjuster and
solving, at the fewest iterations that bring its cost within 1e-5
relative of its own converged cost; median of 5 rounds. On 76 problems
every column reaches the same result: the RMS reprojection errors
agree to 5e-6 relative. The figures below are over those 76. On the
other five:

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

Three of the problems, each the late snapshot of its scene, time to
convergence in seconds:

| problem | frames | observations | ceres | ceres-1 | arael | arael-1 |
|---|---|---|---|---|---|---|
| imc2023/phototourism/sagrada_familia | 90 | 599745 | 2.87 | 3.62 | 0.43 | 0.95 |
| imc2023/phototourism/british_museum | 170 | 750309 | 4.67 | 6.26 | 0.69 | 1.52 |
| tartanair-v2-perspective/infrastructure/Hospital-hard-P000 | 50 | 99627 | 0.60 | 1.36 | 0.07 | 0.13 |

### Pipeline table: as shipped

The mapper run end to end with each backend on its defaults. Each
scene other than the four large ones runs in each of its camera
setups in all four columns, with two random seeds (one for pipes, bike
and UrbanConstruction-easy-P000): 76 runs per column. The four large
scenes run once, with one camera per image, in the 8-thread columns
only: 312 runs in all. It reports what a user of each backend gets:
registered images, pose accuracy (AUC, the area under the pose-error
curve, at 0.5, 1, 5 and 10 degrees), reprojection quantiles, mapper
wall time, bundle adjustment time from the records, peak memory.

The table is labelled as shipped because its columns do not solve the
same problems: the mapper's path diverges after the first solve.

Totals over the 76 runs of every column:

| | ceres | ceres-1 | arael | arael-1 |
|---|---|---|---|---|
| mapper [s] | 7248 | 12296 | 2282 | 3738 |
| bundle adjustment, global [s] | 4026 | 8347 | 576 | 1512 |
| bundle adjustment, local [s] | 1184 | 1618 | 317 | 694 |
| bundle adjustment share of the mapper | 72% | 81% | 39% | 59% |

How many times longer Ceres took than arael on the same run at the
same thread count, geometric mean over the runs of each camera setup:

| camera setup | mapper, 8 threads | mapper, 1 thread | global bundle adjustment, 8 threads | global bundle adjustment, 1 thread |
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

arael's peak memory divided by Ceres's, per run: 0.95 (geometric
mean), 0.40 to 1.14.

Quality: the median difference in AUC at 5 degrees between the two
8-thread columns is 0.00 over their 80 runs, and both register the
same number of images on 73 of them. The runs that differ go both
ways; the one where arael is clearly worse is `imc2024/all/lizard`
with one camera per image (one seed): AUC5 36.4 against 56.1. On that
run 26 of arael's 200 global solves ran to the iteration cap. At 1
thread the columns register the same number of images on 69 of the 76
runs; the one where arael is clearly worse is `imc2023/haiper/bike`
with one camera per image, seed 0: AUC5 1.9 against 51.0, 13 of 15
images registered against 15.

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
