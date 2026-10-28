// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/util/enum_utils.h"

#include <memory>
#include <string>

namespace colmap {

// arael's Levenberg-Marquardt presets: damping driver, termination
// tolerances and iteration defaults tuned for a class of problems.
MAKE_ENUM_CLASS_OVERLOAD_STREAM(AraelBundleAdjustmentPreset,
                                0,
                                DEFAULTS,
                                CONSERVATIVE,
                                WELL_CONDITIONED,
                                ILL_CONDITIONED);

// The damping schedule: the preset's own, or Nielsen's gain-ratio
// driver on the preset's other settings. On a poor start the fixed
// ladder of the other presets can reject every other step where the
// gain-ratio driver holds its steps; near a solution the two take the
// same iterations.
MAKE_ENUM_CLASS_OVERLOAD_STREAM(AraelDriver, 0, PRESET, NIELSEN);

// Whether each LM step eliminates the points (a Schur complement)
// before factorizing: AUTO lets arael's analysis decide, FORCE and
// NEVER settle it.
MAKE_ENUM_CLASS_OVERLOAD_STREAM(AraelSchurPolicy, 0, AUTO, FORCE, NEVER);

// How the reduced camera system is solved: factorized, or by
// preconditioned conjugate gradients on the assembled reduced system
// or on its implicit form. The iterative choices need a reduction.
MAKE_ENUM_CLASS_OVERLOAD_STREAM(
    AraelSchurSolve, 0, FACTORIZE, ITERATIVE, ITERATIVE_IMPLICIT);

// Elimination ordering of the factorized system.
MAKE_ENUM_CLASS_OVERLOAD_STREAM(
    AraelOrdering, 0, AUTO, AMD, MARGINALIZE_FIRST, NATURAL, NESTED_DISSECTION);

// A route arael picks by analysis unless told otherwise.
MAKE_ENUM_CLASS_OVERLOAD_STREAM(AraelRouteMode, 0, AUTO, ALWAYS, NEVER);

// arael-specific bundle adjustment options.
struct AraelBundleAdjustmentOptions {
  // arael's damping and termination preset. Bundle adjustment starts
  // from a good estimate, where the well-conditioned preset converges
  // in the fewest iterations.
  AraelBundleAdjustmentPreset preset =
      AraelBundleAdjustmentPreset::WELL_CONDITIONED;

  // The damping schedule on top of the preset.
  AraelDriver driver = AraelDriver::NIELSEN;

  // Initial LM damping; a non-positive value keeps the preset's.
  double initial_lambda = -1.0;

  // Maximum number of LM iterations.
  int max_num_iterations = 100;

  // Relative cost improvement below which an accepted step counts as
  // small (arael's rel_precision). The presets' 1e-4 stops a step
  // short of the Ceres backend's optimum; 1e-6 reaches it at the cost
  // of at most one more iteration. A non-positive value keeps the
  // preset's.
  double rel_precision = 1e-6;

  // Floor under the damping scale of a parameter with no curvature
  // (a point every observation of which is zeroed), so it stays put
  // instead of ending the solve; the Ceres backend's min_lm_diagonal.
  // A non-positive value leaves the floor off.
  double min_diagonal = 1e-6;

  // Threads for every stage of the solve. -1 uses every core, as the
  // Ceres backend's num_threads does.
  int num_threads = -1;

  // The sparse backend's route (arael's SparseOptions). The reduction
  // is forced: on every bundle adjustment shape measured, AUTO's
  // weighing of the two routes costs more than it ever saves.
  AraelSchurPolicy schur_policy = AraelSchurPolicy::FORCE;
  AraelSchurSolve schur_solve = AraelSchurSolve::FACTORIZE;
  // Iteration cap of the conjugate-gradient solves.
  int max_cg_iterations = 200;
  AraelOrdering ordering = AraelOrdering::AUTO;
  // The reduced system's factorization: the supernodal block Cholesky,
  // or the envelope (banded) route.
  AraelRouteMode block_supernodal = AraelRouteMode::AUTO;
  AraelRouteMode envelope = AraelRouteMode::AUTO;

  bool Check() const {
    CHECK_OPTION_GT(max_num_iterations, 0);
    CHECK_OPTION(num_threads == -1 || num_threads > 0);
    CHECK_OPTION_GT(max_cg_iterations, 0);
    return true;
  }
};

std::unique_ptr<BundleAdjuster> CreateDefaultAraelBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    Reconstruction& reconstruction);

// Bundle adjustment with position priors on arael, as the Ceres backend's
// pose-prior adjuster: with at least three usable priors and an alignment
// to them, the reconstruction is aligned, normalized with its scale
// fixed, solved with a prior on the camera centre of every image whose
// pose is a parameter, and transformed back; otherwise a plain solve
// with the gauge fixed by two cameras.
std::unique_ptr<BundleAdjuster> CreatePosePriorAraelBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const PosePriorBundleAdjustmentOptions& prior_options,
    const BundleAdjustmentConfig& config,
    std::vector<PosePrior> pose_priors,
    Reconstruction& reconstruction);

#ifdef ARAEL_ENABLED
struct AraelBundleAdjustmentSummary : public BundleAdjustmentSummary {
  // What was solved, for the record: "bundle adjustment" or "pose prior
  // bundle adjustment".
  std::string kind = "bundle adjustment";
  // Number of optimized scalar parameters, rotations counted as three.
  int num_effective_parameters = 0;
  int num_iterations = 0;
  int num_accepted_iterations = 0;
  // The loss the costs were measured under (DescribeLossFunction).
  std::string loss;
  // Costs in the Ceres convention, 1/2 sum rho(|r|^2).
  double initial_cost = 0.0;
  double final_cost = 0.0;
  // arael's own termination status, e.g. "converged", "hit max_iters".
  std::string termination_reason;
  // The route the linear solves took, from arael's plan: the reduction
  // and the factorization of the reduced system.
  std::string linear_solver;
  // Threads the factorization ran on (0: every core).
  int num_threads = 0;
  // Wall time of the solve, in seconds.
  double wall_time = 0.0;
  // arael's own per-phase timing, in seconds, gathered when
  // BundleAdjustmentOptions::report_path is set or at log level 3.
  bool has_timing = false;
  double time_total = 0.0;
  double time_assembly = 0.0;
  double time_analysis = 0.0;
  double time_linear_solve = 0.0;
  double time_cost = 0.0;
  double time_advance = 0.0;
  int num_assemblies = 0;
  int num_linear_solves = 0;
  // The call site (BundleAdjustmentOptions::report_site), for the record.
  std::string site;

  std::string BriefReport() const override;
  // One line of JSON for BundleAdjustmentOptions::report_path.
  std::string JsonRecord() const;
};
#endif

}  // namespace colmap
