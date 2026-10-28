// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/bundle_adjustment_arael.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/ceres_loss_function.h"
#include "colmap/sensor/models.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <set>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arael/eigen.hpp>
#include <ba_problem.hpp>

namespace colmap {
namespace {

// The generated interface of the model.
struct Model {
  using Scalar = double;
  using Problem = colmap_arael::BaProblem;
  using Pose = colmap_arael::Pose;
  using Point = colmap_arael::Point;
  using LmConfig = colmap_arael::LmConfig;
  using SparseOptions = colmap_arael::SparseOptions;
  using SolveResult = colmap_arael::SolveResult;
  using SolveError = colmap_arael::SolveError;
  using LmResult = colmap_arael::LmResult;
};

// A rotation counts as its three tangent parameters.
constexpr int kRotationDof = 3;

// The supported COLMAP camera models, in the order of the model
// crate's projection: a camera entity's `model` is the index here.
const std::vector<CameraModelId>& SupportedModels() {
  static const std::vector<CameraModelId> models = {
      PinholeCameraModel::model_id,
      SimplePinholeCameraModel::model_id,
      SimpleRadialCameraModel::model_id,
      RadialCameraModel::model_id,
      OpenCVCameraModel::model_id,
      FullOpenCVCameraModel::model_id,
      OpenCVFisheyeCameraModel::model_id,
      ThinPrismFisheyeCameraModel::model_id,
      RadTanThinPrismFisheyeModel::model_id,
      SimpleRadialFisheyeCameraModel::model_id,
      RadialFisheyeCameraModel::model_id,
      FOVCameraModel::model_id,
      SimpleFisheyeCameraModel::model_id,
      FisheyeCameraModel::model_id,
      SkewedPinholeCameraModel::model_id,
      SimpleDivisionCameraModel::model_id,
      DivisionCameraModel::model_id,
      EUCMCameraModel::model_id,
      EquirectangularCameraModel::model_id,
  };
  return models;
}

// The camera entity's `check`, the test of whether its model projects a
// point, by the `model` index.
uint32_t ProjectionCheck(const uint32_t model) {
  switch (SupportedModels()[model]) {
    case SimpleDivisionCameraModel::model_id:
    case DivisionCameraModel::model_id:
      return 1;
    case EUCMCameraModel::model_id:
      return 2;
    case EquirectangularCameraModel::model_id:
      return 3;
    default:
      return 0;
  }
}

std::string SupportedCameraModels() {
  std::string names;
  for (const CameraModelId model_id : SupportedModels()) {
    names += (names.empty() ? "" : ", ") + CameraModelIdToName(model_id);
  }
  return names;
}

// The camera's `model` index; throws for a model the backend lacks.
uint32_t CheckSupportedCameraModel(const Camera& camera) {
  const std::vector<CameraModelId>& models = SupportedModels();
  const auto it = std::find(models.begin(), models.end(), camera.model_id);
  if (it == models.end()) {
    LOG(FATAL_THROW) << "arael BA backend supports the camera models "
                     << SupportedCameraModels() << "; camera "
                     << camera.camera_id << " is " << camera.ModelName();
  }
  return static_cast<uint32_t>(it - models.begin());
}

// Three fixed points pin the gauge when they span three dimensions.
struct FixedGaugeWithThreePoints {
  Eigen::Index num_fixed_points = 0;
  Eigen::Matrix3d fixed_points = Eigen::Matrix3d::Zero();
  bool MaybeAddFixedPoint(const Eigen::Vector3d& point) {
    if (num_fixed_points >= 3) {
      return false;
    }
    fixed_points.col(num_fixed_points) = point;
    if (fixed_points.colPivHouseholderQr().rank() > num_fixed_points) {
      ++num_fixed_points;
      return true;
    } else {
      fixed_points.col(num_fixed_points).setZero();
      return false;
    }
  }
};

// The extra parameters of a camera handle (`Cam` or `RigCam`), by
// index in COLMAP's order.
template <typename Cam>
struct Extras {
  static constexpr size_t kNum = 12;
#define COLMAP_ARAEL_EXTRAS(M) \
  M(e0) M(e1) M(e2) M(e3) M(e4) M(e5) M(e6) M(e7) M(e8) M(e9) M(e10) M(e11)
#define COLMAP_ARAEL_SET(e) &Cam::set_##e,
#define COLMAP_ARAEL_GET(e) &Cam::e,
#define COLMAP_ARAEL_SET_OPTIMIZE(e) &Cam::set_##e##_optimize,
#define COLMAP_ARAEL_OPTIMIZE(e) &Cam::e##_optimize,
  static constexpr void (Cam::* kSet[kNum])(double) = {
      COLMAP_ARAEL_EXTRAS(COLMAP_ARAEL_SET)};
  static constexpr double (Cam::* kGet[kNum])() const = {
      COLMAP_ARAEL_EXTRAS(COLMAP_ARAEL_GET)};
  static constexpr void (Cam::* kSetOptimize[kNum])(bool) = {
      COLMAP_ARAEL_EXTRAS(COLMAP_ARAEL_SET_OPTIMIZE)};
  static constexpr bool (Cam::* kOptimize[kNum])() const = {
      COLMAP_ARAEL_EXTRAS(COLMAP_ARAEL_OPTIMIZE)};
#undef COLMAP_ARAEL_EXTRAS
#undef COLMAP_ARAEL_SET
#undef COLMAP_ARAEL_GET
#undef COLMAP_ARAEL_SET_OPTIMIZE
#undef COLMAP_ARAEL_OPTIMIZE
};

// Which intrinsics of a camera handle are optimized, by COLMAP's
// refinement groups (focal length, principal point, extra parameters).
// `fy` of a model with one focal length, the extra parameters past the
// model's own and the metadata parameters stay constant, as do the
// focal lengths and principal point of a model without them.
template <typename Cam>
void SetIntrinsicsOptimize(Cam cam,
                           const Camera& camera,
                           const bool focal,
                           const bool pp,
                           const bool extra) {
  const size_t num_focal = camera.FocalLengthIdxs().size();
  cam.set_fx_optimize(focal && num_focal > 0);
  cam.set_fy_optimize(focal && num_focal == 2);
  cam.set_pp_optimize(pp && !camera.PrincipalPointIdxs().empty());
  const size_t num_extra = camera.ExtraParamsIdxs().size();
  for (size_t i = 0; i < Extras<Cam>::kNum; ++i) {
    (cam.*Extras<Cam>::kSetOptimize[i])(extra && i < num_extra);
  }
}

// A COLMAP camera's model and intrinsics into a camera handle, every
// intrinsic constant. A model with one focal length has it in `fx`
// and, unused, in `fy`. The extra parameters, then the metadata
// parameters (EQUIRECTANGULAR's image width and height), fill `e0`
// onwards. A model without a focal length or principal point leaves
// them at 1 and 0, unused.
template <typename Cam>
void SetIntrinsics(Cam cam, const Camera& camera) {
  const std::vector<double>& p = camera.params;
  const span<const size_t> focal = camera.FocalLengthIdxs();
  const span<const size_t> pp = camera.PrincipalPointIdxs();
  const span<const size_t> extra = camera.ExtraParamsIdxs();
  const span<const size_t> metadata = camera.MetaDataParamsIdxs();
  THROW_CHECK_LE(extra.size() + metadata.size(), Extras<Cam>::kNum);
  const uint32_t model = CheckSupportedCameraModel(camera);
  cam.set_model(model);
  cam.set_check(ProjectionCheck(model));
  cam.set_fx(focal.empty() ? 1.0 : p[focal[0]]);
  cam.set_fy(focal.empty() ? 1.0 : p[focal[focal.size() - 1]]);
  if (pp.empty()) {
    cam.set_pp({0.0, 0.0});
  } else {
    cam.set_pp({p[pp[0]], p[pp[1]]});
  }
  for (size_t i = 0; i < extra.size(); ++i) {
    (cam.*Extras<Cam>::kSet[i])(p[extra[i]]);
  }
  for (size_t i = 0; i < metadata.size(); ++i) {
    (cam.*Extras<Cam>::kSet[extra.size() + i])(p[metadata[i]]);
  }
  SetIntrinsicsOptimize(cam, camera, false, false, false);
}

// The camera handle's intrinsics into the COLMAP camera's parameters;
// the metadata parameters are never optimized and stay as they are.
template <typename Cam>
void GetIntrinsics(Cam cam, Camera& camera) {
  std::vector<double>& p = camera.params;
  const span<const size_t> focal = camera.FocalLengthIdxs();
  const span<const size_t> pp = camera.PrincipalPointIdxs();
  const span<const size_t> extra = camera.ExtraParamsIdxs();
  if (!focal.empty()) {
    p[focal[0]] = cam.fx();
  }
  if (focal.size() == 2) {
    p[focal[1]] = cam.fy();
  }
  if (!pp.empty()) {
    p[pp[0]] = cam.pp().x;
    p[pp[1]] = cam.pp().y;
  }
  for (size_t i = 0; i < extra.size(); ++i) {
    p[extra[i]] = (cam.*Extras<Cam>::kGet[i])();
  }
}

// Optimized intrinsics of a camera handle.
template <typename Cam>
int NumIntrinsicParams(Cam cam) {
  int num = cam.fx_optimize() + cam.fy_optimize() + 2 * cam.pp_optimize();
  for (const auto optimize : Extras<Cam>::kOptimize) {
    num += (cam.*optimize)();
  }
  return num;
}

template <typename Scalar, typename Cam>
void SetRigPose(Cam cam, const Rigid3d& sensor_from_rig) {
  cam.set_rot(arael::quatern<Scalar>::from_eigen(
      Eigen::Quaterniond(sensor_from_rig.rotation()).normalized()));
  cam.set_t(arael::vect3<Scalar>::from_eigen(
      Eigen::Vector3d(sensor_from_rig.translation())));
}

template <typename Cam>
void GetRigPose(Cam cam, Rigid3d& sensor_from_rig) {
  sensor_from_rig.rotation() =
      cam.rot().to_eigen().template cast<double>().normalized();
  sensor_from_rig.translation() = cam.t().to_eigen().template cast<double>();
}

template <typename Cam>
void SetRigPoseOptimize(Cam cam, const bool optimize) {
  cam.set_rot_optimize(optimize);
  cam.set_t_optimize(optimize);
}

// Optimized parameters of a rig camera handle: its intrinsics and its
// rig pose, the rotation as three.
template <typename Cam>
int NumRigCamParams(Cam cam) {
  return NumIntrinsicParams(cam) + kRotationDof * cam.rot_optimize() +
         3 * cam.t_optimize();
}

// The solver's observer for BundleAdjustmentOptions::check_if_stopped,
// which `user` points at: false stops the solve.
bool ContinueUnlessStopped(void* user, const colmap_arael::LmIter*) noexcept {
  return !(*static_cast<const std::function<bool()>*>(user))();
}

Model::LmConfig CreateLmConfig(const AraelBundleAdjustmentOptions& options) {
  using LmConfig = Model::LmConfig;
  LmConfig config;
  switch (options.preset) {
    case AraelBundleAdjustmentPreset::DEFAULTS:
      config = LmConfig::defaults();
      break;
    case AraelBundleAdjustmentPreset::CONSERVATIVE:
      config = LmConfig::conservative();
      break;
    case AraelBundleAdjustmentPreset::WELL_CONDITIONED:
      config = LmConfig::well_conditioned();
      break;
    case AraelBundleAdjustmentPreset::ILL_CONDITIONED:
      config = LmConfig::ill_conditioned();
      break;
  }
  if (options.driver == AraelDriver::NIELSEN) {
    // The preset field picks the driver on arael's side; the other
    // fields keep the chosen preset's values.
    config.preset = arael::LmPreset::IllConditioned;
  }
  config.max_iters = options.max_num_iterations;
  // arael's 0 is every core.
  config.num_threads = options.num_threads > 0 ? options.num_threads : 0;
  if (options.initial_lambda > 0) {
    config.initial_lambda = options.initial_lambda;
  }
  if (options.rel_precision > 0) {
    config.rel_precision = options.rel_precision;
  }
  if (options.min_diagonal > 0) {
    config.min_diagonal = options.min_diagonal;
  }
  // arael's per-iteration trace (cost, damping, accepted) on stderr,
  // and its per-phase timing for the summary line.
  config.verbose = VLOG_IS_ON(3);
  config.gather_timing = VLOG_IS_ON(3);
  return config;
}

Model::SparseOptions CreateSparseOptions(
    const AraelBundleAdjustmentOptions& options) {
  Model::SparseOptions sparse;
  switch (options.schur_policy) {
    case AraelSchurPolicy::AUTO:
      sparse.schur = arael::SchurPolicy::Auto;
      break;
    case AraelSchurPolicy::FORCE:
      sparse.schur = arael::SchurPolicy::Force;
      break;
    case AraelSchurPolicy::NEVER:
      sparse.schur = arael::SchurPolicy::Never;
      break;
  }
  switch (options.schur_solve) {
    case AraelSchurSolve::FACTORIZE:
      sparse.schur_solve = arael::SchurSolve::Factorize;
      break;
    case AraelSchurSolve::ITERATIVE:
      sparse.schur_solve = arael::SchurSolve::Iterative;
      break;
    case AraelSchurSolve::ITERATIVE_IMPLICIT:
      sparse.schur_solve = arael::SchurSolve::IterativeImplicit;
      break;
  }
  sparse.cg_max_iters = options.max_cg_iterations;
  switch (options.ordering) {
    case AraelOrdering::AUTO:
      sparse.ordering = arael::FaerOrdering::Auto;
      break;
    case AraelOrdering::AMD:
      sparse.ordering = arael::FaerOrdering::Amd;
      break;
    case AraelOrdering::MARGINALIZE_FIRST:
      sparse.ordering = arael::FaerOrdering::MarginalizeFirst;
      break;
    case AraelOrdering::NATURAL:
      sparse.ordering = arael::FaerOrdering::Natural;
      break;
    case AraelOrdering::NESTED_DISSECTION:
      sparse.ordering = arael::FaerOrdering::NestedDissection;
      break;
  }
  const auto mode = [](const AraelRouteMode m) {
    switch (m) {
      case AraelRouteMode::ALWAYS:
        return arael::BlockSupernodalMode::Always;
      case AraelRouteMode::NEVER:
        return arael::BlockSupernodalMode::Never;
      default:
        return arael::BlockSupernodalMode::Auto;
    }
  };
  sparse.block_supernodal = mode(options.block_supernodal);
  switch (options.envelope) {
    case AraelRouteMode::AUTO:
      sparse.envelope = arael::EnvelopeMode::Auto;
      break;
    case AraelRouteMode::ALWAYS:
      sparse.envelope = arael::EnvelopeMode::Always;
      break;
    case AraelRouteMode::NEVER:
      sparse.envelope = arael::EnvelopeMode::Never;
      break;
  }
  return sparse;
}

// The route in one line: the reduction taken and how the reduced
// system was solved.
std::string DescribePlan(const arael::SchurPlan& plan,
                         const AraelBundleAdjustmentOptions& options) {
  std::ostringstream text;
  if (!plan.reduced) {
    text << "no reduction, sparse Cholesky of " << plan.kept_params
         << " parameters";
    return text.str();
  }
  text << "Schur: " << plan.eliminated_blocks << " points ("
       << plan.eliminated_params << " parameters) eliminated, "
       << plan.kept_params << " kept";
  if (plan.cg_iterations) {
    text << ", "
         << (options.schur_solve == AraelSchurSolve::ITERATIVE_IMPLICIT
                 ? "implicit "
                 : "")
         << "conjugate gradients, " << *plan.cg_iterations << " iterations";
  } else if (plan.envelope) {
    text << ", envelope Cholesky, half-bandwidth " << plan.kept_bandwidth;
  } else if (plan.block_supernodal) {
    text << ", block supernodal Cholesky";
  } else {
    text << ", supernodal Cholesky";
  }
  if (plan.fill_ratio) {
    text << ", fill " << *plan.fill_ratio;
  }
  return text.str();
}

// The arael model of the Ceres backend's problem: one pose per frame
// with images in the config, one fixed image (its cam_from_world as
// data) per image outside the config that observes a point of the
// problem (the constant-pose residuals of the Ceres backend), one
// camera entity per camera with its intrinsics and, for a camera
// that is not its rig's reference, its pose within the rig, one point
// per observed 3D point. Constant parameters are frozen by their
// optimize flags.
class AraelBundleAdjuster : public BundleAdjuster {
 public:
  using M = Model;
  using Scalar = M::Scalar;

  AraelBundleAdjuster(const BundleAdjustmentOptions& options,
                      const BundleAdjustmentConfig& config,
                      Reconstruction& reconstruction)
      : BundleAdjuster(options, config), reconstruction_(reconstruction) {
    VLOG(2) << "Creating arael bundle adjuster";

    THROW_CHECK(reconstruction.IsValid());

    // The loss as the Ceres backend takes it: the kinds are numbered
    // alike, the weight multiplies the loss.
    const CeresBundleAdjustmentOptions& loss =
        *THROW_CHECK_NOTNULL(options_.ceres);
    THROW_CHECK(IsValidCeresLossFunction(loss.loss_function_type,
                                         loss.loss_function_scale,
                                         loss.loss_function_weight));
    problem_.set_loss(static_cast<uint32_t>(loss.loss_function_type));
    problem_.set_k2(loss.loss_function_scale * loss.loss_function_scale);
    problem_.set_w(loss.loss_function_weight);

    // Warning: AddPointObservationsOutsideConfig assumes that AddImage was
    // called for every image in the config. Do not change the order!
    for (const image_t image_id : config_.Images()) {
      AddImage(image_id);
    }
    for (const point3D_t point3D_id : config_.VariablePoints()) {
      AddPointObservationsOutsideConfig(point3D_id);
    }
    for (const point3D_t point3D_id : config_.ConstantPoints()) {
      AddPointObservationsOutsideConfig(point3D_id);
    }

    ParameterizeCameras();
    ParameterizeRigsAndFrames();
    ParameterizePoints();

    switch (config_.FixedGauge()) {
      case BundleAdjustmentGauge::UNSPECIFIED:
        break;
      case BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD:
        FixGaugeWithTwoCamsFromWorld();
        break;
      case BundleAdjustmentGauge::THREE_POINTS:
        FixGaugeWithThreePoints();
        break;
      default:
        LOG(FATAL_THROW) << "Unknown BundleAdjustmentGauge";
    }
  }

  std::shared_ptr<BundleAdjustmentSummary> Solve() override {
    if (num_observations_ == 0) {
      return std::make_shared<BundleAdjustmentSummary>();
    }

    auto summary = std::make_shared<AraelBundleAdjustmentSummary>();
    summary->loss = DescribeLossFunction(options_);
    summary->site = options_.report_site;
    CountResidualsAndParameters(&summary->num_residuals,
                                &summary->num_effective_parameters);

    const AraelBundleAdjustmentOptions& arael_options = *options_.arael;
    M::LmConfig lm_config = CreateLmConfig(arael_options);
    // The per-phase timing goes into the report and the record; gather it
    // whenever either is written.
    if (options_.print_summary || VLOG_IS_ON(1) ||
        !options_.report_path.empty()) {
      lm_config.gather_timing = true;
    }
    // Asked after every damped attempt, as Ceres asks after every
    // iteration.
    if (options_.check_if_stopped) {
      lm_config.observer = &ContinueUnlessStopped;
      lm_config.observer_user = &options_.check_if_stopped;
    }

    M::SparseOptions sparse_options = CreateSparseOptions(arael_options);
    if (sparse_options.schur == arael::SchurPolicy::Force &&
        !HasVariablePoint()) {
      // A forced reduction with nothing to eliminate is an error in
      // arael; a problem with every point constant factors as a whole.
      sparse_options.schur = arael::SchurPolicy::Never;
    }

    // The record and the report, on every exit; a failed solve is a
    // record line too.
    const auto finish = [&]() {
      if (!options_.report_path.empty()) {
        AppendBundleAdjustmentRecord(options_.report_path,
                                     summary->JsonRecord());
      }
      if (options_.print_summary || VLOG_IS_ON(1)) {
        LOG(INFO) << summary->BriefReport();
      }
      return summary;
    };

    M::SolveResult result = M::SolveResult::err({});
    try {
      const auto solve_start = std::chrono::steady_clock::now();
      result = problem_.solve_sparse(lm_config, sparse_options);
      summary->wall_time = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - solve_start)
                               .count();
    } catch (const arael::PanicError& error) {
      LOG(ERROR) << "arael bundle adjustment panicked: " << error.what();
      summary->termination_type = BundleAdjustmentTerminationType::FAILURE;
      summary->termination_reason = "failed: panic";
      return finish();
    }

    if (result.is_err()) {
      const M::SolveError& error = result.error();
      LOG(ERROR) << "arael bundle adjustment failed: "
                 << arael::as_str(error.status) << ": "
                 << (error.message == nullptr ? "" : error.message);
      if (error.partial) {
        FillSummary(*error.partial, summary.get());
      }
      summary->termination_type = BundleAdjustmentTerminationType::FAILURE;
      summary->termination_reason =
          std::string("failed: ") + arael::as_str(error.status) +
          (error.message == nullptr ? "" : std::string(": ") + error.message);
      return finish();
    }

    const M::LmResult& lm_result = result.value();
    if (const arael::option<arael::SchurPlan> plan = lm_result.plan()) {
      summary->linear_solver = DescribePlan(*plan, arael_options);
    }
    summary->num_threads = lm_config.num_threads;
    if (lm_result.has_timing) {
      const arael::LmTiming& t = lm_result.timing;
      summary->has_timing = true;
      summary->time_total = t.total;
      summary->time_assembly = t.assembly;
      summary->time_analysis = t.analysis;
      summary->time_linear_solve = t.linear_solve;
      summary->time_cost = t.cost_eval;
      summary->time_advance = t.advance;
      summary->num_assemblies = t.assembly_count;
      summary->num_linear_solves = t.linear_solve_count;
      VLOG(3) << "arael timing [s]: total " << t.total << ", assembly "
              << t.assembly << " (" << t.assembly_count << ", first "
              << t.first_assembly << "), analysis " << t.analysis
              << ", linear solve " << t.linear_solve << " ("
              << t.linear_solve_count << ", first " << t.first_linear_solve
              << "), cost " << t.cost_eval << " (" << t.cost_eval_count
              << "), advance " << t.advance;
    }
    FillSummary(lm_result, summary.get());
    // A solve stopped through check_if_stopped ends as a Ceres one does.
    if (lm_result.status == arael::LmStatus::ObserverTerminated) {
      summary->termination_type = BundleAdjustmentTerminationType::USER_SUCCESS;
    } else {
      summary->termination_type =
          arael::is_success(lm_result.status)
              ? BundleAdjustmentTerminationType::CONVERGENCE
              : BundleAdjustmentTerminationType::NO_CONVERGENCE;
    }
    summary->termination_reason = arael::as_str(lm_result.status);

    WriteBack();
    return finish();
  }

 private:
  struct CameraSlot {
    // A camera with a pose within its rig, held in `rig_cams`; else the
    // rig's reference camera, held in `cams`.
    bool rig = false;
    // Index in its collection.
    uint32_t index = 0;
    // A rig camera: the entity's rig pose is the sensor_from_rig of
    // that rig, written back after the solve.
    rig_t rig_id = kInvalidRigId;
    sensor_t sensor_id;
    // Whether that rig pose is a parameter of the problem.
    bool variable_rig_pose = false;

    bool IsRig() const { return rig; }
  };

  struct FixedImage {
    bool rig = false;
    uint32_t index = 0;
  };

  const CameraSlot& AddCamera(const Image& image) {
    const Camera& camera = *image.CameraPtr();
    const auto it = cameras_.find(camera.camera_id);
    if (it != cameras_.end()) {
      return it->second;
    }

    CameraSlot slot;
    slot.rig = !image.IsRefInFrame();
    slot.rig_id = image.FramePtr()->RigId();
    slot.sensor_id = camera.SensorId();

    if (slot.rig) {
      slot.index = problem_.rig_cams().size();
      auto cam = problem_.rig_cams().push();
      SetIntrinsics(cam, camera);
      SetRigPose<Scalar>(
          cam, image.FramePtr()->RigPtr()->SensorFromRig(slot.sensor_id));
      SetRigPoseOptimize(cam, false);
    } else {
      slot.index = problem_.cams().size();
      SetIntrinsics(problem_.cams().push(), camera);
    }
    return cameras_.emplace(camera.camera_id, slot).first->second;
  }

  uint32_t PointIndex(const point3D_t point3D_id) {
    const auto it = point_indices_.find(point3D_id);
    if (it != point_indices_.end()) {
      return it->second;
    }
    const uint32_t index = static_cast<uint32_t>(point_ids_.size());
    M::Point point = problem_.points().push();
    point.set_pos(arael::vect3<Scalar>::from_eigen(
        reconstruction_.Point3D(point3D_id).xyz));
    point_ids_.push_back(point3D_id);
    point_indices_.emplace(point3D_id, index);
    return index;
  }

  // Appends a pose holding the frame's rig_from_world, all parameters
  // variable until parameterized.
  uint32_t AddPose(const Rigid3d& rig_from_world) {
    const uint32_t index = problem_.poses().size();
    M::Pose pose = problem_.poses().push();
    pose.set_rot(arael::quatern<Scalar>::from_eigen(
        Eigen::Quaterniond(rig_from_world.rotation()).normalized()));
    pose.set_tx(rig_from_world.translation().x());
    pose.set_ty(rig_from_world.translation().y());
    pose.set_tz(rig_from_world.translation().z());
    pose.set_rot_optimize(true);
    pose.set_tx_optimize(true);
    pose.set_ty_optimize(true);
    pose.set_tz_optimize(true);
    return index;
  }

  template <typename Obs>
  void AddObservation(Obs obs, const Point2D& point2D) {
    obs.set_point(problem_.points().ref_at(PointIndex(point2D.point3D_id)));
    obs.set_px(arael::vect2<Scalar>::from_eigen(point2D.xy));
  }

  // Appends an image of the given camera to the pose with the
  // observations.
  template <typename Image, typename CamRef>
  void FillImage(Image img,
                 const CamRef cam_ref,
                 const std::vector<const Point2D*>& observations) {
    img.set_cam(cam_ref);
    img.obs().reserve(observations.size());
    for (const Point2D* point2D : observations) {
      AddObservation(img.obs().push(), *point2D);
    }
  }

  void AddImage(const image_t image_id) {
    const Image& image = reconstruction_.Image(image_id);
    CheckSupportedCameraModel(*image.CameraPtr());

    // The pose and image entities exist only for images with observations.
    std::vector<const Point2D*> observations;
    for (const Point2D& point2D : image.Points2D()) {
      if (!point2D.HasPoint3D() || config_.IsIgnoredPoint(point2D.point3D_id)) {
        continue;
      }
      const Point3D& point3D = reconstruction_.Point3D(point2D.point3D_id);
      THROW_CHECK_GT(point3D.track.Length(), 1);
      if (options_.min_track_length > 0 &&
          static_cast<int>(point3D.track.Length()) <
              options_.min_track_length) {
        continue;
      }
      observations.push_back(&point2D);
    }
    if (observations.empty()) {
      return;
    }

    const CameraSlot& camera = AddCamera(image);

    auto pose_it = frame_poses_.find(image.FrameId());
    if (pose_it == frame_poses_.end()) {
      pose_it = frame_poses_
                    .emplace(image.FrameId(),
                             AddPose(image.FramePtr()->RigFromWorld()))
                    .first;
    }
    M::Pose pose = problem_.poses()[pose_it->second];

    for (const Point2D* point2D : observations) {
      point3D_num_observations_[point2D->point3D_id] += 1;
    }
    num_observations_ += observations.size();
    parameterized_camera_ids_.insert(image.CameraId());
    parameterized_image_ids_.insert(image_id);

    if (camera.rig) {
      FillImage(pose.rig_images().push(),
                problem_.rig_cams().ref_at(camera.index),
                observations);
    } else {
      FillImage(pose.images().push(),
                problem_.cams().ref_at(camera.index),
                observations);
    }
  }

  // The observations of a point in images outside the config, through a
  // fixed image holding that image's cam_from_world as data (the Ceres
  // backend's constant-pose residuals). Its camera's intrinsics stay
  // variable if a config image made them so; a camera seen only here is
  // constant.
  void AddPointObservationsOutsideConfig(const point3D_t point3D_id) {
    THROW_CHECK(!config_.IsIgnoredPoint(point3D_id));
    const Point3D& point3D = reconstruction_.Point3D(point3D_id);

    if (options_.min_track_length > 0 &&
        static_cast<int>(point3D.track.Length()) < options_.min_track_length) {
      return;
    }

    size_t& num_observations = point3D_num_observations_[point3D_id];
    if (num_observations == point3D.track.Length()) {
      return;
    }

    for (const TrackElement& track_el : point3D.track.Elements()) {
      if (config_.HasImage(track_el.image_id)) {
        continue;
      }

      num_observations += 1;
      num_observations_ += 1;

      const Image& image = reconstruction_.Image(track_el.image_id);
      const FixedImage& fixed = AddFixedImage(image);
      const Point2D& point2D = image.Point2D(track_el.point2D_idx);
      if (fixed.rig) {
        AddObservation(problem_.rig_fixed_images()[fixed.index].obs().push(),
                       point2D);
      } else {
        AddObservation(problem_.fixed_images()[fixed.index].obs().push(),
                       point2D);
      }

      if (parameterized_camera_ids_.insert(image.CameraId()).second) {
        config_.SetConstantCamIntrinsics(image.CameraId());
      }
    }
  }

  template <typename Image, typename CamRef>
  void FillFixedImage(Image img,
                      const CamRef cam_ref,
                      const arael::matrix3<Scalar>& r,
                      const arael::vect3<Scalar>& t) {
    img.set_cam(cam_ref);
    img.set_r(r);
    img.set_t(t);
  }

  const FixedImage& AddFixedImage(const Image& image) {
    const auto it = fixed_images_.find(image.ImageId());
    if (it != fixed_images_.end()) {
      return it->second;
    }
    const CameraSlot& camera = AddCamera(image);
    const Rigid3d cam_from_world = image.CamFromWorld();
    const arael::matrix3<Scalar> r = arael::matrix3<Scalar>::from_eigen(
        cam_from_world.rotation().normalized().toRotationMatrix());
    const arael::vect3<Scalar> t = arael::vect3<Scalar>::from_eigen(
        Eigen::Vector3d(cam_from_world.translation()));
    FixedImage fixed;
    fixed.rig = camera.rig;
    if (camera.rig) {
      fixed.index = problem_.rig_fixed_images().size();
      FillFixedImage(problem_.rig_fixed_images().push(),
                     problem_.rig_cams().ref_at(camera.index),
                     r,
                     t);
    } else {
      fixed.index = problem_.fixed_images().size();
      FillFixedImage(problem_.fixed_images().push(),
                     problem_.cams().ref_at(camera.index),
                     r,
                     t);
    }
    return fixed_images_.emplace(image.ImageId(), fixed).first->second;
  }

  void ParameterizeCameras() {
    const bool constant_camera = !options_.refine_focal_length &&
                                 !options_.refine_principal_point &&
                                 !options_.refine_extra_params;
    for (const camera_t camera_id : parameterized_camera_ids_) {
      const bool constant =
          constant_camera || config_.HasConstantCamIntrinsics(camera_id);
      const bool focal = !constant && options_.refine_focal_length;
      const bool pp = !constant && options_.refine_principal_point;
      const bool extra = !constant && options_.refine_extra_params;
      const CameraSlot& slot = cameras_.at(camera_id);
      const Camera& camera = reconstruction_.Camera(camera_id);
      if (slot.rig) {
        SetIntrinsicsOptimize(
            problem_.rig_cams()[slot.index], camera, focal, pp, extra);
      } else {
        SetIntrinsicsOptimize(
            problem_.cams()[slot.index], camera, focal, pp, extra);
      }
    }
  }

  void SetRigPoseVariable(CameraSlot& slot, const bool variable) {
    THROW_CHECK(slot.IsRig());
    slot.variable_rig_pose = variable;
    SetRigPoseOptimize(problem_.rig_cams()[slot.index], variable);
  }

  void ParameterizeRigsAndFrames() {
    FlatHashSet<rig_t> parameterized_rig_ids;
    FlatHashSet<sensor_t> parameterized_sensor_ids;
    FlatHashSet<frame_t> parameterized_frame_ids;
    for (const image_t image_id : parameterized_image_ids_) {
      const Image& image = reconstruction_.Image(image_id);
      parameterized_rig_ids.insert(image.FramePtr()->RigId());

      const sensor_t sensor_id = image.CameraPtr()->SensorId();
      if (parameterized_sensor_ids.insert(sensor_id).second &&
          !image.IsRefInFrame()) {
        SetRigPoseVariable(
            cameras_.at(image.CameraId()),
            options_.refine_sensor_from_rig &&
                !config_.HasConstantSensorFromRigPose(sensor_id));
      }

      if (parameterized_frame_ids.insert(image.FrameId()).second) {
        const bool variable =
            options_.refine_rig_from_world &&
            !config_.HasConstantRigFromWorldPose(image.FrameId());
        M::Pose pose = problem_.poses()[frame_poses_.at(image.FrameId())];
        pose.set_rot_optimize(variable &&
                              !options_.constant_rig_from_world_rotation);
        pose.set_tx_optimize(variable);
        pose.set_ty_optimize(variable);
        pose.set_tz_optimize(variable);
      }
    }

    // Without the reference sensor in the problem, the other sensors'
    // poses within the rig are not constrained; keep them constant.
    for (const rig_t rig_id : parameterized_rig_ids) {
      const Rig& rig = reconstruction_.Rig(rig_id);
      if (parameterized_sensor_ids.count(rig.RefSensorId()) != 0) {
        continue;
      }
      for (const auto& [sensor_id, sensor_from_rig] : rig.NonRefSensors()) {
        if (sensor_id.type != SensorType::CAMERA) {
          continue;
        }
        const auto it = cameras_.find(sensor_id.id);
        if (it != cameras_.end()) {
          SetRigPoseVariable(it->second, false);
        }
      }
    }
  }

  void ParameterizePoints() {
    for (const auto& [point3D_id, num_observations] :
         point3D_num_observations_) {
      const Point3D& point3D = reconstruction_.Point3D(point3D_id);
      const bool variable = options_.refine_points3D &&
                            point3D.track.Length() == num_observations;
      problem_.points()[point_indices_.at(point3D_id)].set_pos_optimize(
          variable);
    }
    for (const point3D_t point3D_id : config_.ConstantPoints()) {
      const auto it = point_indices_.find(point3D_id);
      if (it != point_indices_.end()) {
        problem_.points()[it->second].set_pos_optimize(false);
      }
    }
  }

  bool HasVariablePoint() {
    for (auto point : problem_.points()) {
      if (point.pos_optimize()) {
        return true;
      }
    }
    return false;
  }

  bool IsVariablePose(const frame_t frame_id) const {
    return options_.refine_rig_from_world &&
           !config_.HasConstantRigFromWorldPose(frame_id);
  }

  // A sensor whose pose within the rig is not a parameter: the
  // reference sensor, or one held constant.
  bool IsParameterizedConstSensor(const Image& image) const {
    return image.IsRefInFrame() ||
           !cameras_.at(image.CameraId()).variable_rig_pose;
  }

  void FreezePose(const frame_t frame_id) {
    M::Pose pose = problem_.poses()[frame_poses_.at(frame_id)];
    pose.set_rot_optimize(false);
    pose.set_tx_optimize(false);
    pose.set_ty_optimize(false);
    pose.set_tz_optimize(false);
  }

  void FreezePoseTranslationCoord(const frame_t frame_id, const int coord) {
    M::Pose pose = problem_.poses()[frame_poses_.at(frame_id)];
    switch (coord) {
      case 0:
        pose.set_tx_optimize(false);
        break;
      case 1:
        pose.set_ty_optimize(false);
        break;
      default:
        pose.set_tz_optimize(false);
        break;
    }
  }

  void FixGaugeWithTwoCamsFromWorld() {
    if (!options_.refine_rig_from_world) {
      return;
    }

    const Image* image1 = nullptr;
    const Image* image2 = nullptr;

    // First, search through the already fixed cameras in the problem.
    for (const image_t image_id : parameterized_image_ids_) {
      const Image& image = reconstruction_.Image(image_id);
      if (config_.HasConstantRigFromWorldPose(image.FrameId()) &&
          IsParameterizedConstSensor(image)) {
        if (image1 == nullptr) {
          image1 = &image;
        } else if (image1->FrameId() != image.FrameId()) {
          // Two frames are already fixed.
          return;
        }
      }
    }

    // Otherwise, search through the variable cameras in the problem.
    int frame2_from_world_fixed_dim = 0;
    for (const image_t image_id : parameterized_image_ids_) {
      const Image& image = reconstruction_.Image(image_id);
      if (image1 == nullptr && IsParameterizedConstSensor(image)) {
        image1 = &image;
      } else if (image1 != nullptr && image1->FrameId() != image.FrameId() &&
                 IsParameterizedConstSensor(image) &&
                 IsVariablePose(image.FrameId())) {
        // Fix the largest baseline coordinate; a negligible baseline
        // does not constrain the scale.
        const Eigen::Vector3d baseline =
            (image1->FramePtr()->RigFromWorld() *
             Inverse(image.FramePtr()->RigFromWorld()))
                .translation();
        Eigen::Index max_coeff_idx = 0;
        if (baseline.cwiseAbs().maxCoeff(&max_coeff_idx) > 1e-9) {
          image2 = &image;
          frame2_from_world_fixed_dim = static_cast<int>(max_coeff_idx);
          break;
        }
      }
    }

    if (image1 == nullptr || image2 == nullptr) {
      LOG(WARNING) << "Failed to fix Gauge with two cameras. "
                      "Falling back to fixing Gauge with three points.";
      FixGaugeWithThreePoints();
      return;
    }

    if (!config_.HasConstantRigFromWorldPose(image1->FrameId())) {
      FreezePose(image1->FrameId());
    }
    if (!config_.HasConstantRigFromWorldPose(image2->FrameId())) {
      FreezePoseTranslationCoord(image2->FrameId(),
                                 frame2_from_world_fixed_dim);
    }
  }

  void FixGaugeWithThreePoints() {
    FixedGaugeWithThreePoints fixed_gauge;

    // First check if we already fixed enough points in the problem.
    for (size_t i = 0; i < point_ids_.size(); ++i) {
      if (!problem_.points()[i].pos_optimize() &&
          fixed_gauge.MaybeAddFixedPoint(
              reconstruction_.Point3D(point_ids_[i]).xyz) &&
          fixed_gauge.num_fixed_points >= 3) {
        return;
      }
    }

    // Otherwise, fix sufficient points in the problem.
    for (size_t i = 0; i < point_ids_.size(); ++i) {
      M::Point point = problem_.points()[i];
      if (point.pos_optimize() &&
          fixed_gauge.MaybeAddFixedPoint(
              reconstruction_.Point3D(point_ids_[i]).xyz)) {
        point.set_pos_optimize(false);
        if (fixed_gauge.num_fixed_points >= 3) {
          return;
        }
      }
    }

    LOG(WARNING)
        << "Failed to fix Gauge due to insufficient number of fixed points: "
        << fixed_gauge.num_fixed_points;
  }

  // Residuals of the images with at least one variable parameter.
  template <typename ImagesVec, typename CamsVec, typename NumParams>
  void CountImageResiduals(ImagesVec images,
                           CamsVec cams,
                           NumParams num_params,
                           const bool pose_variable,
                           int* num_residuals) {
    for (auto img : images) {
      const bool cam_variable = num_params(cams.get(img.cam())) > 0;
      for (auto obs : img.obs()) {
        if (pose_variable || cam_variable ||
            problem_.points().get(obs.point()).pos_optimize()) {
          *num_residuals += 2;
        }
      }
    }
  }

  // Residuals with at least one variable parameter, and the variable
  // parameters, in the counting of the Ceres backend's reduced problem.
  void CountResidualsAndParameters(int* num_residuals, int* num_parameters) {
    *num_residuals = 0;
    *num_parameters = 0;

    const auto num_cam_params = [](auto cam) {
      return NumIntrinsicParams(cam);
    };
    const auto num_rig_cam_params = [](auto cam) {
      return NumRigCamParams(cam);
    };
    for (auto cam : problem_.cams()) {
      *num_parameters += num_cam_params(cam);
    }
    for (auto cam : problem_.rig_cams()) {
      *num_parameters += num_rig_cam_params(cam);
    }
    for (auto point : problem_.points()) {
      *num_parameters += 3 * point.pos_optimize();
    }

    for (auto pose : problem_.poses()) {
      const int num = kRotationDof * pose.rot_optimize() + pose.tx_optimize() +
                      pose.ty_optimize() + pose.tz_optimize();
      *num_parameters += num;
      const bool pose_variable = num > 0;
      CountImageResiduals(pose.images(),
                          problem_.cams(),
                          num_cam_params,
                          pose_variable,
                          num_residuals);
      CountImageResiduals(pose.rig_images(),
                          problem_.rig_cams(),
                          num_rig_cam_params,
                          pose_variable,
                          num_residuals);
    }

    CountImageResiduals(problem_.fixed_images(),
                        problem_.cams(),
                        num_cam_params,
                        /*pose_variable=*/false,
                        num_residuals);
    CountImageResiduals(problem_.rig_fixed_images(),
                        problem_.rig_cams(),
                        num_rig_cam_params,
                        /*pose_variable=*/false,
                        num_residuals);
  }

  static void FillSummary(const M::LmResult& result,
                          AraelBundleAdjustmentSummary* summary) {
    summary->num_iterations = static_cast<int>(result.iterations);
    summary->num_accepted_iterations =
        static_cast<int>(result.accepted_iterations);
    // arael sums rho(|r|^2) without Ceres' 1/2.
    summary->initial_cost = 0.5 * result.start_cost;
    summary->final_cost = 0.5 * result.end_cost;
  }

  // Only what the solve could have moved is written back: a constant
  // parameter keeps the reconstruction's value.
  void WriteBack() {
    for (const auto& [frame_id, index] : frame_poses_) {
      M::Pose pose = problem_.poses()[index];
      Rigid3d& rig_from_world = reconstruction_.Frame(frame_id).RigFromWorld();
      if (pose.rot_optimize()) {
        rig_from_world.rotation() =
            pose.rot().to_eigen().template cast<double>().normalized();
      }
      if (pose.tx_optimize()) rig_from_world.translation().x() = pose.tx();
      if (pose.ty_optimize()) rig_from_world.translation().y() = pose.ty();
      if (pose.tz_optimize()) rig_from_world.translation().z() = pose.tz();
    }

    for (const auto& [camera_id, slot] : cameras_) {
      Camera& camera = reconstruction_.Camera(camera_id);
      if (slot.rig) {
        auto cam = problem_.rig_cams()[slot.index];
        if (NumIntrinsicParams(cam) > 0) {
          GetIntrinsics(cam, camera);
        }
        if (cam.rot_optimize() || cam.t_optimize()) {
          GetRigPose(
              cam,
              reconstruction_.Rig(slot.rig_id).SensorFromRig(slot.sensor_id));
        }
      } else {
        auto cam = problem_.cams()[slot.index];
        if (NumIntrinsicParams(cam) > 0) {
          GetIntrinsics(cam, camera);
        }
      }
    }

    for (uint32_t i = 0; i < problem_.points().size(); ++i) {
      M::Point point = problem_.points()[i];
      if (point.pos_optimize()) {
        reconstruction_.Point3D(point_ids_[i]).xyz =
            point.pos().to_eigen().template cast<double>();
      }
    }
  }

  Reconstruction& reconstruction_;
  M::Problem problem_;

  std::unordered_map<camera_t, CameraSlot> cameras_;
  std::unordered_map<frame_t, uint32_t> frame_poses_;
  std::unordered_map<image_t, FixedImage> fixed_images_;
  std::vector<point3D_t> point_ids_;
  std::unordered_map<point3D_t, uint32_t> point_indices_;
  FlatHashMap<point3D_t, size_t> point3D_num_observations_;

  std::set<camera_t> parameterized_camera_ids_;
  std::set<image_t> parameterized_image_ids_;
  size_t num_observations_ = 0;
};

}  // namespace

std::string AraelBundleAdjustmentSummary::BriefReport() const {
  // Costs as the Ceres report prints them: the root mean square
  // residual in pixels.
  const auto px = [this](const double cost) {
    return num_residuals > 0 ? std::sqrt(cost / num_residuals) : 0.0;
  };
  std::ostringstream report;
  report << "Bundle adjustment report (arael)\n"
         << "    Residuals : " << num_residuals << "\n"
         << "   Parameters : " << num_effective_parameters << "\n"
         << "   Iterations : " << num_iterations << " ("
         << num_accepted_iterations << " accepted)\n"
         << "         Loss : " << loss << "\n"
         << " Initial cost : " << px(initial_cost) << " [px]\n"
         << "   Final cost : " << px(final_cost) << " [px]\n";
  if (!linear_solver.empty()) {
    report << "       Solver : " << linear_solver << "\n";
  }
  if (has_timing) {
    report << "         Time : " << time_total << " [s] (assembly "
           << time_assembly << ", analysis " << time_analysis
           << ", linear solve " << time_linear_solve << ", cost " << time_cost
           << ")\n";
  }
  report << "  Termination : " << termination_type;
  if (!termination_reason.empty()) {
    report << " (" << termination_reason << ")";
  }
  report << "\n";
  return report.str();
}

std::string AraelBundleAdjustmentSummary::JsonRecord() const {
  std::ostringstream line;
  line << std::setprecision(17);
  line << "{\"backend\": \"arael\", \"kind\": \"bundle adjustment\""
       << ", \"site\": " << JsonQuoted(site)
       << ", \"residuals\": " << num_residuals
       << ", \"parameters\": " << num_effective_parameters
       << ", \"iterations\": " << num_iterations
       << ", \"accepted\": " << num_accepted_iterations
       << ", \"initial_cost\": " << JsonNumber(initial_cost)
       << ", \"final_cost\": " << JsonNumber(final_cost)
       << ", \"termination\": " << JsonQuoted(termination_reason)
       << ", \"loss\": " << JsonQuoted(loss)
       << ", \"solver\": " << JsonQuoted(linear_solver)
       << ", \"threads\": " << num_threads
       << ", \"time\": " << JsonNumber(wall_time);
  if (has_timing) {
    line << ", \"timing\": {\"total\": " << JsonNumber(time_total)
         << ", \"assembly\": " << JsonNumber(time_assembly)
         << ", \"assemblies\": " << num_assemblies
         << ", \"analysis\": " << JsonNumber(time_analysis)
         << ", \"linear_solve\": " << JsonNumber(time_linear_solve)
         << ", \"linear_solves\": " << num_linear_solves
         << ", \"cost\": " << JsonNumber(time_cost)
         << ", \"advance\": " << JsonNumber(time_advance) << "}";
  }
  line << "}";
  return line.str();
}

std::unique_ptr<BundleAdjuster> CreateDefaultAraelBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    Reconstruction& reconstruction) {
  THROW_CHECK(options.arael != nullptr);
  THROW_CHECK(options.arael->Check());
  return std::make_unique<AraelBundleAdjuster>(options, config, reconstruction);
}

}  // namespace colmap
