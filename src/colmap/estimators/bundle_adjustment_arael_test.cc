// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/bundle_adjustment_arael.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/math/random.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/models.h"
#include "colmap/util/testing.h"

#include <fstream>
#include <optional>

#include <gtest/gtest.h>

namespace colmap {
namespace {

// Due to pose normalization operations, constant variables may not be perfectly
// fixed during bundle adjustment.
constexpr double kConstantPoseVarEps = 1e-9;

const AraelBundleAdjustmentSummary& GetAraelSummary(
    const BundleAdjustmentSummary* summary) {
  auto* arael_summary =
      dynamic_cast<const AraelBundleAdjustmentSummary*>(summary);
  CHECK_NOTNULL(arael_summary);
  return *arael_summary;
}

std::shared_ptr<BundleAdjustmentSummary> SolveArael(
    BundleAdjustmentOptions options,
    const BundleAdjustmentConfig& config,
    Reconstruction& reconstruction) {
  options.backend = BundleAdjustmentBackend::ARAEL;
  options.print_summary = false;
  return CreateDefaultAraelBundleAdjuster(options, config, reconstruction)
      ->Solve();
}

void ExpectVariableCamera(const Camera& camera, const Camera& orig_camera) {
  const size_t focal_length_idx = SimpleRadialCameraModel::focal_length_idxs[0];
  const size_t extra_param_idx = SimpleRadialCameraModel::extra_params_idxs[0];
  EXPECT_NE(camera.params[focal_length_idx],
            orig_camera.params[focal_length_idx]);
  EXPECT_NE(camera.params[extra_param_idx],
            orig_camera.params[extra_param_idx]);
}

void ExpectConstantCamera(const Camera& camera, const Camera& orig_camera) {
  const size_t focal_length_idx = SimpleRadialCameraModel::focal_length_idxs[0];
  const size_t extra_param_idx = SimpleRadialCameraModel::extra_params_idxs[0];
  EXPECT_EQ(camera.params[focal_length_idx],
            orig_camera.params[focal_length_idx]);
  EXPECT_EQ(camera.params[extra_param_idx],
            orig_camera.params[extra_param_idx]);
}

void ExpectVariableCamFromWorld(const Image& image, const Image& orig_image) {
  EXPECT_THAT(image.CamFromWorld(),
              testing::Not(Rigid3dEq(orig_image.CamFromWorld())));
}

void ExpectConstantCamFromWorld(const Image& image, const Image& orig_image) {
  EXPECT_THAT(
      image.CamFromWorld(),
      Rigid3dNear(
          orig_image.CamFromWorld(), kConstantPoseVarEps, kConstantPoseVarEps));
}

void ExpectConstantCamFromWorldTranslationCoord(const Image& image,
                                                const Image& orig_image) {
  size_t num_constant_coords = 0;
  for (int i = 0; i < 3; ++i) {
    if (std::abs(image.CamFromWorld().translation()(i) -
                 orig_image.CamFromWorld().translation()(i)) <
        kConstantPoseVarEps) {
      ++num_constant_coords;
    }
  }
  EXPECT_EQ(num_constant_coords, 1);
}

void ExpectVariablePoint(const Point3D& point, const Point3D& orig_point) {
  EXPECT_NE(point.xyz, orig_point.xyz);
}

void ExpectConstantPoint(const Point3D& point, const Point3D& orig_point) {
  EXPECT_EQ(point.xyz, orig_point.xyz);
}

size_t NumVariablePoints(const Reconstruction& reconstruction,
                         const Reconstruction& orig_reconstruction) {
  size_t num_variable_points = 0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D != orig_reconstruction.Point3D(point3D_id)) {
      ++num_variable_points;
    }
  }
  return num_variable_points;
}

TEST(AraelBundleAdjuster, NominalMultiCameraRig) {
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 200;
  SynthesizeDataset(synthetic_dataset_options, &gt_reconstruction);

  Reconstruction reconstruction = gt_reconstruction;

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.point3D_stddev = 0.1;
  synthetic_noise_options.rig_from_world_rotation_stddev = 0.5;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  const auto summary =
      SolveArael(BundleAdjustmentOptions(), config, reconstruction);
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
}

TEST(AraelBundleAdjuster, TwoViewRig) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

  const auto summary =
      SolveArael(BundleAdjustmentOptions(), config, reconstruction);
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // 100 points, 4 images, 2 residuals per point per image
  EXPECT_EQ(config.NumResiduals(reconstruction), 800);
  EXPECT_EQ(summary->num_residuals, 800);
  // 97 x 3 point parameters (3 fixed for gauge)
  // + 2 x 6 rig_from_world parameters
  // + 1 x 6 sensor_from_rig parameters
  // + 2 x 2 camera parameters
  EXPECT_EQ(GetAraelSummary(summary.get()).num_effective_parameters, 313);

  ExpectVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  ExpectVariableCamFromWorld(reconstruction.Image(1),
                             orig_reconstruction.Image(1));

  ExpectVariableCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
  ExpectVariableCamFromWorld(reconstruction.Image(2),
                             orig_reconstruction.Image(2));

  EXPECT_EQ(NumVariablePoints(reconstruction, orig_reconstruction), 97);
}

TEST(AraelBundleAdjuster, ManyViewRig) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

  const auto summary =
      SolveArael(BundleAdjustmentOptions(), config, reconstruction);
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // 100 points, 30 images, 2 residuals per point per image
  EXPECT_EQ(config.NumResiduals(reconstruction), 6000);
  EXPECT_EQ(summary->num_residuals, 6000);
  // 97 x 3 point parameters (3 fixed for gauge)
  // + 10 x 6 rig_from_world parameters
  // + 4 x 6 sensor_from_rig parameters
  // + 6 x 2 camera parameters
  EXPECT_EQ(GetAraelSummary(summary.get()).num_effective_parameters, 387);

  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    ExpectVariableCamera(camera, orig_reconstruction.Camera(camera_id));
  }

  for (const image_t image_id : reconstruction.RegImageIds()) {
    ExpectVariableCamFromWorld(reconstruction.Image(image_id),
                               orig_reconstruction.Image(image_id));
  }

  EXPECT_EQ(NumVariablePoints(reconstruction, orig_reconstruction), 97);
}

TEST(AraelBundleAdjuster, ManyViewRigConstantSensorFromRig) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.SetConstantSensorFromRigPose(reconstruction.Camera(2).SensorId());
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

  const auto summary =
      SolveArael(BundleAdjustmentOptions(), config, reconstruction);
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_EQ(summary->num_residuals, 6000);
  // 97 x 3 point parameters (3 fixed for gauge)
  // + 10 x 6 rig_from_world parameters
  // + 3 x 6 sensor_from_rig parameters
  // + 6 x 2 camera parameters
  EXPECT_EQ(GetAraelSummary(summary.get()).num_effective_parameters, 381);

  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    ExpectVariableCamera(camera, orig_reconstruction.Camera(camera_id));
  }

  for (const image_t image_id : reconstruction.RegImageIds()) {
    ExpectVariableCamFromWorld(reconstruction.Image(image_id),
                               orig_reconstruction.Image(image_id));
  }

  EXPECT_EQ(NumVariablePoints(reconstruction, orig_reconstruction), 97);
}

TEST(AraelBundleAdjuster, ManyViewRigConstantRigFromWorld) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  const frame_t constant_frame_id = 1;
  config.SetConstantRigFromWorldPose(constant_frame_id);
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

  const auto summary =
      SolveArael(BundleAdjustmentOptions(), config, reconstruction);
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_EQ(summary->num_residuals, 6000);
  // 97 x 3 point parameters (3 fixed for gauge)
  // + 9 x 6 rig_from_world parameters
  // + 4 x 6 sensor_from_rig parameters
  // + 6 x 2 camera parameters
  EXPECT_EQ(GetAraelSummary(summary.get()).num_effective_parameters, 381);

  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    ExpectVariableCamera(camera, orig_reconstruction.Camera(camera_id));
  }

  for (const image_t image_id : reconstruction.RegImageIds()) {
    const auto& image = reconstruction.Image(image_id);
    if (image.FrameId() == constant_frame_id &&
        image.FramePtr()->RigPtr()->IsRefSensor(
            image.CameraPtr()->SensorId())) {
      ExpectConstantCamFromWorld(image, orig_reconstruction.Image(image_id));
    } else {
      ExpectVariableCamFromWorld(image, orig_reconstruction.Image(image_id));
    }
  }

  EXPECT_EQ(NumVariablePoints(reconstruction, orig_reconstruction), 97);
}

TEST(AraelBundleAdjuster, ConstantRigFromWorldRotation) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 3;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.AddImage(3);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.constant_rig_from_world_rotation = true;
  const auto summary = SolveArael(options, config, reconstruction);
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // 100 points, 3 images, 2 residuals per point per image
  EXPECT_EQ(summary->num_residuals, 600);
  // 100 x 3 point parameters
  // + 2 translation parameters (second image, one coord fixed for gauge)
  // + 3 translation parameters (third image)
  // + 3 x 2 camera parameters
  EXPECT_EQ(GetAraelSummary(summary.get()).num_effective_parameters, 311);

  for (const image_t image_id : reconstruction.RegImageIds()) {
    const auto& image = reconstruction.Image(image_id);
    const auto& orig_image = orig_reconstruction.Image(image_id);
    EXPECT_LE(image.CamFromWorld().rotation().angularDistance(
                  orig_image.CamFromWorld().rotation()),
              kConstantPoseVarEps);
  }

  bool has_variable_translation = false;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    const auto& image = reconstruction.Image(image_id);
    const auto& orig_image = orig_reconstruction.Image(image_id);
    if ((image.CamFromWorld().translation() -
         orig_image.CamFromWorld().translation())
            .norm() > kConstantPoseVarEps) {
      has_variable_translation = true;
      break;
    }
  }
  EXPECT_TRUE(has_variable_translation);

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    ExpectVariablePoint(point3D, orig_reconstruction.Point3D(point3D_id));
  }
}

TEST(AraelBundleAdjuster, PartiallyContainedTracksForceToOptimizePoint) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 3;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.num_points2D_without_point3D = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  const point3D_t variable_point3D_id =
      reconstruction.Image(3).Point2D(0).point3D_id;
  const point3D_t add_variable_point3D_id =
      reconstruction.Image(3).Point2D(1).point3D_id;
  const point3D_t add_constant_point3D_id =
      reconstruction.Image(3).Point2D(2).point3D_id;
  reconstruction.DeleteObservation(3, 0);

  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);
  config.AddVariablePoint(add_variable_point3D_id);
  config.AddConstantPoint(add_constant_point3D_id);

  const auto summary =
      SolveArael(BundleAdjustmentOptions(), config, reconstruction);
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // 100 points, 2 images, 2 residuals per point per image
  // + 2 residuals in 3rd image for added variable 3D point
  // (added constant point does not add residuals since the image/camera
  // is also constant).
  EXPECT_EQ(summary->num_residuals, 402);
  // 2 x 3 point parameters
  // 2 x 2 camera parameters
  EXPECT_EQ(GetAraelSummary(summary.get()).num_effective_parameters, 10);

  ExpectVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  ExpectConstantCamFromWorld(reconstruction.Image(1),
                             orig_reconstruction.Image(1));

  ExpectVariableCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
  ExpectConstantCamFromWorld(reconstruction.Image(2),
                             orig_reconstruction.Image(2));

  ExpectConstantCamera(reconstruction.Camera(3), orig_reconstruction.Camera(3));
  ExpectConstantCamFromWorld(reconstruction.Image(3),
                             orig_reconstruction.Image(3));

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D_id == variable_point3D_id ||
        point3D_id == add_variable_point3D_id) {
      ExpectVariablePoint(point3D, orig_reconstruction.Point3D(point3D_id));
    } else {
      ExpectConstantPoint(point3D, orig_reconstruction.Point3D(point3D_id));
    }
  }
}

TEST(AraelBundleAdjuster, ConstantFocalLength) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.refine_focal_length = false;
  const auto summary = SolveArael(options, config, reconstruction);
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(summary->num_residuals, 400);
  // 100 x 3 point parameters
  // + 5 rig_from_world parameters (pose of second image)
  // + 2 camera parameters
  EXPECT_EQ(GetAraelSummary(summary.get()).num_effective_parameters, 307);

  ExpectConstantCamFromWorld(reconstruction.Image(1),
                             orig_reconstruction.Image(1));
  ExpectConstantCamFromWorldTranslationCoord(reconstruction.Image(2),
                                             orig_reconstruction.Image(2));

  const size_t focal_length_idx = SimpleRadialCameraModel::focal_length_idxs[0];
  const size_t extra_param_idx = SimpleRadialCameraModel::extra_params_idxs[0];
  for (const camera_t camera_id : {1, 2}) {
    const auto& camera = reconstruction.Camera(camera_id);
    const auto& orig_camera = orig_reconstruction.Camera(camera_id);
    EXPECT_EQ(camera.params[focal_length_idx],
              orig_camera.params[focal_length_idx]);
    EXPECT_NE(camera.params[extra_param_idx],
              orig_camera.params[extra_param_idx]);
  }

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    ExpectVariablePoint(point3D, orig_reconstruction.Point3D(point3D_id));
  }
}

TEST(AraelBundleAdjuster, ConstantExtraParam) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.refine_extra_params = false;
  const auto summary = SolveArael(options, config, reconstruction);
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_EQ(summary->num_residuals, 400);
  // 100 x 3 point parameters
  // + 5 rig_from_world parameters (pose of second image)
  // + 2 camera parameters
  EXPECT_EQ(GetAraelSummary(summary.get()).num_effective_parameters, 307);

  ExpectConstantCamFromWorld(reconstruction.Image(1),
                             orig_reconstruction.Image(1));
  ExpectConstantCamFromWorldTranslationCoord(reconstruction.Image(2),
                                             orig_reconstruction.Image(2));

  const size_t focal_length_idx = SimpleRadialCameraModel::focal_length_idxs[0];
  const size_t extra_param_idx = SimpleRadialCameraModel::extra_params_idxs[0];
  for (const camera_t camera_id : {1, 2}) {
    const auto& camera = reconstruction.Camera(camera_id);
    const auto& orig_camera = orig_reconstruction.Camera(camera_id);
    EXPECT_NE(camera.params[focal_length_idx],
              orig_camera.params[focal_length_idx]);
    EXPECT_EQ(camera.params[extra_param_idx],
              orig_camera.params[extra_param_idx]);
  }

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    ExpectVariablePoint(point3D, orig_reconstruction.Point3D(point3D_id));
  }
}

TEST(AraelBundleAdjuster, FixGaugeWithThreePoints) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);

  auto ExpectValidSolve =
      [&config, &reconstruction](const int num_effective_parameters) {
        const auto summary =
            SolveArael(BundleAdjustmentOptions(), config, reconstruction);
        ASSERT_NE(summary->termination_type,
                  BundleAdjustmentTerminationType::FAILURE);
        EXPECT_EQ(GetAraelSummary(summary.get()).num_effective_parameters,
                  num_effective_parameters);
      };

  ExpectValidSolve(316);

  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);
  ExpectValidSolve(307);

  config.AddConstantPoint(1);
  ExpectValidSolve(307);

  config.AddConstantPoint(2);
  config.AddConstantPoint(3);
  ExpectValidSolve(307);

  config.AddConstantPoint(4);
  ExpectValidSolve(304);
}

TEST(AraelBundleAdjuster, AllPointsConstant) {
  // The default route forces the reduction; with every point constant
  // there is nothing to eliminate and the solve must still run.
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);

  BundleAdjustmentOptions options;
  ASSERT_EQ(options.arael->schur_policy, AraelSchurPolicy::FORCE);
  options.refine_points3D = false;
  const auto summary = SolveArael(options, config, reconstruction);
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);
  EXPECT_EQ(GetAraelSummary(summary.get()).num_effective_parameters, 16);
}

TEST(AraelBundleAdjuster, FixGaugeWithTwoCamsFromWorld) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  BundleAdjustmentOptions options;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.AddImage(3);
  config.AddImage(4);

  auto ExpectValidSolve =
      [&options, &config, &reconstruction](const int num_effective_parameters) {
        const auto summary = SolveArael(options, config, reconstruction);
        ASSERT_NE(summary->termination_type,
                  BundleAdjustmentTerminationType::FAILURE);
        EXPECT_EQ(GetAraelSummary(summary.get()).num_effective_parameters,
                  num_effective_parameters);
      };

  options.refine_rig_from_world = false;
  ExpectValidSolve(320);

  options.refine_rig_from_world = true;
  ExpectValidSolve(332);

  options.refine_rig_from_world = false;
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  ExpectValidSolve(320);

  options.refine_rig_from_world = true;
  ExpectValidSolve(325);

  config.SetConstantRigFromWorldPose(1);
  ExpectValidSolve(325);

  config.SetConstantRigFromWorldPose(2);
  ExpectValidSolve(320);
}

TEST(AraelBundleAdjuster, FixGaugeWithTwoCamsFromWorldFixSensorFromRig) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  BundleAdjustmentOptions options;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.AddImage(3);
  config.AddImage(4);

  auto ExpectValidSolve =
      [&options, &config, &reconstruction](const int num_effective_parameters) {
        const auto summary = SolveArael(options, config, reconstruction);
        ASSERT_NE(summary->termination_type,
                  BundleAdjustmentTerminationType::FAILURE);
        EXPECT_EQ(GetAraelSummary(summary.get()).num_effective_parameters,
                  num_effective_parameters);
      };

  options.refine_rig_from_world = false;
  options.refine_sensor_from_rig = false;
  ExpectValidSolve(308);

  options.refine_rig_from_world = true;
  ExpectValidSolve(320);

  options.refine_rig_from_world = false;
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  ExpectValidSolve(308);

  options.refine_rig_from_world = true;
  ExpectValidSolve(313);

  config.SetConstantRigFromWorldPose(1);
  ExpectValidSolve(313);

  config.SetConstantRigFromWorldPose(2);
  ExpectValidSolve(308);
}

TEST(AraelBundleAdjuster, FixGaugeWithTwoCamsFromWorldNoReferenceSensor) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  // Delete observations from the two reference images.
  THROW_CHECK(reconstruction.Image(1).IsRefInFrame());
  THROW_CHECK(reconstruction.Image(3).IsRefInFrame());
  for (point2D_t i = 0; i < reconstruction.Image(1).NumPoints2D(); ++i) {
    if (reconstruction.Image(1).Point2D(i).HasPoint3D()) {
      reconstruction.DeleteObservation(1, i);
    }
  }
  for (point2D_t i = 0; i < reconstruction.Image(3).NumPoints2D(); ++i) {
    if (reconstruction.Image(3).Point2D(i).HasPoint3D()) {
      reconstruction.DeleteObservation(3, i);
    }
  }

  // Only add two non-reference images.
  BundleAdjustmentOptions options;
  BundleAdjustmentConfig config;
  config.AddImage(2);
  config.AddImage(4);

  auto ExpectValidSolve =
      [&options, &config, &reconstruction](const int num_effective_parameters) {
        const auto summary = SolveArael(options, config, reconstruction);
        THROW_CHECK_NE(summary->termination_type,
                       BundleAdjustmentTerminationType::FAILURE);
        THROW_CHECK_EQ(GetAraelSummary(summary.get()).num_effective_parameters,
                       num_effective_parameters);
      };

  // refine_sensor_from_rig has no effect without reference sensors.
  options.refine_rig_from_world = true;
  options.refine_sensor_from_rig = true;
  ExpectValidSolve(316);

  options.refine_rig_from_world = false;
  ExpectValidSolve(304);

  options.refine_rig_from_world = true;
  ExpectValidSolve(316);

  options.refine_rig_from_world = false;
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  ExpectValidSolve(304);

  options.refine_sensor_from_rig = false;
  ExpectValidSolve(304);

  options.refine_rig_from_world = true;
  ExpectValidSolve(309);

  config.SetConstantRigFromWorldPose(1);
  ExpectValidSolve(309);
  options.refine_rig_from_world = false;
  ExpectValidSolve(304);

  config.SetConstantRigFromWorldPose(2);
  options.refine_rig_from_world = true;
  ExpectValidSolve(304);
  options.refine_rig_from_world = false;
  ExpectValidSolve(304);
}

TEST(AraelBundleAdjuster, FixGaugeWithTwoCamsFromWorldFallback) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);

  // Two reference cameras in different frames are needed to fix the gauge
  // with two cameras; without them the gauge is fixed with three points.
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  const auto summary =
      SolveArael(BundleAdjustmentOptions(), config, reconstruction);
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);
  EXPECT_EQ(GetAraelSummary(summary.get()).num_effective_parameters, 307);
}

// A constant parameter keeps the reconstruction's value exactly: the
// backend writes back only what the solve could have moved.
TEST(AraelBundleAdjuster, ConstantParametersKeepTheirValues) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 3;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.num_points2D_without_point3D = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  reconstruction.DeleteObservation(3, 0);

  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);
  config.SetConstantCamIntrinsics(2);

  const auto summary =
      SolveArael(BundleAdjustmentOptions(), config, reconstruction);
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  ExpectVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  ExpectConstantCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
  ExpectConstantCamera(reconstruction.Camera(3), orig_reconstruction.Camera(3));
  for (const image_t image_id : {1, 2, 3}) {
    EXPECT_THAT(reconstruction.Image(image_id).CamFromWorld(),
                Rigid3dEq(orig_reconstruction.Image(image_id).CamFromWorld()));
  }
}

// Every COLMAP camera model builds an adjuster; a model added upstream
// later fails here until the backend has it.
TEST(AraelBundleAdjuster, SupportsEveryCameraModel) {
  int num_models = 0;
  for (int id = 0; ExistsCameraModelWithId(static_cast<CameraModelId>(id));
       ++id) {
    const CameraModelId model_id = static_cast<CameraModelId>(id);
    SCOPED_TRACE(CameraModelIdToName(model_id));
    ++num_models;
    Reconstruction reconstruction;
    SyntheticDatasetOptions synthetic_dataset_options;
    synthetic_dataset_options.num_rigs = 3;
    synthetic_dataset_options.num_cameras_per_rig = 1;
    synthetic_dataset_options.num_frames_per_rig = 1;
    synthetic_dataset_options.num_points3D = 100;
    synthetic_dataset_options.camera_model_id = model_id;
    synthetic_dataset_options.camera_params =
        CameraModelInitializeParams(model_id,
                                    /*focal_length=*/1000,
                                    synthetic_dataset_options.camera_width,
                                    synthetic_dataset_options.camera_height);
    SynthesizeDataset(synthetic_dataset_options, &reconstruction);

    BundleAdjustmentConfig config;
    config.AddImage(1);
    config.AddImage(2);
    config.AddImage(3);

    EXPECT_NO_THROW(CreateDefaultAraelBundleAdjuster(
        BundleAdjustmentOptions(), config, reconstruction));
  }
  EXPECT_EQ(num_models, 19);
}

TEST(AraelBundleAdjuster, SolveRecord) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.report_path = (CreateTestDir() / "ba_report.jsonl").string();
  // Two solves append two records; each gathers the timing.
  for (int i = 0; i < 2; ++i) {
    const auto summary = SolveArael(options, config, reconstruction);
    ASSERT_NE(summary->termination_type,
              BundleAdjustmentTerminationType::FAILURE);
    const AraelBundleAdjustmentSummary& arael_summary =
        GetAraelSummary(summary.get());
    EXPECT_TRUE(arael_summary.has_timing);
    EXPECT_GT(arael_summary.wall_time, 0.0);
    EXPECT_NE(arael_summary.BriefReport().find("Time :"), std::string::npos);
  }

  std::ifstream file(options.report_path);
  std::vector<std::string> lines;
  for (std::string line; std::getline(file, line);) {
    lines.push_back(line);
  }
  ASSERT_EQ(lines.size(), 2);
  EXPECT_NE(lines[0].find("\"backend\": \"arael\""), std::string::npos);
  EXPECT_NE(lines[0].find("\"iterations\": "), std::string::npos);
  EXPECT_NE(lines[0].find("\"timing\": {"), std::string::npos);
  EXPECT_NE(lines[0].find("\"termination\": \"converged\""), std::string::npos);
}

// A non-finite value is written as null; JSON has no infinity or NaN.
TEST(AraelBundleAdjustmentSummary, JsonRecordWritesNonFiniteAsNull) {
  AraelBundleAdjustmentSummary summary;
  summary.initial_cost = std::numeric_limits<double>::infinity();
  summary.final_cost = 1.5;
  summary.wall_time = std::numeric_limits<double>::quiet_NaN();
  const std::string line = summary.JsonRecord();
  EXPECT_NE(line.find("\"initial_cost\": null"), std::string::npos);
  EXPECT_NE(line.find("\"final_cost\": 1.5"), std::string::npos);
  EXPECT_NE(line.find("\"time\": null"), std::string::npos);
  EXPECT_EQ(line.find("inf"), std::string::npos);
  EXPECT_EQ(line.find("nan"), std::string::npos);
}

// The gain-ratio driver is the default on the preset's settings, and
// the preset's own schedule stays selectable; both converge.
TEST(AraelBundleAdjuster, Drivers) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  EXPECT_EQ(BundleAdjustmentOptions().arael->driver, AraelDriver::NIELSEN);
  for (const AraelDriver driver : {AraelDriver::NIELSEN, AraelDriver::PRESET}) {
    std::ostringstream name;
    name << driver;
    SCOPED_TRACE(name.str());
    Reconstruction solved = reconstruction;
    BundleAdjustmentOptions options;
    options.arael->driver = driver;
    const auto summary = SolveArael(options, config, solved);
    EXPECT_EQ(summary->termination_type,
              BundleAdjustmentTerminationType::CONVERGENCE);
  }
}

TEST(AraelBundleAdjuster, BriefReport) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.print_summary = true;
  const auto summary =
      CreateDefaultAraelBundleAdjuster(options, config, reconstruction)
          ->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);
  const std::string report = summary->BriefReport();
  EXPECT_NE(report.find("arael"), std::string::npos);
  EXPECT_NE(report.find("Parameters"), std::string::npos);
  EXPECT_NE(report.find("CONVERGENCE"), std::string::npos);
  EXPECT_NE(report.find("Loss : TRIVIAL"), std::string::npos);

  options.ceres->loss_function_type = CeresLossFunctionType::CAUCHY;
  options.ceres->loss_function_scale = 2.0;
  const std::string robust_report =
      CreateDefaultAraelBundleAdjuster(options, config, reconstruction)
          ->Solve()
          ->BriefReport();
  EXPECT_NE(robust_report.find("Loss : CAUCHY (scale 2 px)"),
            std::string::npos);
}

// The Ceres backend's cost of the reconstruction under the options.
double CeresCost(const BundleAdjustmentOptions& options,
                 const BundleAdjustmentConfig& config,
                 const Reconstruction& reconstruction) {
  Reconstruction copy = reconstruction;
  auto bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, copy);
  double cost = 0.0;
  bundle_adjuster->Problem()->Evaluate(
      ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
  return cost;
}

struct OptimumCase {
  std::string name;
  int num_rigs = 1;
  int num_cameras_per_rig = 1;
  int num_frames_per_rig = 1;
  CameraModelId camera_model_id = SimpleRadialCameraModel::model_id;
  std::vector<double> camera_params;
  BundleAdjustmentGauge gauge = BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD;
  CeresLossFunctionType loss = CeresLossFunctionType::TRIVIAL;
  // Scales the pose and point noise. A robust loss makes the cost
  // non-convex, so those cases start closer to the optimum for both
  // backends to reach the same one.
  double noise = 1.0;
  // Local bundle adjustment shape: only the first images are in the
  // config, with every point they observe variable, so the points'
  // other observations enter through images outside the config.
  int num_config_images = 0;
  // Whether the camera's extra parameters are variable.
  bool refine_extra_params = true;
};

// Both backends minimize the same cost from the same start; their optima
// must agree to solver tolerance, so arael is held to Ceres' tolerances.
TEST(AraelBundleAdjuster, MatchesCeresOptimum) {
  std::vector<OptimumCase> cases;
  cases.push_back({"simple_radial/4frames", 1, 1, 4});
  cases.push_back({"simple_radial/rig", 2, 3, 5});
  cases.push_back({"simple_radial/3points/soft_l1",
                   1,
                   1,
                   4,
                   SimpleRadialCameraModel::model_id,
                   {},
                   BundleAdjustmentGauge::THREE_POINTS,
                   CeresLossFunctionType::SOFT_L1,
                   0.1});
  cases.push_back({"pinhole/rig/cauchy",
                   1,
                   2,
                   4,
                   PinholeCameraModel::model_id,
                   {1000, 1000, 512, 384},
                   BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD,
                   CeresLossFunctionType::CAUCHY,
                   0.1});
  cases.push_back({"simple_radial/local",
                   1,
                   1,
                   6,
                   SimpleRadialCameraModel::model_id,
                   {},
                   BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD,
                   CeresLossFunctionType::TRIVIAL,
                   1.0,
                   2});
  cases.push_back({"simple_radial/rig/local",
                   1,
                   3,
                   4,
                   SimpleRadialCameraModel::model_id,
                   {},
                   BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD,
                   CeresLossFunctionType::TRIVIAL,
                   1.0,
                   4});

  // Every other camera model of the backend, with distortion; the
  // fisheye models at a shorter focal length for a wide field of view.
  cases.push_back({"simple_pinhole",
                   1,
                   1,
                   4,
                   SimplePinholeCameraModel::model_id,
                   {1000, 512, 384}});
  cases.push_back({"radial",
                   1,
                   1,
                   4,
                   RadialCameraModel::model_id,
                   {1000, 512, 384, -0.1, 0.02}});
  cases.push_back({"opencv",
                   1,
                   1,
                   4,
                   OpenCVCameraModel::model_id,
                   {1000, 1000, 512, 384, -0.1, 0.02, 0.001, -0.002}});
  cases.push_back({"opencv/rig",
                   1,
                   3,
                   4,
                   OpenCVCameraModel::model_id,
                   {1000, 1000, 512, 384, -0.1, 0.02, 0.001, -0.002}});
  cases.push_back({"opencv/rig/local",
                   1,
                   3,
                   4,
                   OpenCVCameraModel::model_id,
                   {1000, 1000, 512, 384, -0.1, 0.02, 0.001, -0.002},
                   BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD,
                   CeresLossFunctionType::TRIVIAL,
                   1.0,
                   4});
  // The rational model's numerator and denominator radial terms are
  // nearly interchangeable at small radius, a flat valley where each
  // backend stops at a point of its own; the distortion stays constant
  // and the projection is checked through the other parameters.
  cases.push_back({"full_opencv",
                   1,
                   1,
                   4,
                   FullOpenCVCameraModel::model_id,
                   {1000,
                    1000,
                    512,
                    384,
                    -0.1,
                    0.02,
                    0.001,
                    -0.002,
                    0.005,
                    -0.05,
                    0.01,
                    0.002},
                   BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD,
                   CeresLossFunctionType::TRIVIAL,
                   1.0,
                   0,
                   /*refine_extra_params=*/false});
  cases.push_back({"opencv_fisheye",
                   1,
                   1,
                   4,
                   OpenCVFisheyeCameraModel::model_id,
                   {600, 600, 512, 384, -0.02, 0.005, -0.001, 0.0005}});
  cases.push_back({"opencv_fisheye/rig",
                   1,
                   2,
                   4,
                   OpenCVFisheyeCameraModel::model_id,
                   {600, 600, 512, 384, -0.02, 0.005, -0.001, 0.0005}});
  cases.push_back({"thin_prism_fisheye",
                   1,
                   1,
                   4,
                   ThinPrismFisheyeCameraModel::model_id,
                   {600,
                    600,
                    512,
                    384,
                    -0.02,
                    0.005,
                    0.001,
                    -0.001,
                    -0.001,
                    0.0005,
                    0.0005,
                    -0.0005}});
  cases.push_back({"rad_tan_thin_prism_fisheye",
                   1,
                   1,
                   4,
                   RadTanThinPrismFisheyeModel::model_id,
                   {600,
                    600,
                    512,
                    384,
                    -0.02,
                    0.005,
                    -0.001,
                    0.0005,
                    0.0001,
                    -0.0001,
                    0.001,
                    -0.001,
                    0.0005,
                    -0.0005,
                    0.0003,
                    -0.0003}});
  cases.push_back({"simple_radial_fisheye",
                   1,
                   1,
                   4,
                   SimpleRadialFisheyeCameraModel::model_id,
                   {600, 512, 384, -0.02}});
  cases.push_back({"radial_fisheye",
                   1,
                   1,
                   4,
                   RadialFisheyeCameraModel::model_id,
                   {600, 512, 384, -0.02, 0.005}});
  cases.push_back({"radial_fisheye/rig",
                   1,
                   2,
                   4,
                   RadialFisheyeCameraModel::model_id,
                   {600, 512, 384, -0.02, 0.005}});
  cases.push_back(
      {"fov", 1, 1, 4, FOVCameraModel::model_id, {600, 610, 512, 384, 0.8}});
  // Below omega^2 = 1e-4 the model takes its series in omega; omega is
  // held so the solve stays there.
  cases.push_back({"fov/small_omega",
                   1,
                   1,
                   4,
                   FOVCameraModel::model_id,
                   {1000, 1010, 512, 384, 0.005},
                   BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD,
                   CeresLossFunctionType::TRIVIAL,
                   1.0,
                   0,
                   /*refine_extra_params=*/false});
  cases.push_back({"simple_fisheye",
                   1,
                   1,
                   4,
                   SimpleFisheyeCameraModel::model_id,
                   {600, 512, 384}});
  cases.push_back(
      {"fisheye", 1, 1, 4, FisheyeCameraModel::model_id, {600, 610, 512, 384}});
  cases.push_back({"skewed_pinhole",
                   1,
                   1,
                   4,
                   SkewedPinholeCameraModel::model_id,
                   {1000, 1010, 512, 384, 2.0}});
  cases.push_back({"simple_division",
                   1,
                   1,
                   4,
                   SimpleDivisionCameraModel::model_id,
                   {1000, 512, 384, -0.1}});
  cases.push_back({"division",
                   1,
                   1,
                   4,
                   DivisionCameraModel::model_id,
                   {1000, 1010, 512, 384, 0.05}});
  // EUCM's alpha and beta, with the focal length, leave a nearly flat
  // valley: Ceres is still descending it at 500 iterations. They stay
  // constant, as the rational model's terms do above.
  cases.push_back({"eucm",
                   1,
                   1,
                   4,
                   EUCMCameraModel::model_id,
                   {600, 610, 512, 384, 0.6, 1.1},
                   BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD,
                   CeresLossFunctionType::TRIVIAL,
                   1.0,
                   0,
                   /*refine_extra_params=*/false});
  // No intrinsics to refine: the image width and height are fixed.
  cases.push_back({"equirectangular",
                   1,
                   1,
                   4,
                   EquirectangularCameraModel::model_id,
                   {1024, 768}});

  for (const OptimumCase& c : cases) {
    SCOPED_TRACE(c.name);
    // Each case starts from the same seed, so its problem is the same on
    // every machine, whatever the earlier cases drew.
    SetPRNGSeed();

    Reconstruction reconstruction;
    SyntheticDatasetOptions synthetic_dataset_options;
    synthetic_dataset_options.num_rigs = c.num_rigs;
    synthetic_dataset_options.num_cameras_per_rig = c.num_cameras_per_rig;
    synthetic_dataset_options.num_frames_per_rig = c.num_frames_per_rig;
    synthetic_dataset_options.num_points3D = 200;
    synthetic_dataset_options.camera_model_id = c.camera_model_id;
    if (!c.camera_params.empty()) {
      synthetic_dataset_options.camera_width = 1024;
      synthetic_dataset_options.camera_height = 768;
      synthetic_dataset_options.camera_params = c.camera_params;
    }
    SynthesizeDataset(synthetic_dataset_options, &reconstruction);
    SyntheticNoiseOptions synthetic_noise_options;
    synthetic_noise_options.point2D_stddev = 1.0;
    synthetic_noise_options.point3D_stddev = 0.1 * c.noise;
    synthetic_noise_options.rig_from_world_rotation_stddev = 0.5 * c.noise;
    synthetic_noise_options.rig_from_world_translation_stddev = 0.1 * c.noise;
    SynthesizeNoise(synthetic_noise_options, &reconstruction);

    BundleAdjustmentConfig config;
    const std::vector<image_t> reg_image_ids = reconstruction.RegImageIds();
    const size_t num_config_images =
        c.num_config_images > 0 ? static_cast<size_t>(c.num_config_images)
                                : reg_image_ids.size();
    ASSERT_LE(num_config_images, reg_image_ids.size());
    for (size_t i = 0; i < num_config_images; ++i) {
      config.AddImage(reg_image_ids[i]);
    }
    if (c.num_config_images > 0) {
      for (const image_t image_id : config.Images()) {
        for (const Point2D& point2D :
             reconstruction.Image(image_id).Points2D()) {
          if (point2D.HasPoint3D() && !config.HasPoint(point2D.point3D_id)) {
            config.AddVariablePoint(point2D.point3D_id);
          }
        }
      }
    }
    config.FixGauge(c.gauge);
    if (c.gauge == BundleAdjustmentGauge::THREE_POINTS) {
      // Each backend picks its own three points to fix, and fixing three
      // noisy points over-constrains the problem; pin the same three.
      for (const point3D_t point3D_id : {1, 2, 3}) {
        config.AddConstantPoint(point3D_id);
      }
    }

    BundleAdjustmentOptions options;
    options.print_summary = false;
    options.ceres->loss_function_type = c.loss;
    options.ceres->loss_function_scale = 2.0;
    options.ceres->solver_options.max_num_iterations = 500;
    options.ceres->solver_options.function_tolerance = 1e-12;
    options.ceres->solver_options.gradient_tolerance = 1e-12;
    options.ceres->solver_options.parameter_tolerance = 1e-12;
    options.arael->max_num_iterations = 500;
    options.arael->rel_precision = 1e-12;
    options.refine_extra_params = c.refine_extra_params;

    Reconstruction ceres_reconstruction = reconstruction;
    const auto ceres_summary =
        CreateDefaultCeresBundleAdjuster(options, config, ceres_reconstruction)
            ->Solve();
    ASSERT_EQ(ceres_summary->termination_type,
              BundleAdjustmentTerminationType::CONVERGENCE);

    // Both backends report the cost of the problem they solved, with the
    // poses of images outside the config baked in at construction.
    const double ceres_final =
        dynamic_cast<const CeresBundleAdjustmentSummary&>(*ceres_summary)
            .ceres_summary.final_cost;

    // With every image in the config nothing is baked in, so a fresh
    // evaluation of each written-back solution is its final cost.
    if (c.num_config_images == 0) {
      EXPECT_NEAR(CeresCost(options, config, ceres_reconstruction),
                  ceres_final,
                  1e-9 * ceres_final);
    }

    Reconstruction arael_reconstruction = reconstruction;
    const auto arael_summary =
        SolveArael(options, config, arael_reconstruction);
    ASSERT_EQ(arael_summary->termination_type,
              BundleAdjustmentTerminationType::CONVERGENCE);
    const double arael_final = GetAraelSummary(arael_summary.get()).final_cost;
    EXPECT_NEAR(arael_final, ceres_final, 1e-5 * ceres_final);

    if (c.num_config_images == 0) {
      EXPECT_NEAR(CeresCost(options, config, arael_reconstruction),
                  arael_final,
                  1e-9 * arael_final);
    }

    EXPECT_THAT(ceres_reconstruction,
                ReconstructionNear(arael_reconstruction,
                                   /*max_rotation_error_deg=*/1e-2,
                                   /*max_proj_center_error=*/1e-2,
                                   /*max_scale_error=*/std::nullopt,
                                   /*num_obs_tolerance=*/0.0));
  }
}

// A point moved to `in_cam` in the camera frame of the first image of
// its track; with `observed_x` set, that image's observation of it moves
// to that column.
struct PlacedPoint {
  Eigen::Vector3d in_cam;
  std::optional<double> observed_x;
};

struct ResidualCase {
  std::string name;
  CameraModelId camera_model_id;
  std::vector<double> camera_params;
  std::vector<PlacedPoint> placed;
};

// arael's cost equals the Ceres backend's on the same noisy problem, for
// every camera model, with points where the model's projection fails
// (a zero residual in both) or behaves at its edges: behind the camera,
// a negative division discriminant, EUCM's checks, FOV's series near
// omega = 0 and near the optical axis, the equirectangular seam.
TEST(AraelBundleAdjuster, ResidualsMatchCeres) {
  const std::vector<PlacedPoint> behind = {{{0.1, 0.1, -1.0}}};
  const std::vector<ResidualCase> cases = {
      {"pinhole", PinholeCameraModel::model_id, {1000, 1010, 512, 384}, behind},
      {"simple_pinhole",
       SimplePinholeCameraModel::model_id,
       {1000, 512, 384},
       behind},
      {"simple_radial",
       SimpleRadialCameraModel::model_id,
       {1000, 512, 384, -0.1},
       behind},
      {"radial",
       RadialCameraModel::model_id,
       {1000, 512, 384, -0.1, 0.02},
       behind},
      {"opencv",
       OpenCVCameraModel::model_id,
       {1000, 1010, 512, 384, -0.1, 0.02, 0.001, -0.002},
       behind},
      {"full_opencv",
       FullOpenCVCameraModel::model_id,
       {1000,
        1010,
        512,
        384,
        -0.1,
        0.02,
        0.001,
        -0.002,
        0.005,
        -0.05,
        0.01,
        0.002},
       behind},
      {"opencv_fisheye",
       OpenCVFisheyeCameraModel::model_id,
       {600, 610, 512, 384, -0.02, 0.005, -0.001, 0.0005},
       behind},
      {"thin_prism_fisheye",
       ThinPrismFisheyeCameraModel::model_id,
       {600,
        610,
        512,
        384,
        -0.02,
        0.005,
        0.001,
        -0.001,
        -0.001,
        0.0005,
        0.0005,
        -0.0005},
       behind},
      {"rad_tan_thin_prism_fisheye",
       RadTanThinPrismFisheyeModel::model_id,
       {600,
        610,
        512,
        384,
        -0.02,
        0.005,
        -0.001,
        0.0005,
        0.0001,
        -0.0001,
        0.001,
        -0.001,
        0.0005,
        -0.0005,
        0.0003,
        -0.0003},
       behind},
      {"simple_radial_fisheye",
       SimpleRadialFisheyeCameraModel::model_id,
       {600, 512, 384, -0.02},
       behind},
      {"radial_fisheye",
       RadialFisheyeCameraModel::model_id,
       {600, 512, 384, -0.02, 0.005},
       behind},
      {"fov", FOVCameraModel::model_id, {600, 610, 512, 384, 0.8}, behind},
      // Within 1e-2 of the optical axis: the series in the radius.
      {"fov/near_axis",
       FOVCameraModel::model_id,
       {600, 610, 512, 384, 0.8},
       {{{1e-3, 1e-3, 2.0}}}},
      // omega^2 below 1e-4: the series in omega.
      {"fov/small_omega",
       FOVCameraModel::model_id,
       {1000, 1010, 512, 384, 0.005},
       behind},
      {"simple_fisheye",
       SimpleFisheyeCameraModel::model_id,
       {600, 512, 384},
       behind},
      {"fisheye", FisheyeCameraModel::model_id, {600, 610, 512, 384}, behind},
      {"skewed_pinhole",
       SkewedPinholeCameraModel::model_id,
       {1000, 1010, 512, 384, 2.0},
       behind},
      {"simple_division",
       SimpleDivisionCameraModel::model_id,
       {1000, 512, 384, 0.1},
       {}},
      // Far off the axis: w^2 - 4 k (u^2 + v^2) < 0.
      {"simple_division/negative_discriminant",
       SimpleDivisionCameraModel::model_id,
       {1000, 512, 384, 0.1},
       {{{2.0, 1.0, 1.0}}}},
      // The division models check no depth: a point behind the camera
      // projects, far outside the image.
      {"division/behind",
       DivisionCameraModel::model_id,
       {1000, 1010, 512, 384, -0.1},
       {{{0.3, 0.2, -2.0}}}},
      {"eucm",
       EUCMCameraModel::model_id,
       {600, 610, 512, 384, 0.6, 1.1},
       behind},
      // A negative beta, far off the axis: beta (u^2 + v^2) + w^2 < 0.
      {"eucm/negative_radicand",
       EUCMCameraModel::model_id,
       {600, 610, 512, 384, 0.6, -0.5},
       {{{2.0, 1.0, 1.0}}}},
      // Behind the camera, near the seam at longitude pi: the projection
      // lands at the right edge and the observation at the left, so the
      // residual wraps across the seam.
      {"equirectangular",
       EquirectangularCameraModel::model_id,
       {1024, 768},
       {{{0.5, 0.1, -2.0}}, {{1e-3, 0.1, -2.0}, 0.5}}},
  };

  for (const ResidualCase& c : cases) {
    SCOPED_TRACE(c.name);
    SetPRNGSeed();

    Reconstruction reconstruction;
    SyntheticDatasetOptions synthetic_dataset_options;
    synthetic_dataset_options.num_rigs = 1;
    synthetic_dataset_options.num_cameras_per_rig = 1;
    synthetic_dataset_options.num_frames_per_rig = 4;
    synthetic_dataset_options.num_points3D = 100;
    synthetic_dataset_options.camera_model_id = c.camera_model_id;
    synthetic_dataset_options.camera_width = 1024;
    synthetic_dataset_options.camera_height = 768;
    synthetic_dataset_options.camera_params = c.camera_params;
    SynthesizeDataset(synthetic_dataset_options, &reconstruction);
    SyntheticNoiseOptions synthetic_noise_options;
    synthetic_noise_options.point2D_stddev = 1.0;
    synthetic_noise_options.point3D_stddev = 0.1;
    synthetic_noise_options.rig_from_world_rotation_stddev = 0.5;
    synthetic_noise_options.rig_from_world_translation_stddev = 0.1;
    SynthesizeNoise(synthetic_noise_options, &reconstruction);

    std::vector<point3D_t> point3D_ids;
    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      point3D_ids.push_back(point3D_id);
    }
    std::sort(point3D_ids.begin(), point3D_ids.end());
    ASSERT_LE(c.placed.size(), point3D_ids.size());
    for (size_t i = 0; i < c.placed.size(); ++i) {
      Point3D& point3D = reconstruction.Point3D(point3D_ids[i]);
      const TrackElement& element = point3D.track.Elements().front();
      Image& image = reconstruction.Image(element.image_id);
      point3D.xyz = Inverse(image.CamFromWorld()) * c.placed[i].in_cam;
      if (c.placed[i].observed_x) {
        image.Point2D(element.point2D_idx).xy.x() = *c.placed[i].observed_x;
      }
    }

    BundleAdjustmentConfig config;
    for (const image_t image_id : reconstruction.RegImageIds()) {
      config.AddImage(image_id);
    }
    config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

    BundleAdjustmentOptions options;
    options.print_summary = false;
    options.arael->max_num_iterations = 1;
    const double ceres_cost = CeresCost(options, config, reconstruction);
    ASSERT_TRUE(std::isfinite(ceres_cost));

    Reconstruction arael_reconstruction = reconstruction;
    const auto summary = SolveArael(options, config, arael_reconstruction);
    EXPECT_NEAR(GetAraelSummary(summary.get()).initial_cost,
                ceres_cost,
                1e-9 * ceres_cost);
  }
}

// Smallest depth of any observation of a registered image.
double MinObservationDepth(const Reconstruction& reconstruction) {
  double min_depth = std::numeric_limits<double>::infinity();
  for (const image_t image_id : reconstruction.RegImageIds()) {
    const Image& image = reconstruction.Image(image_id);
    const Rigid3d cam_from_world = image.CamFromWorld();
    for (const Point2D& point2D : image.Points2D()) {
      if (point2D.HasPoint3D()) {
        const Eigen::Vector3d point_in_cam =
            cam_from_world * reconstruction.Point3D(point2D.point3D_id).xyz;
        min_depth = std::min(min_depth, point_in_cam.z());
      }
    }
  }
  return min_depth;
}

// Points moved behind one of their cameras: that view's observation is
// zeroed, as in the Ceres backend, the solve stays finite and
// converges, and the points' other views bring them back in front.
// The optimum of the clean scene is out of reach from this start for
// either backend, so the cost is not checked.
TEST(AraelBundleAdjuster, PointBehindCamera) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 200;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.point3D_stddev = 0.1;
  synthetic_noise_options.rig_from_world_rotation_stddev = 0.5;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  // Move five points behind the first camera of their track, at two
  // percent of their depth.
  std::vector<point3D_t> point3D_ids;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    point3D_ids.push_back(point3D_id);
  }
  std::sort(point3D_ids.begin(), point3D_ids.end());
  for (size_t i = 0; i < 5; ++i) {
    Point3D& point3D = reconstruction.Point3D(point3D_ids[i]);
    const image_t image_id = point3D.track.Elements().front().image_id;
    const Eigen::Vector3d proj_center =
        Inverse(reconstruction.Image(image_id).CamFromWorld()).translation();
    point3D.xyz = proj_center - 0.02 * (point3D.xyz - proj_center);
  }
  ASSERT_LT(MinObservationDepth(reconstruction), 0.0);

  const auto summary =
      SolveArael(BundleAdjustmentOptions(), config, reconstruction);
  ASSERT_EQ(summary->termination_type,
            BundleAdjustmentTerminationType::CONVERGENCE);
  EXPECT_TRUE(std::isfinite(GetAraelSummary(summary.get()).final_cost));
  EXPECT_GT(MinObservationDepth(reconstruction), 0.0);
}

// A point behind every camera that observes it has all its observations
// zeroed and so no curvature. Without the damping floor the solve ends
// in failure, which the record reports; with it the solve converges and
// the point stays where it was.
TEST(AraelBundleAdjuster, PointWithoutActiveObservations) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 200;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  // Keep two observations of one point and move it behind both cameras,
  // far out along the opposite of their viewing directions.
  std::vector<point3D_t> point3D_ids;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    point3D_ids.push_back(point3D_id);
  }
  std::sort(point3D_ids.begin(), point3D_ids.end());
  const point3D_t point3D_id = point3D_ids.front();
  const std::vector<TrackElement> elements =
      reconstruction.Point3D(point3D_id).track.Elements();
  ASSERT_GE(elements.size(), 3);
  for (size_t i = 2; i < elements.size(); ++i) {
    reconstruction.DeleteObservation(elements[i].image_id,
                                     elements[i].point2D_idx);
  }
  Eigen::Vector3d viewing_sum = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < 2; ++i) {
    const Rigid3d world_from_cam =
        Inverse(reconstruction.Image(elements[i].image_id).CamFromWorld());
    viewing_sum += world_from_cam.rotation() * Eigen::Vector3d::UnitZ();
  }
  const Eigen::Vector3d xyz_behind = -1e3 * viewing_sum;
  reconstruction.Point3D(point3D_id).xyz = xyz_behind;
  for (size_t i = 0; i < 2; ++i) {
    const Rigid3d cam_from_world =
        reconstruction.Image(elements[i].image_id).CamFromWorld();
    ASSERT_LT((cam_from_world * xyz_behind).z(), 0.0);
  }

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  {
    Reconstruction copy = reconstruction;
    BundleAdjustmentOptions options;
    options.arael->min_diagonal = 0;
    options.report_path = (CreateTestDir() / "ba_report.jsonl").string();
    const auto summary = SolveArael(options, config, copy);
    EXPECT_EQ(summary->termination_type,
              BundleAdjustmentTerminationType::FAILURE);
    std::ifstream file(options.report_path);
    std::string line;
    ASSERT_TRUE(std::getline(file, line));
    EXPECT_NE(line.find("\"termination\": \"failed: "), std::string::npos);
    EXPECT_NE(line.find("Hessian diagonal"), std::string::npos);
  }

  const auto summary =
      SolveArael(BundleAdjustmentOptions(), config, reconstruction);
  EXPECT_EQ(summary->termination_type,
            BundleAdjustmentTerminationType::CONVERGENCE);
  EXPECT_TRUE(std::isfinite(GetAraelSummary(summary.get()).final_cost));
  EXPECT_EQ(reconstruction.Point3D(point3D_id).xyz, xyz_behind);
}

}  // namespace
}  // namespace colmap
