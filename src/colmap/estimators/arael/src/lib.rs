//! Bundle adjustment model for the COLMAP arael backend.
//!
//! Structure follows COLMAP's own: a problem has frame poses, points
//! and cameras; a frame owns its images, one per camera it saw
//! through; an image owns its observations; a camera carries its
//! intrinsics and, for a non-reference camera of a rig, its pose
//! within the rig. The reference camera of a rig sits at the rig's
//! origin, so it carries no rig pose and its observations project the
//! frame pose straight through the intrinsics: a smaller point-camera
//! tile and a shorter Jacobian for every single-camera image, which is
//! the default COLMAP pipeline.
//!
//! An image whose pose is not part of the solve (outside the bundle
//! adjustment config, observing a point that is) is a fixed image:
//! it holds its world-to-camera transform as plain data, the frame
//! and rig poses composed, and its observations couple only the
//! point and the camera's intrinsics.
//!
//! The cost is the Ceres backend's 2-row reprojection residual under
//! the robust loss COLMAP's LossFunctionType names, applied per solve.
//! As in Ceres, an observation the camera model cannot project has its
//! residual and Jacobian zeroed: for most models a point below
//! machine-epsilon depth, so a point behind a camera feels no pull from
//! that view, and the pipeline's filters drop the observation.
//!
//! Position priors, for COLMAP's pose-prior bundle adjustment, add a
//! 3-row residual per image: the prior position minus the camera centre,
//! whitened by the prior's covariance, under their own loss
//! (`prior_loss`, `prior_k2`).
//!
//! Camera models: one camera entity for every COLMAP model. A camera
//! holds its model as data (`model`), two focal lengths, the principal
//! point and twelve extra parameters, the most any model has. The
//! projection matches on the model at runtime; arael compiles every
//! switch on it into one `match` per observation, so only the camera's
//! own formula runs. A model with one focal length reads `fx` for both
//! axes; `fy` and the extra parameters a model does not have stay out
//! of the solve (their `optimize` flags cleared). EQUIRECTANGULAR has
//! no focal length or principal point: its image width and height sit
//! in `e0` and `e1`, never optimized.
//!
//! Poses are stored in COLMAP's direction: a frame pose takes a world
//! point into rig coordinates, a camera's rig pose takes a rig point
//! into camera coordinates. The rotation retracts as Ceres'
//! EigenQuaternionManifold does; the translation is additive, three
//! scalar Params so COLMAP's TWO_CAMS_FROM_WORLD gauge can freeze one
//! axis.
//!
//! Hessian tiles: the (pose, cam) tile of an image is shared by all
//! its observations and lives on the image; the point tiles are one
//! per observation, they are the Schur W blocks.

use arael::matrix::matrix3;
use arael::model::{CrossBlock, Param, QuaternionParam, SelfBlock, VarCrossBlock, VarSelfBlock};
use arael::refs::{self, Ref};
use arael::sym::{matrix3sym, vect2sym, vect3sym, E};
use arael::utils::Float;
use arael::vect::{vect2, vect3};

/// COLMAP's LossFunctionType on the squared residual norm `s`:
/// 0 trivial, 1 soft L1, 2 Cauchy, 3 Huber. `k2` is the squared loss
/// scale. Any other `kind` panics.
#[arael::function]
fn colmap_loss(kind: E, s: E, k2: E) -> E {
    match kind {
        0 => s,
        1 => loss_soft_l1(s, k2),
        2 => loss_cauchy(s, k2),
        3 => loss_huber(s, k2),
    }
}

// ---------------------------------------------------------------------------
// Projection pieces
// ---------------------------------------------------------------------------

/// A world point `x` in a frame's rig coordinates: the frame rotation
/// `rot`, then its translation held as three scalars.
#[arael::function]
fn in_rig(rot: matrix3sym, tx: E, ty: E, tz: E, x: vect3sym) -> vect3sym {
    let r = rot * x;
    vect3sym::from_components(r.x + tx, r.y + ty, r.z + tz)
}

/// The normalized image point of a camera-frame point `p`, and its
/// depth test: positive when the point lies in front of the camera by
/// more than machine epsilon.
#[arael::function]
fn normalized(p: vect3sym) -> (vect2sym, E) {
    (vect2sym::from_components(p.x / p.z, p.y / p.z), p.z - epsilon_for(p.z))
}

/// The equidistant fisheye image of a normalized point: the direction
/// scaled by theta over r with theta = atan(r); the point itself on the
/// optical axis, where r is below machine epsilon.
#[arael::function]
fn fisheye_normal(n: vect2sym) -> vect2sym {
    let r = safe_sqrt(n.x * n.x + n.y * n.y);
    let scale = branch(r - epsilon_for(r), atan(r) / r, 1.0);
    n * scale
}

/// The displacement of a normalized point under two radial and two
/// tangential Brown-Conrady coefficients, COLMAP's OPENCV distortion.
#[arael::function]
fn brown(n: vect2sym, k1: E, k2: E, p1: E, p2: E) -> vect2sym {
    let (u, v) = (n.x, n.y);
    let u2 = u * u;
    let v2 = v * v;
    let uv = u * v;
    let r2 = u2 + v2;
    let radial = k1 * r2 + k2 * r2 * r2;
    vect2sym::from_components(u * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2),
                              v * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2))
}

// ---------------------------------------------------------------------------
// Distortions, one per camera model with extra parameters: the
// normalized point `n` and the model's extra parameters in COLMAP's
// order, to the distorted normalized point.
// ---------------------------------------------------------------------------

/// SIMPLE_RADIAL: `k`, one radial term.
#[arael::function]
fn simple_radial(n: vect2sym, k: E) -> vect2sym {
    n * (1.0 + k * (n.x * n.x + n.y * n.y))
}

/// RADIAL: `k1, k2`, two radial terms.
#[arael::function]
fn radial(n: vect2sym, k1: E, k2: E) -> vect2sym {
    let r2 = n.x * n.x + n.y * n.y;
    n * (1.0 + r2 * (k1 + r2 * k2))
}

/// OPENCV: `k1, k2, p1, p2`.
#[arael::function]
fn opencv(n: vect2sym, k1: E, k2: E, p1: E, p2: E) -> vect2sym {
    n + brown(n, k1, k2, p1, p2)
}

/// FULL_OPENCV: `k1, k2, p1, p2, k3, k4, k5, k6`, the rational radial
/// model.
#[arael::function]
fn full_opencv(n: vect2sym, k1: E, k2: E, p1: E, p2: E, k3: E, k4: E, k5: E, k6: E)
    -> vect2sym {
    let (u, v) = (n.x, n.y);
    let u2 = u * u;
    let v2 = v * v;
    let uv = u * v;
    let r2 = u2 + v2;
    let radial = (1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1.0 + r2 * (k4 + r2 * (k5 + r2 * k6)));
    vect2sym::from_components(u * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2),
                              v * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2))
}

/// OPENCV_FISHEYE: `k1, k2, k3, k4`, four radial terms in the fisheye
/// angle.
#[arael::function]
fn opencv_fisheye(n: vect2sym, k1: E, k2: E, k3: E, k4: E) -> vect2sym {
    let m = fisheye_normal(n);
    let t2 = m.x * m.x + m.y * m.y;
    m * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
}

/// THIN_PRISM_FISHEYE: `k1, k2, p1, p2, k3, k4, sx1, sy1`, radial,
/// tangential and thin-prism terms in the fisheye angle.
#[arael::function]
fn thin_prism_fisheye(n: vect2sym, k1: E, k2: E, p1: E, p2: E, k3: E, k4: E, sx1: E, sy1: E)
    -> vect2sym {
    let m = fisheye_normal(n);
    let (uu, vv) = (m.x, m.y);
    let u2 = uu * uu;
    let v2 = vv * vv;
    let uv = uu * vv;
    let r2 = u2 + v2;
    let radial = r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)));
    vect2sym::from_components(uu + uu * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2) + sx1 * r2,
                              vv + vv * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2) + sy1 * r2)
}

/// RAD_TAN_THIN_PRISM_FISHEYE: `k0, k1, k2, k3, k4, k5, p0, p1, s0, s1,
/// s2, s3`; six radial terms in the fisheye angle, then the tangential
/// and thin-prism terms on the radially distorted point.
#[arael::function]
fn rad_tan_thin_prism_fisheye(n: vect2sym, k0: E, k1: E, k2: E, k3: E, k4: E, k5: E,
                              p0: E, p1: E, s0: E, s1: E, s2: E, s3: E) -> vect2sym {
    let m = fisheye_normal(n);
    let t2 = m.x * m.x + m.y * m.y;
    let radial = 1.0 + t2 * (k0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * (k4 + t2 * k5)))));
    let x = radial * m.x;
    let y = radial * m.y;
    let x2 = x * x;
    let y2 = y * y;
    let xy = x * y;
    let r2 = x2 + y2;
    vect2sym::from_components(x + 2.0 * p1 * xy + p0 * (r2 + 2.0 * x2) + r2 * (s0 + r2 * s1),
                              y + 2.0 * p0 * xy + p1 * (r2 + 2.0 * y2) + r2 * (s2 + r2 * s3))
}

/// SIMPLE_RADIAL_FISHEYE: `k`, one radial term in the fisheye angle.
#[arael::function]
fn simple_radial_fisheye(n: vect2sym, k: E) -> vect2sym {
    let m = fisheye_normal(n);
    m * (1.0 + k * (m.x * m.x + m.y * m.y))
}

/// RADIAL_FISHEYE: `k1, k2`, two radial terms in the fisheye angle.
#[arael::function]
fn radial_fisheye(n: vect2sym, k1: E, k2: E) -> vect2sym {
    let m = fisheye_normal(n);
    let t2 = m.x * m.x + m.y * m.y;
    m * (1.0 + t2 * (k1 + t2 * k2))
}

/// FOV: `omega`, the field-of-view model. Near `omega = 0`, and near
/// the optical axis, a series replaces the atan form, at COLMAP's
/// thresholds.
#[arael::function]
fn fov(n: vect2sym, omega: E) -> vect2sym {
    let r2 = n.x * n.x + n.y * n.y;
    let omega2 = omega * omega;
    let t = tan(omega / 2.0);
    let factor = branch(omega2 - 1e-4,
                        branch(r2 - 1e-4,
                               atan(2.0 * sqrt(r2) * t) / (sqrt(r2) * omega),
                               -2.0 * t * (4.0 * r2 * t * t - 3.0) / (3.0 * omega)),
                        omega2 / 12.0 - omega2 * r2 / 3.0 + 1.0);
    n * factor
}

/// The division models' discriminant `w^2 - 4 k (u^2 + v^2)` of the
/// camera-frame point `p`: the models project `p` where it is
/// non-negative, whatever its depth.
#[arael::function]
fn division_disc2(p: vect3sym, k: E) -> E {
    p.z * p.z - 4.0 * k * (p.x * p.x + p.y * p.y)
}

/// SIMPLE_DIVISION and DIVISION: `k`, the distorted normalized point of
/// the camera-frame point `p`, projected from `p` itself; defined where
/// [`division_disc2`] is non-negative.
#[arael::function]
fn division(p: vect3sym, k: E) -> vect2sym {
    let r = 2.0 / (p.z + sqrt(division_disc2(p, k)));
    vect2sym::from_components(r * p.x, r * p.y)
}

/// EUCM's radicand `beta (u^2 + v^2) + w^2` of the camera-frame point
/// `p`.
#[arael::function]
fn eucm_rho2(p: vect3sym, beta: E) -> E {
    beta * (p.x * p.x + p.y * p.y) + p.z * p.z
}

/// EUCM's denominator `alpha rho + (1 - alpha) w` of the camera-frame
/// point `p`; defined where [`eucm_rho2`] is non-negative.
#[arael::function]
fn eucm_den(p: vect3sym, alpha: E, beta: E) -> E {
    alpha * sqrt(eucm_rho2(p, beta)) + (1.0 - alpha) * p.z
}

/// Whether EUCM projects the camera-frame point `p`: non-negative when
/// its depth, the radicand and the denominator pass COLMAP's checks.
#[arael::function]
fn eucm_ok(p: vect3sym, alpha: E, beta: E) -> E {
    let den = eucm_den(p, alpha, beta);
    branch(p.z - epsilon_for(p.z), branch(eucm_rho2(p, beta), den - epsilon_for(den), -1.0), -1.0)
}

/// EUCM: `alpha, beta`, the distorted normalized point of the
/// camera-frame point `p` under the extended unified camera model;
/// defined where [`eucm_ok`] is non-negative.
#[arael::function]
fn eucm(p: vect3sym, alpha: E, beta: E) -> vect2sym {
    let den = eucm_den(p, alpha, beta);
    vect2sym::from_components(p.x / den, p.y / den)
}

/// Whether EQUIRECTANGULAR projects the camera-frame point `p`:
/// non-negative unless `p` is at the camera centre.
#[arael::function]
fn equirectangular_ok(p: vect3sym) -> E {
    safe_sqrt(p.x * p.x + p.z * p.z) + abs(p.y) - epsilon_for(p.y)
}

/// EQUIRECTANGULAR: the camera-frame point `p` on a panorama `width`
/// pixels wide, as its longitude less the observed pixel `px`'s, wrapped
/// into half a turn either way as COLMAP wraps the residual across the
/// image seam, and its latitude over minus pi. [`project`] scales the
/// two to pixels.
#[arael::function]
fn equirectangular(p: vect3sym, width: E, px: vect2sym) -> vect2sym {
    let pi = 3.141592653589793;
    let longitude_obs = (px.x / width - 0.5) * (2.0 * pi);
    vect2sym::from_components(rad_diff(safe_atan2(p.x, p.z), longitude_obs),
                              -safe_atan2(-p.y, safe_sqrt(p.x * p.x + p.z * p.z)) / pi)
}

/// The pixel residual of the camera-frame point `p` through a camera of
/// the given `model` (see [`Cam::model`]) and projection test `check`
/// (see [`Cam::check`]) with focal lengths `fx, fy`, principal point
/// `pp` and extra parameters `e0` to `e11`, against the observed pixel
/// `px`; zero where the model cannot project `p`.
#[arael::function]
fn project(p: vect3sym, model: E, check: E, fx: E, fy: E, pp: vect2sym,
           e0: E, e1: E, e2: E, e3: E, e4: E, e5: E, e6: E, e7: E, e8: E, e9: E, e10: E,
           e11: E, px: vect2sym) -> [E; 2] {
    let (n, depth) = normalized(p);
    // Non-negative when the model projects p. Its own match, on `check`
    // rather than `model`, comes first, so an observation failing it
    // skips the projection.
    let ok = match check {
        0 => depth,
        1 => division_disc2(p, e0),
        2 => eucm_ok(p, e0, e1),
        _ => equirectangular_ok(p),
    };
    // The distorted point `d`, scaled by `(su, sv)` and offset by `c` to
    // the pixel. The models with one focal length read fx on both axes;
    // SKEWED_PINHOLE's skew shifts the horizontal offset.
    let (d, su, sv, c) = match model {
        0 => (n, fx, fy, pp),
        1 => (n, fx, fx, pp),
        2 => (simple_radial(n, e0), fx, fx, pp),
        3 => (radial(n, e0, e1), fx, fx, pp),
        4 => (opencv(n, e0, e1, e2, e3), fx, fy, pp),
        5 => (full_opencv(n, e0, e1, e2, e3, e4, e5, e6, e7), fx, fy, pp),
        6 => (opencv_fisheye(n, e0, e1, e2, e3), fx, fy, pp),
        7 => (thin_prism_fisheye(n, e0, e1, e2, e3, e4, e5, e6, e7), fx, fy, pp),
        8 => (rad_tan_thin_prism_fisheye(n, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11),
              fx, fy, pp),
        9 => (simple_radial_fisheye(n, e0), fx, fx, pp),
        10 => (radial_fisheye(n, e0, e1), fx, fx, pp),
        11 => (fov(n, e0), fx, fy, pp),
        12 => (fisheye_normal(n), fx, fx, pp),
        13 => (fisheye_normal(n), fx, fy, pp),
        14 => (n, fx, fy, vect2sym::from_components(pp.x + e0 * n.y, pp.y)),
        15 => (division(p, e0), fx, fx, pp),
        16 => (division(p, e0), fx, fy, pp),
        17 => (eucm(p, e0, e1), fx, fy, pp),
        _ => (equirectangular(p, e0, px), e0 / 6.283185307179586, e1,
              vect2sym::from_components(px.x, 0.5 * e1)),
    };
    [branch(ok, su * d.x + c.x - px.x, 0.0), branch(ok, sv * d.y + c.y - px.y, 0.0)]
}

// ---------------------------------------------------------------------------
// Entities
// ---------------------------------------------------------------------------

/// A frame: the rig's pose at one capture instant (world-to-rig) and
/// the images taken at it, one per camera. Freeze one translation
/// axis by clearing that axis' `Param`.
#[arael::model]
#[derive(Default)]
pub struct Pose<T: Float> {
    pub rot: QuaternionParam<T>,
    pub tx: Param<T>,
    pub ty: Param<T>,
    pub tz: Param<T>,
    pub images: std::vec::Vec<Image<T>>,
    pub rig_images: std::vec::Vec<RigImage<T>>,
    /// The position prior of the frame's reference camera, if any.
    pub prior: Option<PositionPrior<T>>,
    pub hb: SelfBlock<Pose<T>, T>,
}

/// A 3D point in world coordinates.
#[arael::model]
#[derive(Default)]
pub struct Point<T: Float> {
    pub pos: Param<vect3<T>>,
    pub hb: SelfBlock<Point<T>, T>,
}

/// A rig's reference camera (at the rig's origin). arael assembles an
/// entity up to its last live parameter, so the parameters are
/// declared in the order they are usually held: the focal lengths,
/// the extra parameters, the principal point last (COLMAP holds it by
/// default); a model's unused extra parameters sit past the ones it
/// has.
#[arael::model]
#[derive(Default)]
pub struct Cam<T: Float> {
    /// The camera model: 0 PINHOLE, 1 SIMPLE_PINHOLE, 2 SIMPLE_RADIAL,
    /// 3 RADIAL, 4 OPENCV, 5 FULL_OPENCV, 6 OPENCV_FISHEYE,
    /// 7 THIN_PRISM_FISHEYE, 8 RAD_TAN_THIN_PRISM_FISHEYE,
    /// 9 SIMPLE_RADIAL_FISHEYE, 10 RADIAL_FISHEYE, 11 FOV,
    /// 12 SIMPLE_FISHEYE, 13 FISHEYE, 14 SKEWED_PINHOLE,
    /// 15 SIMPLE_DIVISION, 16 DIVISION, 17 EUCM, 18 EQUIRECTANGULAR.
    pub model: u32,
    /// The test of whether the camera projects a point, which follows
    /// from the model: 0 in front of the camera (models 0 to 14), 1 the
    /// division models' discriminant, 2 EUCM's checks, 3 not at the
    /// camera centre (EQUIRECTANGULAR).
    pub check: u32,
    pub fx: Param<T>,
    /// Unused by the models with one focal length.
    pub fy: Param<T>,
    /// The extra parameters in COLMAP's order; a model reads as many as
    /// it has.
    pub e0: Param<T>,
    pub e1: Param<T>,
    pub e2: Param<T>,
    pub e3: Param<T>,
    pub e4: Param<T>,
    pub e5: Param<T>,
    pub e6: Param<T>,
    pub e7: Param<T>,
    pub e8: Param<T>,
    pub e9: Param<T>,
    pub e10: Param<T>,
    pub e11: Param<T>,
    /// (cx, cy)
    pub pp: Param<vect2<T>>,
    pub hb: VarSelfBlock<Cam<T>, T>,
}

/// A camera with a pose within its rig (rig-to-camera): the fields of
/// [`Cam`], then the pose.
#[arael::model]
#[derive(Default)]
pub struct RigCam<T: Float> {
    pub model: u32,
    pub check: u32,
    pub fx: Param<T>,
    pub fy: Param<T>,
    pub e0: Param<T>,
    pub e1: Param<T>,
    pub e2: Param<T>,
    pub e3: Param<T>,
    pub e4: Param<T>,
    pub e5: Param<T>,
    pub e6: Param<T>,
    pub e7: Param<T>,
    pub e8: Param<T>,
    pub e9: Param<T>,
    pub e10: Param<T>,
    pub e11: Param<T>,
    pub pp: Param<vect2<T>>,
    pub rot: QuaternionParam<T>,
    pub t: Param<vect3<T>>,
    pub hb: VarSelfBlock<RigCam<T>, T>,
}

// ---------------------------------------------------------------------------
// Images and observations.
//
// Moving images belong to a frame: the image holds the camera ref and
// the (pose, cam) tile all its observations write to; each observation
// holds its point ref and the two point tiles. In a body the frame
// reads as `pose` (the entity holding the image, `parent.parent =
// pose`), the image as `image` (`parent = image`), the camera through
// it as `image.cam`, the point through the observation's own ref.
//
// Fixed images hold their world-to-camera transform `r`, `t` as data
// (the frame and rig poses composed) and the camera ref; each
// observation holds its point ref and the point-camera tile.
// ---------------------------------------------------------------------------

/// One frame's observations through a reference camera.
#[arael::model]
#[derive(Default)]
pub struct Image<T: Float> {
    #[arael(ref = root.cams)]
    pub cam: Ref<Cam<T>>,
    pub obs: std::vec::Vec<Observation<T>>,
    pub hb_pose_cam: VarCrossBlock<Pose<T>, Cam<T>, T>,
}

/// One observation through a reference camera: pixel `px` of `point`.
#[arael::model]
#[arael(constraint([hb_point_pose, hb_point_cam, parent.hb_pose_cam],
    parent = image, parent.parent = pose, name = "pixel",
    loss = |s| root.w * colmap_loss(root.loss, s, root.k2), {
    project(in_rig(pose.rot.rotation_matrix(), pose.tx, pose.ty, pose.tz, point.pos),
        image.cam.model, image.cam.check, image.cam.fx, image.cam.fy, image.cam.pp,
        image.cam.e0, image.cam.e1, image.cam.e2, image.cam.e3, image.cam.e4, image.cam.e5,
        image.cam.e6, image.cam.e7, image.cam.e8, image.cam.e9, image.cam.e10, image.cam.e11,
        observation.px)
}))]
#[derive(Default)]
pub struct Observation<T: Float> {
    #[arael(ref = root.points)]
    pub point: Ref<Point<T>>,
    pub px: vect2<T>,
    pub hb_point_pose: CrossBlock<Point<T>, Pose<T>, T>,
    pub hb_point_cam: VarCrossBlock<Point<T>, Cam<T>, T>,
}

/// One frame's observations through a rig camera.
#[arael::model]
#[derive(Default)]
pub struct RigImage<T: Float> {
    #[arael(ref = root.rig_cams)]
    pub cam: Ref<RigCam<T>>,
    pub obs: std::vec::Vec<RigObservation<T>>,
    pub hb_pose_cam: VarCrossBlock<Pose<T>, RigCam<T>, T>,
}

/// One observation through a rig camera: pixel `px` of `point`.
#[arael::model]
#[arael(constraint([hb_point_pose, hb_point_cam, parent.hb_pose_cam],
    parent = image, parent.parent = pose, name = "pixel",
    loss = |s| root.w * colmap_loss(root.loss, s, root.k2), {
    let p_rig = in_rig(pose.rot.rotation_matrix(), pose.tx, pose.ty, pose.tz, point.pos);
    project(image.cam.rot.rotation_matrix() * p_rig + image.cam.t,
        image.cam.model, image.cam.check, image.cam.fx, image.cam.fy, image.cam.pp,
        image.cam.e0, image.cam.e1, image.cam.e2, image.cam.e3, image.cam.e4, image.cam.e5,
        image.cam.e6, image.cam.e7, image.cam.e8, image.cam.e9, image.cam.e10, image.cam.e11,
        rigobservation.px)
}))]
#[derive(Default)]
pub struct RigObservation<T: Float> {
    #[arael(ref = root.points)]
    pub point: Ref<Point<T>>,
    pub px: vect2<T>,
    pub hb_point_pose: CrossBlock<Point<T>, Pose<T>, T>,
    pub hb_point_cam: VarCrossBlock<Point<T>, RigCam<T>, T>,
}

/// The observations of one image with a constant pose, through a
/// reference camera.
#[arael::model]
#[derive(Default)]
pub struct FixedImage<T: Float> {
    #[arael(ref = root.cams)]
    pub cam: Ref<Cam<T>>,
    pub r: matrix3<T>,
    pub t: vect3<T>,
    pub obs: std::vec::Vec<FixedObservation<T>>,
}

/// One observation from a constant pose through a reference camera:
/// pixel `px` of `point`.
#[arael::model]
#[arael(constraint([hb_point_cam],
    parent = image, name = "pixel",
    loss = |s| root.w * colmap_loss(root.loss, s, root.k2), {
    project(image.r * point.pos + image.t,
        image.cam.model, image.cam.check, image.cam.fx, image.cam.fy, image.cam.pp,
        image.cam.e0, image.cam.e1, image.cam.e2, image.cam.e3, image.cam.e4, image.cam.e5,
        image.cam.e6, image.cam.e7, image.cam.e8, image.cam.e9, image.cam.e10, image.cam.e11,
        fixedobservation.px)
}))]
#[derive(Default)]
pub struct FixedObservation<T: Float> {
    #[arael(ref = root.points)]
    pub point: Ref<Point<T>>,
    pub px: vect2<T>,
    pub hb_point_cam: VarCrossBlock<Point<T>, Cam<T>, T>,
}

/// The observations of one image with a constant pose, through a rig
/// camera.
#[arael::model]
#[derive(Default)]
pub struct RigFixedImage<T: Float> {
    #[arael(ref = root.rig_cams)]
    pub cam: Ref<RigCam<T>>,
    pub r: matrix3<T>,
    pub t: vect3<T>,
    pub obs: std::vec::Vec<RigFixedObservation<T>>,
}

/// One observation from a constant pose through a rig camera: pixel
/// `px` of `point`.
#[arael::model]
#[arael(constraint([hb_point_cam],
    parent = image, name = "pixel",
    loss = |s| root.w * colmap_loss(root.loss, s, root.k2), {
    project(image.r * point.pos + image.t,
        image.cam.model, image.cam.check, image.cam.fx, image.cam.fy, image.cam.pp,
        image.cam.e0, image.cam.e1, image.cam.e2, image.cam.e3, image.cam.e4, image.cam.e5,
        image.cam.e6, image.cam.e7, image.cam.e8, image.cam.e9, image.cam.e10, image.cam.e11,
        rigfixedobservation.px)
}))]
#[derive(Default)]
pub struct RigFixedObservation<T: Float> {
    #[arael(ref = root.points)]
    pub point: Ref<Point<T>>,
    pub px: vect2<T>,
    pub hb_point_cam: VarCrossBlock<Point<T>, RigCam<T>, T>,
}

// ---------------------------------------------------------------------------
// Position priors: the prior position of a camera centre against the
// camera centre, whitened by `sqrt_info` (the transposed lower Cholesky
// factor of the prior's inverse covariance), under the root's prior
// loss.
// ---------------------------------------------------------------------------

/// The position prior of a frame's reference camera, whose centre is
/// the frame's: `-R^T t` of its world-to-rig pose.
#[arael::model]
#[arael(constraint(parent.hb, name = "prior",
    loss = |s| colmap_loss(root.prior_loss, s, root.prior_k2), {
    let t = vect3sym::from_components(pose.tx, pose.ty, pose.tz);
    let r = positionprior.sqrt_info
        * (positionprior.prior + pose.rot.rotation_matrix().transpose() * t);
    [r.x, r.y, r.z]
}))]
#[derive(Default)]
pub struct PositionPrior<T: Float> {
    pub prior: vect3<T>,
    pub sqrt_info: matrix3<T>,
}

/// The position prior of a rig camera in one frame, whose centre goes
/// through its rig pose: `-R^T (t + Rc^T tc)`, with `(R, t)` the
/// frame's world-to-rig pose and `(Rc, tc)` the camera's rig-to-camera
/// pose.
#[arael::model]
#[arael(constraint(hb, name = "prior",
    loss = |s| colmap_loss(root.prior_loss, s, root.prior_k2), {
    let t = vect3sym::from_components(pose.tx, pose.ty, pose.tz);
    let t_rig = t + cam.rot.rotation_matrix().transpose() * cam.t;
    let r = rigpositionprior.sqrt_info
        * (rigpositionprior.prior + pose.rot.rotation_matrix().transpose() * t_rig);
    [r.x, r.y, r.z]
}))]
#[derive(Default)]
pub struct RigPositionPrior<T: Float> {
    #[arael(ref = root.poses)]
    pub pose: Ref<Pose<T>>,
    #[arael(ref = root.rig_cams)]
    pub cam: Ref<RigCam<T>>,
    pub prior: vect3<T>,
    pub sqrt_info: matrix3<T>,
    pub hb: VarCrossBlock<Pose<T>, RigCam<T>, T>,
}

// ---------------------------------------------------------------------------
// Roots
// ---------------------------------------------------------------------------

/// One COLMAP bundle adjustment problem. The loss is a property of
/// the solve, not of an observation, so it lives here. The points are
/// the blocks the Schur complement eliminates.
#[arael::model]
#[arael(root, marginalize(points))]
#[derive(Default)]
pub struct BaProblem {
    pub poses: refs::Vec<Pose<f64>>,
    pub points: refs::Vec<Point<f64>>,
    pub cams: refs::Vec<Cam<f64>>,
    pub rig_cams: refs::Vec<RigCam<f64>>,
    pub fixed_images: refs::Vec<FixedImage<f64>>,
    pub rig_fixed_images: refs::Vec<RigFixedImage<f64>>,
    pub rig_priors: refs::Vec<RigPositionPrior<f64>>,
    /// COLMAP's loss function type for this solve.
    pub loss: u32,
    /// Squared loss scale, Ceres' `b = a * a`.
    pub k2: f64,
    /// Multiplier on the loss, COLMAP's loss function weight.
    pub w: f64,
    /// The position priors' loss function type, as `loss`.
    pub prior_loss: u32,
    /// Squared scale of the position priors' loss.
    pub prior_k2: f64,
}
