from pathlib import Path

import numpy as np

import pycolmap

from .imc import DatasetIMC2024
from .utils import SceneInfo


def _write_sfm(sfm_path: Path, image_names: list[str]) -> None:
    reconstruction = pycolmap.Reconstruction()
    camera = pycolmap.Camera.create_from_model_name(
        1, "SIMPLE_PINHOLE", 100.0, 100, 100
    )
    reconstruction.add_camera_with_trivial_rig(camera)
    for image_id, name in enumerate(image_names, start=1):
        image = pycolmap.Image(
            name=name, camera_id=camera.camera_id, image_id=image_id
        )
        reconstruction.add_image_with_trivial_frame(
            image,
            pycolmap.Rigid3d(
                pycolmap.Rotation3d(),
                np.array([[float(image_id)], [0.0], [0.0]]),
            ),
        )
    sfm_path.mkdir(parents=True)
    reconstruction.write_text(sfm_path)


def test_prepare_scene_takes_the_on_disk_image_names(tmp_path: Path) -> None:
    scene_path = tmp_path / "imc2024" / "train" / "all" / "scene"
    image_path = scene_path / "images"
    image_path.mkdir(parents=True)
    for name in ("00001.png", "00002.png"):
        (image_path / name).touch()
    # The ground truth names the images jpg; one of them is not on disk.
    _write_sfm(scene_path / "sfm", ["00001.jpg", "00002.jpg", "00003.jpg"])

    scene_info = SceneInfo(
        dataset="imc2024",
        category="all",
        scene="scene",
        num_images=2,
        workspace_path=tmp_path / "workspace",
        image_path=image_path,
        sparse_gt_path=scene_path / "sparse_gt",
        has_camera_priors=False,
        colmap_extra_args=[],
    )
    dataset = DatasetIMC2024(tmp_path, [], [], tmp_path / "runs", "run")
    dataset.prepare_scene(scene_info)

    sparse_gt = pycolmap.Reconstruction(str(scene_info.sparse_gt_path))
    assert sorted(image.name for image in sparse_gt.images.values()) == [
        "00001.png",
        "00002.png",
    ]
