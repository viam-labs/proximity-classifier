import asyncio
import pathlib

import numpy as np
import pytest

from viam.components.camera import Camera
from viam.errors import ValidationError
from viam.proto.app.robot import ComponentConfig
from viam.services.vision import CaptureAllResult
from viam.utils import dict_to_struct

from src.grid import layout_for, world_to_pixel
from src.pcd import parse_pcd
from src.proximity import ProximityClassifier

ROOT = pathlib.Path(__file__).resolve().parents[1]
CUPS = ROOT / "testdata" / "cups.pcd"

# Short cup on the left (+Y), tall cup on the right. These are the numbers in
# the PCD, which is what GetPointCloud returns.
LEFT = {"x": 9.1, "y": 3, "z": -3}
RIGHT = {"x": 6.3, "y": -3, "z": -1.5}


class CloudCamera:
    def __init__(self, data: bytes):
        self.data = data

    async def get_point_cloud(self, **kwargs):
        return self.data, "pointcloud/pcd"


def cups_bytes() -> bytes:
    return CUPS.read_bytes()


def classifier(classes: dict | None = None, radius: float = 0.5) -> ProximityClassifier:
    service = ProximityClassifier("cups")
    service.camera_name = "cup-camera"
    service.radius = radius
    service.min_points = 1
    service.view = "yz"
    if classes is None:
        classes = {"left-cup": LEFT, "right-cup": RIGHT}
    service.classes = {name: (point["x"], point["y"], point["z"]) for name, point in classes.items()}
    service.deps = {Camera.get_resource_name("cup-camera"): CloudCamera(cups_bytes())}
    return service


def test_both_cups_classify():
    found = asyncio.run(classifier().get_classifications_from_camera("cup-camera", 5))
    labels = {item.class_name for item in found}
    assert labels == {"left-cup", "right-cup"}
    assert all(item.confidence > 0 for item in found)


def test_empty_space_is_not_a_cup():
    found = asyncio.run(
        classifier({"left-cup": LEFT, "right-cup": RIGHT, "nowhere": {"x": 0, "y": 0, "z": 20}}).get_classifications_from_camera(
            "cup-camera", 5
        )
    )
    assert "nowhere" not in {item.class_name for item in found}


def test_grid_without_classes():
    service = classifier({})
    captured: CaptureAllResult = asyncio.run(
        service.capture_all_from_camera(
            "cup-camera",
            return_image=True,
            return_classifications=True,
        )
    )
    assert captured.classifications == []
    assert captured.image is not None
    jpeg = captured.image.data
    assert jpeg[:2] == b"\xff\xd8"
    assert len(jpeg) > 1000


def test_side_view_puts_the_short_cup_on_the_left():
    points = parse_pcd(cups_bytes())
    layout = layout_for(points, "yz")
    short_x, short_y = world_to_pixel(layout, np.array([LEFT["x"], LEFT["y"], LEFT["z"]]))
    tall_x, tall_y = world_to_pixel(layout, np.array([RIGHT["x"], RIGHT["y"], RIGHT["z"]]))
    assert short_x < tall_x
    assert short_y > tall_y


def test_camera_is_required():
    with pytest.raises(ValidationError):
        ProximityClassifier.validate(ComponentConfig(attributes=dict_to_struct({})))
    required, optional = ProximityClassifier.validate(
        ComponentConfig(attributes=dict_to_struct({"camera": "cup-camera", "classes": {"left-cup": LEFT}}))
    )
    assert required == ["cup-camera"]
    assert optional == []
