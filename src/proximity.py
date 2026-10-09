from __future__ import annotations

from typing import ClassVar, Mapping, Optional, Sequence

import numpy as np
from typing_extensions import Self

from viam.components.camera import Camera
from viam.errors import NotSupportedError, ValidationError
from viam.media.video import CameraMimeType, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection, GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily
from viam.services.vision import CaptureAllResult, Vision
from viam.utils import ValueTypes, struct_to_dict

from .grid import VIEWS, render_grid
from .pcd import parse_pcd

DEFAULT_RADIUS = 0.5
DEFAULT_MIN_POINTS = 1
DEFAULT_VIEW = "yz"


class ProximityClassifier(Vision):
    """Classify a point cloud when points fall inside a radius of a configured location.

    Every matching class is returned. CaptureAllFromCamera draws a grid of the
    cloud even when no classes are configured, which is how you read off coordinates.
    """

    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "vision"), "proximity-classifier")

    def __init__(self, name: str):
        super().__init__(name)
        self.camera_name = ""
        self.classes: dict[str, tuple[float, float, float]] = {}
        self.radius = DEFAULT_RADIUS
        self.min_points = DEFAULT_MIN_POINTS
        self.view = DEFAULT_VIEW
        self.deps: Mapping[ResourceName, ResourceBase] = {}

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        classifier = cls(config.name)
        classifier.reconfigure(config, dependencies)
        return classifier

    @classmethod
    def validate(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        camera = str(attrs.get("camera") or "")
        if not camera:
            raise ValidationError("camera is required")
        _parse_classes(attrs.get("classes") or {})
        radius = attrs.get("radius", DEFAULT_RADIUS)
        if float(radius) <= 0:
            raise ValidationError("radius must be greater than 0")
        min_points = attrs.get("min_points", DEFAULT_MIN_POINTS)
        if int(min_points) <= 0:
            raise ValidationError("min_points must be greater than 0")
        view = str(attrs.get("view") or DEFAULT_VIEW)
        if view not in VIEWS:
            raise ValidationError("view must be yz, xz, or xy")
        return [camera], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        attrs = struct_to_dict(config.attributes)
        camera = str(attrs.get("camera") or "")
        if Camera.get_resource_name(camera) not in dependencies:
            raise ValidationError(f"camera {camera!r} is not available")
        self.camera_name = camera
        self.classes = _parse_classes(attrs.get("classes") or {})
        self.radius = float(attrs.get("radius", DEFAULT_RADIUS))
        self.min_points = int(attrs.get("min_points", DEFAULT_MIN_POINTS))
        self.view = str(attrs.get("view") or DEFAULT_VIEW)
        self.deps = dependencies

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> list[Classification]:
        self._check_camera(camera_name)
        found = self._classify(await self._points())
        if count > 0:
            return found[:count]
        return found

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> list[Classification]:
        raise NotSupportedError(f"proximity classifier reads a point cloud from camera {self.camera_name!r}")

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> list[Detection]:
        raise NotSupportedError("proximity classifier does not detect")

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> list[Detection]:
        raise NotSupportedError("proximity classifier does not detect")

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> list[PointCloudObject]:
        raise NotSupportedError("proximity classifier does not segment")

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> GetPropertiesResponse:
        return GetPropertiesResponse(
            classifications_supported=True,
            detections_supported=False,
            object_point_clouds_supported=False,
            default_camera=self.camera_name,
        )

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        self._check_camera(camera_name)
        points = await self._points()
        image = None
        classifications = None
        if return_image:
            view = self.view
            if extra and extra.get("view"):
                view = str(extra["view"])
                if view not in VIEWS:
                    raise ValidationError("view must be yz, xz, or xy")
            jpeg = render_grid(points, self.classes, self.radius, view)
            image = ViamImage(jpeg, CameraMimeType.JPEG)
        if return_classifications:
            classifications = self._classify(points)
        return CaptureAllResult(image=image, classifications=classifications)

    def _check_camera(self, camera_name: str) -> None:
        if camera_name and camera_name != self.camera_name:
            raise ValidationError(f"proximity classifier reads camera {self.camera_name!r}, not {camera_name!r}")

    async def _points(self) -> np.ndarray:
        camera = self.deps[Camera.get_resource_name(self.camera_name)]
        data, _mime = await camera.get_point_cloud()
        if not data:
            raise ValidationError(f"camera {self.camera_name!r} returned no point cloud")
        return parse_pcd(data)

    def _classify(self, points: np.ndarray) -> list[Classification]:
        if not self.classes or points.size == 0:
            return []
        found: list[Classification] = []
        radius2 = self.radius * self.radius
        for name, center in self.classes.items():
            delta = points - np.asarray(center, dtype=np.float64)
            count = int(np.count_nonzero(np.einsum("ij,ij->i", delta, delta) <= radius2))
            if count < self.min_points:
                continue
            confidence = min(1.0, count / (self.min_points * 10))
            found.append(Classification(class_name=name, confidence=confidence))
        found.sort(key=lambda item: (-item.confidence, item.class_name))
        return found


def _parse_classes(raw) -> dict[str, tuple[float, float, float]]:
    classes: dict[str, tuple[float, float, float]] = {}
    for name, point in raw.items():
        if not name:
            raise ValidationError("a class name is empty")
        try:
            classes[str(name)] = (float(point["x"]), float(point["y"]), float(point["z"]))
        except (KeyError, TypeError, ValueError) as err:
            raise ValidationError(f"class {name!r} needs x, y, and z") from err
    return classes
