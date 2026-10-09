import asyncio

from viam.module.module import Module
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.services.vision import Vision

from .proximity import ProximityClassifier

Registry.register_resource_creator(
    Vision.API,
    ProximityClassifier.MODEL,
    ResourceCreatorRegistration(ProximityClassifier.new, ProximityClassifier.validate),
)


async def main():
    module = Module.from_args()
    module.add_model_from_registry(Vision.API, ProximityClassifier.MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())
