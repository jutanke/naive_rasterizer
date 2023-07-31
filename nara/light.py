from enum import IntEnum
import numpy as np


class LightType(IntEnum):
    POINT = 0


class Light:

    def __init__(
        self,
        light_type: LightType,
        intensity: float,
        location: np.ndarray
    ):
        if isinstance(location, list):
            location = np.array(location)
        self.light_type = light_type
        self.intensity = intensity
        self.location = location

    def plot(self, ax):
        ax.scatter(self.location[0], self.location[1], self.location[2])


class Material:

    def __init__(
        self,
        specular_reflection_constant: float,
        diffuse_reflection_constant: float,
        ambient_reflection_constant: float,
        shininess: float
    ):
        self.specular_reflection_constant = specular_reflection_constant
        self.diffuse_reflection_constant = diffuse_reflection_constant
        self.ambient_reflection_constant = ambient_reflection_constant
        self.shininess = shininess
