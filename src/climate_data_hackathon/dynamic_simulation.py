"""Author: Andrew Martin
Creation date: 31/3/26

Script holding class definitions for classes allowing for dynamical simulations between objects with inter-object forces.
The dynamical system is implemented through damped Verlet integration.
"""

from __future__ import annotations
import numpy as np
from typing import Any, TypeAlias, Callable, Self
from collections import deque

class VerletIntegrable:
    def __init__(self, initial_x, damping: float):
        self.x_values = deque(maxlen=2)
        for _ in range(2):
            self.x_values.append(initial_x)

        self.accelarations = list()
        self.partial_dx = None
        self.damping = damping

    @property
    def x(self):
        return self.x_values[-1]

    @property
    def net_accelaration(self):
        return np.sum(self.accelarations, axis=0)

    def reset_velocity(self):
        self.x_values.append(self.x)
        return self

    def start_frame(self):
        self.accelarations = list()
        return self

    def integrated_dx(self, dt: float):
        dx = self.x - self.x_values[-2] + self.net_accelaration * dt**2
        return dx

    def update(self):
        self.x_values.append(self.x + self.partial_dx*self.damping)
        return self.partial_dx

    def apply_partial_update(self, dt:float):
        self.partial_dx = self.integrated_dx(dt)
        return self

    def cap_partial_dx(self, cap: float):
        self.partial_dx = self.partial_dx / (norm:=np.linalg.norm(self.partial_dx)) * np.clip(norm,0,cap)
        return self




class Point(VerletIntegrable):
    def __init__(self, initial_x: np.ndarray, mass: float, radius: float, name: str, color="blue"):
        super().__init__(initial_x)
        self.mass = mass
        self.radius = radius
        self.name = name
        self.forces = list()
        self.color = color

    def start_frame(self):
        super().start_frame()
        self.forces = list()
        return self
    
    def displacement_to(self, other) -> np.ndarray:
        return other.x - self.x

    def distance_to(self, other) -> float:
        return np.linalg.norm(self.displacement_to(other)) + epsilon

    def determine_exclusion_displacement(self, other: Point) -> np.ndarray:
        """Given the radius of two objects, calculate how much overlap there is between the objects, as a vector from self to other"""
        distance = self.distance_to(other)
        if distance > self.radius + other.radius:
            return np.array([0,0])
        overlap = self.radius + other.radius - distance
        return self.displacement_to(other) / distance * overlap

    def apply_exclusion_displacement(self, other: Point, fraction_reduce_overlap: float=1) -> np.ndarray:
        """Given the  exclusion displacement and masses of the two points, shift them such that their overlap is reduced by fraction_reduce_overlap.
        NOTE: this applies the displacement to both objects, so care should be taken to not double exlusion displacements.
        """
        combined_mass = self.mass + other.mass
        exclusion_displacement = self.determine_exclusion_displacement(other)
        other.x_values[-1] += exclusion_displacement * 0.5 * fraction_reduce_overlap#* self.mass / combined_mass * fraction_reduce_overlap
        self.x_values[-1] -= exclusion_displacement * 0.5 * fraction_reduce_overlap#* other.mass / combined_mass * fraction_reduce_overlap
        return self

    def overlaps(self, other):
        return not np.all((self.determine_exclusion_displacement(other) == np.array([0,0])))

    def add_force(self, force):
        self.forces.append(force)
        return self
    
    def set_accelaration_from_forces(self):
        self.accelarations = [
            force / self.mass
            for force in self.forces
        ]
        return self


def inverse_square_force(p1: Point, p2: Point, G: float):
    f_2_on_1 = p1.displacement_to(p2) / p1.distance_to(p2)**3 * G * p1.mass * p2.mass
    return f_2_on_1

def inverse_square_with_constant_at_proximity(p1: Point, p2: Point, G: float, r_cutoff: float, constant: float):
    disp = p1.displacement_to(p2)
    r = p1.distance_to(p2)
    if r < r_cutoff:
        f_2_on_1 = disp / r * constant
    else:
        f_2_on_1 = disp / r**3 * G
    return f_2_on_1 * p1.mass * p2.mass
