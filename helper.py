import numpy as np
from numpy.random import *

def der_arctan(traj_point: np.ndarray, AP_pos: np.ndarray):
    """ This function returns the derivative of arctan and the arctan value at the trajectory point, both utilized to
    compute the Taylor expansion.

    Args:
    traj_point (numpy array): The trajectory point in 2D space.
    AP_pos (numpy array): The position of the APs in 2D space.

    Returns:
    der_AoA (numpy array): The derivative of arctan value.
    AoA_0 (numpy array): The arctan value.
    """
    dim = AP_pos.shape[0]
    AoA_0 = np.zeros(dim)
    # if the MU is connected to two APs
    der_AoA = np.zeros((dim, 2))
    for i in np.arange(dim):
        delta_xy = traj_point - AP_pos[i]
        temp = delta_xy / np.sum(delta_xy ** 2)
        temp[1] = -temp[1]
        der_AoA[i] = temp[::-1]
        AoA_0[i] = np.arctan2(delta_xy[1], delta_xy[0])
        # AoA_0[i] = AoA_0[i] + (AoA_0[i] < 0)*2*np.pi
    return der_AoA, AoA_0


def create_uniform_particles(x_range: tuple, y_range: tuple, N: int):
    """ Creates uniform particles across the environment.

    Args:
        x_range (tuple): range of x-axis for the particles
        y_range (tuple): range of y-axis for the particles
        N (int): number of particles to be created

    Returns:
        particles (np.array): array of size (N,2) representing the positions of the particles on x and y axis
    """
    particles = np.empty((N, 2))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    return particles


def resample(particles: np.ndarray, particle_noise: np.ndarray, weights: np.ndarray, N: int):
    """ Resamples the particles based on their weight distribution.

    Args:
    - particles: array of particles
    - particle_noise: noise to add to the particles
    - weights: weight distribution of the particles
    - N: number of particles

    Returns:
    -resampled particles
    """
    position_indx = np.arange(N)
    # Dedicating more particles to the intervals with higher prob.
    bins = np.add.accumulate(weights)
    position_indx_resampled = position_indx[np.digitize(random_sample(N), bins)]
    particles_resampled = particles[position_indx_resampled] + particle_noise
    return particles_resampled


def aoa_std(x, a1: float = 1.82, b1: float = 22.52, a2: float = 1.68, b2: float = 28.38,
            a3: float = 0.68, b3: float = 9.34):
    """ Computes the standard deviation corresponding to a ground-truth AoA calulated using the trajectory of a user.
    The function has been fitted using the simulation data from Quadriga channel model for a straight line scenario.
    """
    return a1 * np.exp(-(x - 45) ** 2 / 2 / b1 ** 2) + a2 * np.exp(-(x - 130) ** 2 / 2 / b2 ** 2) - a3 * np.exp(
        -(x - 90) ** 2 / 2 / b3 ** 2)
