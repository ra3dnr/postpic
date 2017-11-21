#
# This file is part of postpic.
#
# postpic is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# postpic is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with postpic. If not, see <http://www.gnu.org/licenses/>.
#
# Vasily Kharin, 2017
"""
Reconstructing the far field spectrum using Lienartd Wiechert potentials
"""

import numpy as np


__all__ = ['lienard_wiechert']


def lienard_wiechert(trajs, n)
    """
    Returns the spectrum of the radiation in the direction of vector n
    """
    # Add two basis vectors to n


    return 0

def amplitudes(trajs, basis)
    """
    Returns the complex amplitudes of the radiation in the direction of basis[0].
    basis[1,2] are the basis vectors for decomposition. Everything is
    supposed to be orthonormalized
    """

    return 0

def _deduce_freq_grid(trajs, n):
    """
    Returns the frequency grid with the highest resolution necessary for all
    the trajectories
    """

    return np.linspace(0.,1.,100)

def _single_traj_vect_pot(xs, us, n)
    """ 
    Returns the frequency and spectrum of the vector-potential in the direction
    of vector n for a single particle. xs[:,0] is time, xs[:,1,2,3] are the
    space components of the coordinate. Similar to us[:,:] which are the
    components of four-velocity. Units are assumed to be the same for space and
    time. The frequency is $omega$. Vector n is assumed to be normalized
    """

    # Retarded time
    zs = xs[:,0] - np.dot(xs[:,1:], n)

    # Vector potential depending on the lab time
    denoms = 1./(us[:,0] - np.dot(us[:,1:], n))
    At = us[:,i]*denoms

    # Get the timestep for the retarded time
    dz = np.min(np.diff(zs))

    # Make even grid in the retarded time
    zs_even = np.linspace(zs[0],zs[-1],int((zs[-1]-zs[0])/dz)+1)

    # Interpolate the vector potential on evenly distributed grid
    At_even = [np.interp(zs_even, zs, At, left=0., right = 0.) for i in [1,2,3]]

    # Get the transformed vector potential and frequencies
    As = np.fft.fft(At_even)*(zs_even[-1]-zs_even[0])
    freqs = np.fft.fftfreq(len(zs_even),(zs_even[1]-zs_even[0])/2/np.pi)

    # Return non-negative frequencies and corresponding components of vector
    # potential
    return freqs[:len(freqs)//2], As[:,:len(freqs)//2]

def _normalize_units(trajs):
    """
    Converts incomplete data in SI units to complete arrays xs,us with the same
    spacetime scale.
    Returns xs, us, [normalization factor for time, for space, for spectral intensity].
    """
    intNorm = 1.
    spaceNorm = 1.
    timeNorm = 1.
    xs = []
    us = []
    return xs, us, [timeNorm, spaceNorm, intNorm]
