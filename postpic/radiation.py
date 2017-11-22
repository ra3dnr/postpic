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
Reconstructing the far field spectrum using 3D Lienard Wiechert potentials.

How it works:
    + Take the Lienard---Wiechert formulas for the retarded potentials
    + Deduce the retarded time grid that will capture all the particles with the
    finest resolution
    + Interpolate the four-velocities from the retarded time to the evenly
    spaced time grid for given direction
    + Sum the contributions from different particles
    + Fourier transform them to get the spectrum
    + Project potentials on the transverse direction to get rid of
    electrostatics
    + Interpolate the result on user-defined frequency grid
    + Multiply by frequency to pass from the potential to the field
    + Convert back to dimensional units
    + Repeat for all the values of angles
"""

import numpy as np


__all__ = ['lienard_wiechert', 'amplitudes', 'single_dir_amplitudes']


def lienard_wiechert(trajs, n):
    """
    Returns the spectrum of the radiation in the direction of vector n
    """
    # Add two basis vectors to n

    return 0


def amplitudes(trajs, freqs, thetas, phis, verboselevel=0):
    """
    Returns 4d array of amplitudes on the detector grid
    verboselevel parameter stands for printing the current status:
    0 - no printing
    1 - print only the angle iteration info
    >1 - print also the log for every direction
    """
    # Get ready for debugging
    verboseprint = print if verboselevel else lambda *a, **k: None

    res = []

    verboseprint("Angular grid:\ntheta_min=%f, theta_max=%f, N_theta=%d\n\
                    phi_min=%f, phi_max=%f, N_phi=%d." %
                 (thetas[0], thetas[-1], len(thetas), phis[0], phis[-1], len(phis)))
    verboseprint("Values (theta, phi):")
    for p in phis:
        res_t = []
        for t in thetas:
            # "almost" holonomic basis in spherical coordinates
            basis = np.array([
                [np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)],
                [np.cos(t) * np.cos(p), np.cos(t) * np.sin(p), -np.sin(t)],
                [np.sin(p), -np.cos(p), 0.]])
            # Calculate the spectrum for given direction
            res_t.append(single_dir_amplitudes(
                trajs, freqs, basis, verbose=(verboselevel > 1)))
            verboseprint("(%f, %f)" % (t, p), end=" ")
        verboseprint("\n")
        res.append(res_t)

    res = np.array(res)
    return res


def single_dir_amplitudes(trajs, freqs, basis, verbose=False):
    """
    Returns the complex amplitudes of the radiation in the direction of
    basis[0] interpolated on the frequency grid freqs.
    basis[1,2] are the basis vectors for decomposition. Everything is
    supposed to be orthonormalized.
    Verbose parameter stands for printing the current status.
    """
    # Get ready for the debugging
    verboseprint = print if verbose else lambda *a, **k: None

    # Get the range of rearded times we are interested in
    verboseprint("Deducing the retarded time grid..")
    zs_even = _deduce_ret_time_grid(trajs, basis[0])
    verboseprint("Retarded time grid parameters: min=%f, max=%f, N=%d" %
                 (zs_even[0], zs_even[-1], len(zs_even)))

    # Here the vector-potential in time domain will be stored
    vect_pot = np.zeros((3, len(zs_even)))

    verboseprint("Processing %d trajectories.." % len(trajs))
    # Counter for the verbose mode
    traj_count = 0

    for t in trajs:
        verboseprint(traj_count, end=" ")
        xs, us = t

        # Retarded time for the particle
        zs = xs[:, 0] - np.dot(xs[:, 1:], basis[0])

        # Interpolate four velocities on uniform retarded time grid
        us_even = np.array([np.interp(zs_even, zs, us[:, i],
                                      left=us[0, i], right=us[-1, i]) for i in np.arange(4)])

        # Vector potential depending on the lab time
        denoms = 1. / (us_even[0] - np.dot(basis[0], us_even[1:]))
        vect_pot += us_even[1:, :] * denoms[None, :]

        traj_count += 1

    # Get the transformed vector potential and the frequencies
    verboseprint("\nFourier transform..")
    As = np.fft.fft(vect_pot, axis=1) * (zs_even[1] - zs_even[0])
    freqs_old = np.fft.fftfreq(
        len(zs_even), (zs_even[1] - zs_even[0]) / 2 / np.pi)

    # We only need non-negative frequencies
    As = As[:, :len(freqs_old) // 2]
    freqs_old = freqs_old[0:len(freqs_old) // 2]

    # Project on the transverse direction
    projected = [np.dot(basis[i], As) for i in [1, 2]]

    # Interpolate on the desired frequency grid and multipy by frequency to pass
    # from the potential to the field
    res = np.array([_interp_cplx(freqs, freqs_old, p, left=0., right=0.) * freqs
                    for p in projected])

    return res


def _deduce_ret_time_grid(trajs, n):
    """
    Returns the linspace for retarded time in given direction that captures all
    the particles with the finest resolution needed
    """

    # Values to start with: take from the first trajectory
    xs = trajs[0][0]

    # Retarded time
    zs = xs[:, 0] - np.dot(xs[:, 1:], n)

    # Take the minimal step
    fdz = np.min(np.diff(zs))

    # Define the range of retarded times
    fzmax = zs[-1]
    fzmin = zs[0]

    # Repeat for all the particles. Take the maximal grid
    for t in trajs[1:]:
        xs, = t
        zs = xs[:, 0] - np.dot(xs[:, 1:], n)
        dz = np.min(np.diff(zs))
        zmax = zs[-1]
        zmin = zs[0]
        if dz < fdz:
            fdz = dz
        if zmax > fzmax:
            fzmax = zmax
        if zmin < fzmin:
            fzmin = zmin

    # Return the retarded time grid as a numpy array
    return np.linspace(fzmin, fzmax, int((fzmax - fzmin) / fdz) + 1)


def _interp_cplx(x, xp, fp, left=0., right=0.):
    """
    Smooth interpolation for the spectrum. Since it is complex-valued, and the
    phase can be rapidly changing, interpolates absolute values and phases
    separately.
    """
    rl, rr = np.absolute(left), np.absolute(right)
    pl, pr = np.angle(left), np.angle(right)
    rs = np.absolute(fp)
    ps = np.angle(fp)
    return np.interp(x, xp, rs, left=rl, right=rr) *\
        np.exp(1j * np.interp(x, xp, ps, left=pl, right=pr))


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
