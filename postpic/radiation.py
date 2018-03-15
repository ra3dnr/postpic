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
    + Project potentials on the transverse direction to get rid of
    electrostatics and reduce to two components
    + Fourier transform them to get the spectrum
    + Interpolate the result on user-defined frequency grid
    + Multiply by frequency to pass from the potential to the field
    + Convert back to dimensional units
    + Repeat for all the values of angles
"""

import numpy as np
import pyfftw

try:
    import datahandling
    import particles
except ImportError:
    pass


__all__ = ['lienard_wiechert', 'lw_amplitudes', 'single_dir_amplitudes']


def lienard_wiechert(trajs, theta, phi, charges=np.array([])):
    """
    Returns the frequency grid and the spectral photon density of the radiation
    in the direction given by the angles (theta, phi).
    """
    basis = _spherical_basis(theta, phi)
    # Calculate the spectrum for given direction
    res_freqs, Es = single_dir_amplitudes(trajs, basis, charges=charges)

    return res_freqs, (np.absolute(Es[0])**2 +
            np.absolute(Es[1])**2)/137/4/np.pi**2*res_freqs


def lw_amplitudes(trajs, freqs, thetas, phis, charges=np.array([]), verboselevel=0):
    """
    Returns two 3d array of vector potential spectral amplitudes on the detector
    grid: for polarization alog theta and along phi respectively. The order of
    indeces is (phi, theta, omega)
    verboselevel parameter stands for printing the current status:
    0 - no printing
    1 - print only the angle iteration info
    >1 - print also the log for every direction
    """
    # debugging
    verboseprint = print if verboselevel else lambda *a, **k: None

    res = []

    verboseprint("Angular grid:\ntheta_min=%f, theta_max=%f, N_theta=%d\n\
                    phi_min=%f, phi_max=%f, N_phi=%d." %
                 (thetas[0], thetas[-1], len(thetas), phis[0], phis[-1], len(phis)))
    verboseprint("Values (theta, phi):")
    for p in phis:
        res_t = []
        for t in thetas:
            basis = _spherical_basis(t, p)
            # Calculate the spectrum for given direction
            fr, sp = single_dir_amplitudes(trajs, basis, verbose=(verboselevel > 1), charges=charges)
            res_t.append(np.array([_interp_cplx(freqs, fr, a, left=0., right=0.) for a in sp]))
            verboseprint("(%f, %f)" % (t, p), end=" ")
        verboseprint("\n")
        res.append(res_t)

    res = np.array(res)
    return res[:, :, 0, :], res[:, :, 1, :]

def _get_opt_size(n):
    """
    finds the next number which is the product of powers of 2,3,5,7
    """
    def no_good(n):
        primes = [2,3,5,7]
        for p in primes:
            while not n%p:
                n = n//p
        return n != 1

    while no_good(n):
        n += 1
    return n

def single_dir_amplitudes(trajs, basis, charges=np.array([]), verbose=False):
    """
    Returns the frequency grid and complex amplitudes of the radiation (transverse vector
    potential) in the direction of basis[0].
    basis[1,2] are the basis vectors for decomposition. Everything is
    supposed to be orthonormalized.  Verbose parameter stands for printing the
    current status.
    """
    # debugging
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

    # If no charges given, set them to unity
    if not charges.size:
        charges = np.full(len(trajs), 1.)

    for t, q in zip(trajs, charges):
        verboseprint(traj_count, end=" ")
        xs, us = t

        # Retarded time for the particle
        zs = xs[:, 0] - np.dot(xs[:, 1:], basis[0])

        # Interpolate four velocities on uniform retarded time grid
        us_even = np.array([np.interp(zs_even, zs, us[:, i],
                                      left=us[0, i], right=us[-1, i]) for i in np.arange(4)])

        # Vector potential depending on the lab time
        denoms = q / (us_even[0] - np.dot(basis[0], us_even[1:]))
        vect_pot += us_even[1:, :] * denoms[None, :]

        traj_count += 1

    # Plan the fft
    projected = pyfftw.empty_aligned((2,len(zs_even)), dtype='float64')
    As = pyfftw.empty_aligned((2,len(zs_even)//2+1), dtype='complex128')
    fft_obj = pyfftw.FFTW(projected, As)

    # Project on the transverse direction
    projected[:,:] = np.array([np.dot(basis[i], vect_pot) for i in [1, 2]])

    # Get the transformed vector potential and the frequencies
    verboseprint("\nFourier transform..")
    fft_obj()

    dz = zs_even[1] - zs_even[0]
    As = As * dz

    freqs = 2*np.pi*np.linspace(0,len(zs_even)//2,len(zs_even)//2+1)/(dz*len(zs_even))

    return freqs, As


def _deduce_ret_time_grid(trajs, n):
    """
    Returns the linspace for retarded time in given direction that captures all
    the particles with the finest resolution needed. Also takes care of fftw
    padding.
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
        xs = t[0]
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
    num_pts = _get_opt_size(int((fzmax - fzmin) / fdz) + 1)
    return np.linspace(fzmin, fzmax, num_pts)


def _spherical_basis(theta, phi):
    """
    Return the triple of orthonormal basis vectors for the angle (theta, phi):
    basis[0] - along radius
    basis[1] - along theta
    basis[2] - along phi
    """
    return np.array([
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)],
        [np.sin(phi), -np.cos(phi), 0.]])


def _interp_cplx(x, xp, fp, left=0., right=0.):
    """
    Interpolation for the spectrum. Since it is complex-valued, and the
    phase can be rapidly changing, interpolates absolute values and phases
    separately.
    """
    rl, rr = np.absolute(left), np.absolute(right)
    pl, pr = np.angle(left), np.angle(right)
    rs = np.absolute(fp)
    ps = np.angle(fp)
    return np.interp(x, xp, rs, left=rl, right=rr) *\
        np.exp(1j * np.interp(x, xp, ps, left=pl, right=pr))


def _normalize_units(particles):
    """
    Converts incomplete data in SI units to complete arrays xs,us with the same
    spacetime scale. Charge is in SI units.
    Returns xs, us, [normalization factor for time, for space, for spectral intensity].
    """
    # epsilon_0
    eps0 = 1.2566370614e-6
    # We will measure time in meters
    intNorm = (1. / 4 / np.pi)**2 / eps0

    data = particles.collect('t*c', 'x', 'y', 'z', 'gamma', 'gamma*beta_x',
                             'gamma*beta_y', 'gamma*beta_z', 'q')

    trajs = [(d[:4], d[4:8, :]) for d in data]
    charges = np.array([d[8] for d in data])

    return trajs, charges, intNorm
