"""This is a module to compute the axion-photon conversion in a rotating magnetic field.
"""

import numpy as np
from numpy.linalg import eig
from scipy.integrate import simpson
from scipy.interpolate import interp1d


def M2_over_2om(m1, m2, m3):
    """The matrix M^2/(2 omega)
    """
    res = np.array([[m1/2, 0, 0], [0, m2/2, 0], [0, 0, m3/2]])
    return res


def Hint(cB, th):
    """ The interaction matrix that is responsible for
    the aovided level crossing
    """
    res = np.array([[0, 0, cB*np.sin(th)/2.], [0, 0, cB *
                   np.cos(th)/2.], [cB*np.sin(th)/2., cB*np.cos(th)/2., 0]])
    return res


def diagonalize(hermitian_mtx, verify=False):
    """diagonalize a hermitian matrix and output
    the special unitary matrix that diagonalize it.
    The eivenvectors are sorted according to the size
    of the eigenvalues.
    """
    val, vec = eig(hermitian_mtx)
    sorted_idx_arr = val.argsort()
    val = val[sorted_idx_arr]
    # unitary_mtx = vec.transpose()[sorted_idx_arr]
    # unitary_mtx = unitary_mtx.transpose()
    unitary_mtx = vec[:, sorted_idx_arr]

    # correct the sign
    unitary_mtx = unitary_mtx*np.sign(np.linalg.det(unitary_mtx))

    if verify:
        print("eigenvalue:", val)
        print("hamiltonian:\n", hermitian_mtx)
        print("unitary before reordering:\n", vec)
        print("unitary matrix:\n", unitary_mtx)
        print("determinant: ", np.linalg.det(unitary_mtx))
        print("UHU^T before reordering\n", np.dot(
            vec.transpose(), np.dot(hermitian_mtx, vec)))
        print("UHU^T\n", np.dot(unitary_mtx.transpose(),
              np.dot(hermitian_mtx, unitary_mtx)))

    return val, unitary_mtx


def derivs(x, y,
           ma,
           omega,
           cB,
           mg2_over_om_fn,
           theta_fn,
           ):
    """The integrand to be evolved, corresponding to the coupled ODE in the notes.

    :param x: the distance propagated
    :param y: the array of gamma_perp, gamma_parallal, a
    :param ma: the axion mass
    :param omega: the energy of the axion-photon system
    :param cB: c_agamma * B
    :param mg2_over_om_fn: mgamma^2/omega as a function of distance
    :param theta_fn: theta(x)

    """

    ma2_over_om = ma**2/omega

    # integrand
    h_arr = np.zeros((3, 3), dtype='complex_')
    h_arr += np.array(M2_over_2om(mg2_over_om_fn(x),
                                  mg2_over_om_fn(x),
                                  ma2_over_om)
                      + Hint(cB, theta_fn(x))) * (-1.j)

    res = np.dot(h_arr, y)

    return res


def mixing_angle(x,
                 ma,
                 omega,
                 cB,
                 mg2_over_om_fn,
                 # dmg2_over_om_dx,
                 # dthdx
                 ):
    """The mixing angle

    :param x: the distance propagated
    :param ma: the axion mass
    :param omega: the energy of the axion-photon system
    :param cB: c_agamma * B
    :param mg2_over_om_fn: mgamma^2/omega as a function of distance
    """

    ma2_over_om = ma**2/omega
    x_arr, is_scalar = treat_as_arr(x)

    sin_alpha = np.sqrt(
        4.*cB**2/(4.*cB**2+(mg2_over_om_fn(x_arr)-ma2_over_om)**2))

    if is_scalar:
        sin_alpha = np.squeeze(sin_alpha)

    return sin_alpha


def Pag_nr_analytical(x, dthetadx, cB):
    """This is the analytical expression from Seokhoon's notes

    :param ma: the axion mass
    :param mg: the photon plasma frequency
    :param om: the energy
    :param dthetadx: dtheta/dx

    """
    Delta_ag = cB/2
    Delta_phi = dthetadx
    k = np.sqrt(Delta_ag**2 + Delta_phi**2)
    prob = Delta_ag**2/(Delta_ag**2+Delta_phi**2)**2*np.sin(x/2.*k)**2\
        * (2.*Delta_phi**2 + Delta_ag**2*(1.+np.cos(x*k)))
    return prob


def treat_as_arr(arg):
    """A routine to cleverly return scalars as (temporary and fake) arrays. True arrays are returned unharmed.
    """

    arr = np.asarray(arg)
    is_scalar = False

    # making sure scalars are treated properly
    if arr.ndim == 0:  # it is really a scalar!
        arr = arr[None]  # turning scalar into temporary fake array
        is_scalar = True  # keeping track of its scalar nature

    return arr, is_scalar


def get_theta(x_arr, domain_size, rnd_seed=None, order=2, cache=None):
    """Generate a realization of the magnetic field

    :param x:
    :returns:

    """
    xi = x_arr[0]
    xe = x_arr[-1]

    if order == 0:
        # discontinuous orientations
        domain_arr = np.arange(xi, xe, domain_size)
        if rnd_seed is not None:
            np.random.seed(rnd_seed)
        domain_phase = np.random.rand(len(domain_arr)) * 2.*np.pi

        res = []
        for x in x_arr:
            idx = np.searchsorted(domain_arr, x, side='right')
            if idx == len(domain_arr):
                idx = idx - 1
            phase = domain_phase[idx]
            res.append(phase + (x-domain_arr[idx])/domain_size*2.*np.pi)

    elif order == 1:
        # first order
        raise Exception('first order orientation angles is not realized yet.')

    elif order == 2:
        # second order
        domain_arr = np.arange(xi, xe, domain_size)

        # average ddtheta: 2pi ~ .5*ddthate*domain_size**2
        ddtheta_max = (2.*np.pi)*2/domain_size**2
        if rnd_seed is not None:
            np.random.seed(rnd_seed)
        ddtheta_edge_arr = (np.random.rand(len(domain_arr))-0.5) * ddtheta_max

        # populate the denser array of x_arr
        dthetadx2_arr = interp1d(
            domain_arr, ddtheta_edge_arr, kind='previous', bounds_error=False, fill_value='extrapolate')(x_arr)

        dx_arr = np.diff(x_arr, prepend=x_arr[0])
        # dx_arr = np.diff(x_arr, prepend=0.)

        # first integral
        dthetadx_arr = np.cumsum(dthetadx2_arr*dx_arr)

        # second integral
        theta_arr = np.cumsum(dthetadx_arr*dx_arr)

        res = (dthetadx2_arr, dthetadx_arr, theta_arr)

    return np.array(res)
