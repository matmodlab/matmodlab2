import numpy as np
from .tensor import array_rep, matrix_rep, inv, symmetric_part, dot
import matmodlab2.core.linalg as la


def update_deformation(farg, darg, dt, kappa):
    """Update the deformation gradient and strain

    Parameters
    ----------
    farg : ndarray
        The deformation gradient
    darg : ndarray
        The symmetric part of the velocity gradient
    dt : float
        The time increment
    kappa : int or real
        The Seth-Hill parameter

    Returns
    -------
    f : ndarray
        The updated deformation gradient
    e : ndarray
        The updated strain

    Notes
    -----
    From the value of the Seth-Hill parameter kappa, current strain E,
    deformation gradient F, and symmetric part of the velocit gradient d,
    update the strain and deformation gradient.
    Deformation gradient is given by

                 dFdt = L*F                                             (1)

    where F and L are the deformation and velocity gradients, respectively.
    The solution to (1) is

                 F = F0*exp(Lt)

    Solving incrementally,

                 Fc = Fp*exp(Lp*dt)

    where the c and p refer to the current and previous values, respectively.

    With the updated F, Fc, known, the updated stretch is found by

                 U = (trans(Fc)*Fc)**(1/2)

    Then, the updated strain is found by

                 E = 1/kappa * (U**kappa - I)

    where kappa is the Seth-Hill strain parameter.
    """
    f0 = farg.reshape((3, 3))
    d = matrix_rep(darg, 0)
    ff = np.dot(la.expm(d * dt), f0)
    u = la.sqrtm(np.dot(ff.T, ff))
    if kappa == 0:
        eps = la.logm(u)
    else:
        eps = 1.0 / kappa * (la.powm(u, kappa) - np.eye(3, 3))

    if la.det(ff) <= 0.0:
        raise Exception("negative jacobian encountered")

    f = np.reshape(ff, (9,))
    e = array_rep(eps, (6,))

    return f, e


def defgrad_from_strain(E, kappa, flatten=1):
    """Compute the deformation gradient from the strain measure

    Parameters
    ----------
    E : ndarray (6,)
        Strain measure
    kappa : int or float
        Seth-Hill strain parameter
    flatten : bool, optional
        If True (default), return a flattened array

    Returns
    -------
    F : ndarray
        The deformation gradient

    """
    R = np.eye(3)
    I = np.eye(3)
    E = matrix_rep(E, 0)
    if kappa == 0:
        U = la.expm(E)
    else:
        U = la.powm(kappa * E + I, 1.0 / kappa)
    F = np.dot(R, U)
    if la.det(F) <= 0.0:
        raise Exception("negative jacobian encountered")
    if flatten:
        return F.flatten()
    return F


def strain_from_defgrad(farg, kappa):
    """Compute the strain measure from the deformation gradient

    Parameters
    ----------
    farg : ndarray (9,)
        Deformation gradient
    kappa : int or float
        Seth-Hill strain parameter
    flatten : bool, optional
        If True (default), return a flattened array

    Returns
    -------
    E : ndarray
        The strain measure

    Notes
    -----
    Update strain by

                 E = 1/kappa * (U**kappa - I)

    where kappa is the Seth-Hill strain parameter.
    """
    F = farg.reshape((3, 3))
    jac = la.det(F)
    if jac <= 0:
        raise ValueError("Negative or zero Jacobian")
    # convert deformation gradient to strain E with associated rotation
    R, U = la.polar_decomp(F)
    eps = strain_from_stretch(U, kappa)
    strain = array_rep(eps, (6,))
    return strain, R


def strain_from_stretch(u, kappa):
    """Convert the 3x3 stretch tensor to a strain tensor using the
    Seth-Hill parameter kappa and return a 9x1 array"""
    mat, orig_shape = matrix_rep(u)
    if abs(kappa) > 1e-12:
        mat = 1.0 / kappa * (la.powm(mat, kappa) - np.eye(3))
    else:
        mat = la.logm(mat)
    return array_rep(mat, orig_shape)


def rate_of_strain_to_rate_of_deformation(dedt, e, kappa, disp=0):
    """Compute symmetric part of velocity gradient given depsdt

    Parameters
    ----------
    dedt : ndarray
        Strain rate
    e : ndarray
        Strain
    kappa : int or float
        Seth-Hill parameter
    disp : bool, optional
        If True, return both d and dU

    Returns
    -------
    d : ndarray
        Symmetric part of the velocity gradient
    dU : ndarray
        Rate of stretch (if disp is True)

    Notes
    -----
    Velocity gradient L is given by
                L = dFdt * Finv
                  = dRdt*I*Rinv + R*dUdt*Uinv*Rinv
    where F, I, R, U are the deformation gradient, identity, rotation, and
    right stretch tensor, respectively. d*dt and *inv are the rate and
    inverse or *, respectively,

    The stretch U is given by
                 if kappa != 0:
                     U = (kappa*E + I)**(1/kappa)
                 else:
                     U = exp(E)
    and its rate (dUdt) is calculated using `rate_of_matrix_function`.
       Then
                 L = dot(dUdt, inv(U))
                 d = sym(L)
                 w = skew(L)

    """

    eps = matrix_rep(e, 0)
    depsdt = matrix_rep(dedt, 0)

    # Calculate the right stretch (U) and its rate
    if abs(kappa) <= 1e-12:
        u = la.expm(eps)
        dudt = la.rate_of_matrix_function(eps, depsdt, np.exp, np.exp)
    else:
        u = la.powm(kappa * eps + np.eye(3), 1.0 / kappa)
        f = lambda x: (kappa * x + 1.0) ** (1.0 / kappa)
        fprime = lambda x: (kappa * x + 1.0) ** (1.0 / kappa - 1.0)
        dudt = la.rate_of_matrix_function(eps, depsdt, f, fprime)

    # Calculate the velocity gradient (L) and its symmetric part
    l = np.dot(dudt, la.inv(u))
    d = (l + l.T) / 2.0

    d = array_rep(d, (6,))
    if not disp:
        return d
    return d, dudt


def scalar_volume_strain_to_tensor(ev, kappa):
    """Convert the scalar volume strain ev to a 2nd order symmetric tensor.

    Parameters
    ----------
    ev : float
        The volume strain
    kappa : float
        The Seth-Hill parameter

    Returns
    -------
    components : ndarray (6,)
        The components of the strain tensor

    Notes
    -----
    The returned tensor is represented as a first order 6d tensor.

    """
    if kappa * ev + 1.0 < 0.0:
        raise ValueError("1 + kappa * ev must be positive")

    if abs(kappa) < np.finfo(np.float).eps:
        # Log strain
        eij = ev / 3.0
    else:
        eij = (kappa * ev + 1.0) ** (1.0 / 3.0) - 1.0
        eij = eij / kappa
    components = np.array([eij, eij, eij, 0.0, 0.0, 0.0])
    return components


def rate_of_defomation_from_defgrad(F0, F1, dtime):
    Fdot = (F1 - F0) / dtime
    L = dot(Fdot, inv(F1))
    return symmetric_part(L)
