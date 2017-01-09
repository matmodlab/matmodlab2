import warnings
import numpy as np
import scipy.linalg
from copy import deepcopy as copy
from .environ import environ

VOIGT = np.array([1, 1, 1, 2, 2, 2], dtype=np.float64)
I6 = np.array([1, 1, 1, 0, 0, 0], dtype=np.float64)
epsilon = np.finfo(float).eps


def is_listlike(item):
    """Is item list like?"""
    try:
        [x for x in item]
        return not is_stringlike(item)
    except TypeError:
        return False

def is_stringlike(item):
    """Is item string like?"""
    try:
        item + 'string'
        return True
    except TypeError:
        return False

def is_scalarlike(item):
    """Is item scalar like?"""
    try:
        item + 4.
        return True
    except TypeError:
        return False

def determinant(a):
    """Determinant of a"""
    return apply_matrix_fun_to_array(a, np.linalg.det)

def inverse(a):
    """Inverse of a"""
    return apply_matrix_fun_to_array(a, np.linalg.inv)

def expm(a):
    """Compute the matrix exponential of a 3x3 matrix"""
    return apply_matrix_fun_to_array(a, scipy.linalg.expm)

def powm(a, m):
    """Compute the matrix power of a 3x3 matrix"""
    fun = lambda x: scipy.linalg.fractional_matrix_power(x, m)
    return apply_matrix_fun_to_array(a, fun)

def sqrtm(a):
    """Compute the square root of a 3x3 matrix"""
    return apply_matrix_fun_to_array(a, scipy.linalg.sqrtm)

def logm(a):
    """Compute the matrix logarithm of a 3x3 matrix"""
    return apply_matrix_fun_to_array(a, scipy.linalg.logm)

def apply_matrix_fun_to_array(arr, matrix_fun):
    """Apply the matrix fun to array arr

    Parameters
    ----------
    arr : ndarray
        Some array
    matrix_fun : callable
        A matrix function

    Returns
    -------
    a_fun : ndarray
        Array with matrix_fun applied

    """
    mat, orig_shape = array_to_mat(arr)
    mat2 = matrix_fun(mat)
    # Determine if mat2 is a scalar
    try:
        return float(mat2)
    except TypeError:
        return mat_to_array(mat2, orig_shape)

def dot(a, b):
    """perform matrix multiplication on two 3x3 matricies"""
    return np.dot(a, b)

def stretch_to_strain(u, k):
    """Convert the 3x3 stretch tensor to a strain tensor using the
    Seth-Hill parameter k and return a 6x1 array"""
    if abs(k) > 1e-12:
        I = np.eye(3,3)
        fun = lambda x: 1./k*(scipy.linalg.fractional_matrix_power(x,k)-I)
        return apply_matrix_fun_to_array(u, fun) * VOIGT
    return apply_matrix_fun_to_array(u, scipy.linalg.logm) * VOIGT

def symarray(a):
    """Convert a 3x3 matrix to a 6x1 array representing a symmetric matix."""
    mat = (a + a.T) / 2.0
    return np.array([mat[0, 0], mat[1, 1], mat[2, 2],
                     mat[0, 1], mat[1, 2], mat[0, 2]])

def asarray(a, n=6):
    """Convert a 3x3 matrix to array form"""
    if n == 6:
        return symarray(a)
    elif n == 9:
        return np.reshape(a, (1, 9))[0]
    else:
        raise Exception("Invalid value for n. Given {0}".format(n))

def as_3x3_matrix(a):
    """Convert a 6x1 array to a 3x3 symmetric matrix"""
    a = np.asarray(a)
    if a.shape == (6,):
        # Symmetric matrix
        return np.array([[a[0], a[3], a[5]],
                         [a[3], a[1], a[4]],
                         [a[5], a[4], a[2]]])
    elif a.shape == (9,):
        return np.reshape(a, (3,3))
    elif a.shape == (3,3):
        return a
    raise ValueError('Unknown shape')

def array_to_mat(a):
    """Convert array to matrix"""
    if a.shape == (6,):
        mat = as_3x3_matrix(a)
        orig_shape = (6,)
    elif a.shape == (9,):
        mat = np.reshape(a, (3,3))
        orig_shape = (9,)
    elif a.shape == (3,3):
        mat = a
        orig_shape = (3,3)
    else:
        raise ValueError('Unknown shape')
    return mat, orig_shape

def mat_to_array(mat, shape):
    """Reverse of array_to_mat"""
    if shape == (6,):
        return symarray(mat)
    if shape == (9,):
        return mat.flatten()
    if shape == (3,3):
        return mat
    raise ValueError('Unknown shape')

def diag(a):
    """Returns the diagonal part of a 3x3 matrix."""
    return np.array([[a[0, 0],     0.0,     0.0],
                       [0.0,   a[1, 1],     0.0],
                       [0.0,       0.0, a[2, 2]]])

def isdiag(a):
    """Determines if a matrix is diagonal."""
    return np.sum(np.abs(a - diag(a))) <= epsilon(a)

def apply_fun_to_matrix(a, f):
    """Apply function to eigenvalues of a 3x3 matrix then recontruct the matrix
    with the new eigenvalues and the eigenprojectors"""
    if isdiag(a):
        return np.array([[f(a[0, 0]),        0.0,        0.0],
                         [       0.0, f(a[1, 1]),        0.0],
                         [       0.0,        0.0, f(a[2, 2])]])

    vals, vecs = np.linalg.eig(a)

    # Compute eigenprojections
    p0 = np.outer(vecs[:, 0], vecs[:, 0])
    p1 = np.outer(vecs[:, 1], vecs[:, 1])
    p2 = np.outer(vecs[:, 2], vecs[:, 2])

    return f(vals[0]) * p0 + f(vals[1]) * p1 + f(vals[2]) * p2

def rate_of_matrix_function(A, Adot, f, fprime):
    """Find the rate of the tensor A

    Parameters
    ----------
    A : ndarray (3,3)
        A diagonalizable tensor
    Adot : ndarray (3,3)
        Rate of A
    f : callable
    fprime : callable
        Derivative of f

    Returns
    -------
    Ydot : ndarray (3,3)

    Notes
    -----
    For a diagonalizable tensor A (the strain) which has a quasi-arbitrary
    spectral expansion

    .. math::
        A = \sum_{i=1}^3 \lambda_i P_{i}

    and if a second tensor Y is a principal function of A, defined by

    .. math::
        Y = \sum_{i=1}^3 f(\lambda_i) P_i,

    compute the time rate \dot{Y}. Algorithm taken from Brannon's
    Tensor book, from the highlighted box near Equation (28.404) on
    page 550.

    """

    # Compute the eigenvalues and eigenprojections.
    eig_vals, eig_vecs = np.linalg.eig(A)
    eig_projs = [np.outer(eig_vecs[:, i], eig_vecs[:, i]) for i in [0, 1, 2]]

    # Assemble the rate of Y.
    Ydot = np.zeros((3, 3))
    for eigi, proji in zip(eig_vals, eig_projs):
        for eigj, projj in zip(eig_vals, eig_projs):
            if eigi == eigj:
                gamma = fprime(eigi)
            else:
                gamma = (f(eigi) - f(eigj)) / (eigi - eigj)
            Ydot += gamma * np.dot(proji, np.dot(Adot, projj))

    return Ydot

def rate_of_strain_to_rate_of_deformation(dedt, e, k, disp=0):
    """ Compute symmetric part of velocity gradient given depsdt

    Parameters
    ----------
    dedt : ndarray
        Strain rate
    e : ndarray
        Strain
    k : int or float
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
                 if k != 0:
                     U = (k*E + I)**(1/k)
                 else:
                     U = exp(E)
    and its rate (dUdt) is calculated using `rate_of_matrix_function`.
       Then
                 L = dot(dUdt, inv(U))
                 d = sym(L)
                 w = skew(L)

    """

    eps = as_3x3_matrix(e / VOIGT)
    depsdt = as_3x3_matrix(dedt / VOIGT)

    # Calculate the right stretch (U) and its rate
    if abs(k) <= 1e-12:
        u = scipy.linalg.expm(eps)
        dudt = rate_of_matrix_function(eps, depsdt, np.exp, np.exp)
    else:
        u = scipy.linalg.fractional_matrix_power(k * eps + np.eye(3), 1.0 / k)
        f = lambda x: (k * x + 1.0) ** (1.0 / k)
        fprime = lambda x: (k * x + 1.0) ** (1.0 / k - 1.0)
        dudt = rate_of_matrix_function(eps, depsdt, f, fprime)

    # Calculate the velocity gradient (L) and its symmetric part
    l = np.dot(dudt, np.linalg.inv(u))
    d = (l + l.T) / 2.0

    d = symarray(d) * VOIGT
    if not disp:
        return d
    return d, dudt

def update_deformation(farg, darg, dt, k):
    """Update the deformation gradient and strain

    Parameters
    ----------
    farg : ndarray
        The deformation gradient
    darg : ndarray
        The symmetric part of the velocity gradient
    dt : float
        The time increment
    k : int or real
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

                 E = 1/k * (U**k - I)

    where k is the Seth-Hill strain parameter.
    """
    f0 = farg.reshape((3, 3))
    d = as_3x3_matrix(darg / VOIGT)
    ff = np.dot(expm(d * dt), f0)
    u = sqrtm(np.dot(ff.T, ff))
    if k == 0:
        eps = logm(u)
    else:
        eps = 1.0 / k * (powm(u, k) - np.eye(3, 3))

    if np.linalg.det(ff) <= 0.0:
        raise Exception("negative jacobian encountered")

    f = asarray(ff, 9)
    e = symarray(eps) * VOIGT

    return f, e

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

                 E = 1/k * (U**k - I)

    where k is the Seth-Hill strain parameter.
    """
    f = farg.reshape((3, 3))
    u = sqrtm(np.dot(f.T, f))
    if k == 0:
        eps = logm(u)
    else:
        eps = 1.0 / k * (powm(u, k) - np.eye(3, 3))

    if np.linalg.det(f) <= 0.0:
        raise Exception("negative jacobian encountered")

    e = symarray(eps) * VOIGT

    return e

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
    E, _ = array_to_mat(E / VOIGT)
    if kappa == 0:
        U = expm(E)
    else:
        U = powm(k * E + I, 1. / k)
    F = np.dot(R, U)
    if np.linalg.det(F) <= 0.0:
        raise Exception("negative jacobian encountered")
    if flatten:
        return F.flatten()
    return F

def _isotropic_part(A):
    assert A.shape == (3,3)
    return np.trace(A) / 3. * np.eye(3)

def isotropic_part(a):
    """Return isotropic part of A"""
    return apply_matrix_fun_to_array(a, _isotropic_part)

def _deviatoric_part(A):
    assert A.shape == (3,3)
    return A - isotropic_part(A)

def deviatoric_part(a):
    return apply_matrix_fun_to_array(a, _deviatoric_part)

def double_dot(a, b):
    a_mat, a_shape = array_to_mat(a)
    b_mat, b_shape = array_to_mat(b)
    assert a.shape == (3,3)
    assert b.shape == a.shape
    return np.sum(a * b)

def magnitude(a):
    fun = lambda x: np.sqrt(double_dot(x, x))
    m = apply_matrix_fun_to_array(a, fun)
    return m

def symmetric_dyad(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == (3,)
    assert a.shape == b.shape
    return np.array([a[0] * b[0], a[1] * b[1], a[2] * b[2],
                     a[0] * b[1], a[1] * b[2], a[0] * b[2]],
                    dtype=np.float64)

def ddot(a, b):
    # double of symmetric tensors stored as 6x1 arrays
    assert a.shape == (6,)
    assert a.shape == b.shape
    return np.sum(a * b * VOIGT)

def trace(a):
    return apply_matrix_fun_to_array(a, np.trace)

def invariants(a, n=None, mechanics=False):
    if mechanics:
        i1 = trace(a)
        rootj2 = magnitude(deviatoric_part(a)) / np.sqrt(2.)
        return i1, rootj2

    a = as_3x3_matrix(a)
    asq = np.dot(a, a)
    deta = np.linalg.det(a)
    tra = np.trace(a)

    b = np.zeros(5)
    b[0] = tra
    b[1] = .5 * (tra ** 2 - np.trace(asq))
    b[2] = deta
    if n in None:
        return b[:3]

    b[3] = np.dot(np.dot(n, a), n)
    b[4] = np.dot(np.dot(n, asq), n)

    return b

def polar_decomp(F):
    F = F.reshape(3,3)
    I = np.eye(3)
    R = F.copy()
    for j in range(20):
        R = .5 * np.dot(R, 3. * I - np.dot(R.T, R))
        if (np.amax(np.abs(np.dot(R.T, R) - I)) < 1.e-6):
            U = np.dot(R.T, F)
            return R, U
    raise RuntimeError('Fast polar decompositon failed to converge')

def sig2d(material, t, dt, temp, dtemp, f0, f, stran, d, sig, statev,
          v, sigspec):
    '''Determine the symmetric part of the velocity gradient given stress

    Parameters
    ----------

    Returns
    -------

    Approach
    --------
    Seek to determine the unknown components of the symmetric part of
    velocity gradient d[v] satisfying

                               P(d[v]) = Ppres[:]                      (1)

    where P is the current stress, d the symmetric part of the velocity
    gradient, v is a vector subscript array containing the components for
    which stresses (or stress rates) are prescribed, and Ppres[:] are the
    prescribed values at the current time.

    Solution is found iteratively in (up to) 3 steps
      1) Call newton to solve 1, return stress, statev, d if converged
      2) Call newton with d[v] = 0. to solve 1, return stress, statev, d
         if converged
      3) Call simplex with d[v] = 0. to solve 1, return stress, statev, d

    '''
    dsave = d.copy()

    d = newton(material, t, dt, temp, dtemp, f0, f, stran, d,
               sig, statev, v, sigspec)
    if d is not None:
        return d

    # --- didn't converge, try Newton's method with initial
    # --- d[v]=0.
    d = dsave.copy()
    d[v] = np.zeros(len(v))
    d = newton(material, t, dt, temp, dtemp, f0, f, stran, d,
               sig, statev, v, sigspec)
    if d is not None:
        return d

    # --- Still didn't converge. Try downhill simplex method and accept
    #     whatever answer it returns:
    d = dsave.copy()
    return simplex(material, t, dt, temp, dtemp, f0, f, stran, d,
                   sig, statev, v, sigspec)

def newton(material, t, dt, temp, dtemp, f0, farg, stran, darg,
           sigarg, statev_arg, v, sigspec):
    '''Seek to determine the unknown components of the symmetric part of velocity
    gradient d[v] satisfying

                               sig(d[v]) = sigspec

    where sig is the current stress, d the symmetric part of the velocity
    gradient, v is a vector subscript array containing the components for
    which stresses (or stress rates) are prescribed, and sigspec are the
    prescribed values at the current time.

    Parameters
    ----------
    material : instance
        constiutive model instance
    dt : float
        time step
    sig : ndarray
        stress at beginning of step
    statev_arg : ndarray
        state dependent variables at beginning of step
    v : ndarray
        vector subscript array containing the components for which
        stresses (or stress rates) are specified
    sigspec : ndarray
        Prescribed stress

    Returns
    -------
    d : ndarray || None
        If converged, the symmetric part of the velocity gradient, else None

    Notes
    -----
    The approach is an iterative scheme employing a multidimensional Newton's
    method. Each iteration begins with a call to subroutine jacobian, which
    numerically computes the Jacobian submatrix

                                  Js = J[v, v]

    where J[:,;] is the full Jacobian matrix J = dsig/deps. The value of
    d[v] is then updated according to

                d[v] = d[v] - Jsi*sigerr(d[v])/dt

    where

                   sigerr(d[v]) = sig(d[v]) - sigspec

    The process is repeated until a convergence critierion is satisfied. The
    argument converged is a flag indicat- ing whether or not the procedure
    converged:

    '''
    depsmag = lambda a: np.sqrt(sum(a[:3] ** 2) + 2. * sum(a[3:] ** 2)) * dt

    # Initialize
    eps = np.finfo(np.float).eps
    tol1, tol2 = eps, np.sqrt(eps)
    maxit1, maxit2, depsmax = 20, 30, .2

    sig = sigarg.copy()
    d = darg.copy()
    f = farg.copy()
    statev = copy(statev_arg)

    sigsave = sig.copy()
    statev_save = copy(statev)

    # --- Check if strain increment is too large
    if (depsmag(d) > depsmax):
        return None

    # update the material state to get the first guess at the new stress
    sig, statev, stif = material.eval(t, dt, temp, dtemp,
                                      f0, f, stran, d, sig, statev)
    sigerr = sig[v] - sigspec

    # --- Perform Newton iteration
    for i in range(maxit2):
        sig = sigsave.copy()
        statev = copy(statev_save)
        stif = material.eval(t, dt, temp, dtemp,
                             f0, f, stran, d, sig, statev)[2]
        if stif is None:
            # material models without an analytic jacobian send the Jacobian
            # back as None so that it is found numerically here. Likewise, we
            # find the numerical jacobian for visco materials - otherwise we
            # would have to convert the the stiffness to that corresponding to
            # the Truesdell rate, pull it back to the reference frame, apply
            # the visco correction, push it forward, and convert to Jaummann
            # rate. It's not as trivial as it sounds...
            statev = copy(statev_save)
            stif = numerical_jacobian(material, t, dt, temp, dtemp, f0,
                                      f, stran, d, sig, statev, v)
        else:
            stif = stif[[[i] for i in v], v]

        if environ.SQA:
            try:
                evals = np.linalg.eigvalsh(stif)
            except np.linalg.LinAlgError:
                raise RuntimeError('failed to determine elastic '
                                   'stiffness eigenvalues')
            else:
                if np.any(evals < 0.):
                    negevals = evals[np.where(evals < 0.)]
                    warnings.warn('negative eigen value[s] encountered '
                                  'in material Jacobian: '
                                  '{0} ({1:.2f})'.format(negevals, t))

        try:
            d[v] -= np.linalg.solve(stif, sigerr) / dt
        except LinAlgError:
            if environ.SQA:
                warnings.warn('using least squares approximation to '
                              'matrix inverse')
            d[v] -= np.linalg.lstsq(stif, sigerr)[0] / dt

        if (depsmag(d) > depsmax or  np.any(np.isnan(d)) or np.any(np.isinf(d))):
            # increment too large
            return None

        # with the updated rate of deformation, update stress and check
        sig = sigsave.copy()
        statev = copy(statev_save)
        fp, ep = update_deformation(f, d, dt, 0)
        sig, statev, stif = material.eval(t, dt, temp, dtemp,
                                          f0, fp, ep, d, sig, statev)

        sigerr = sig[v] - sigspec
        dnom = max(np.amax(np.abs(sigspec)), 1.)
        relerr = np.amax(np.abs(sigerr) / dnom)

        if i <= maxit1 and relerr < tol1:
            return d

        elif i > maxit1 and relerr < tol2:
            return d

        continue

    # didn't converge, restore restore data and exit
    return None

def simplex(material, t, dt, temp, dtemp, f0, farg, stran, darg, sigarg,
            statev_arg, v, sigspec):
    '''Perform a downhill simplex search to find sym_velgrad[v] such that

                        sig(sym_velgrad[v]) = sigspec[v]

    Parameters
    ----------
    material : instance
        constiutive model instance
    dt : float
        time step
    sig : ndarray
        stress at beginning of step
    statev_arg : ndarray
        state dependent variables at beginning of step
    v : ndarray
        vector subscript array containing the components for which
        stresses (or stress rates) are specified
    sigspec : ndarray
        Prescribed stress

    Returns
    -------
    d : ndarray
        the symmetric part of the velocity gradient

    '''
    # --- Perform the simplex search
    import scipy.optimize
    d = darg.copy()
    f = farg.copy()
    sig = sigarg.copy()
    statev = copy(statev_arg)
    args = (material, t, dt, temp, dtemp, f0, f, stran, d,
            sig, statev, v, sigspec)
    d[v] = scipy.optimize.fmin(_func, d[v], args=args, maxiter=20, disp=False)
    return d

def _func(x, material, t, dt, temp, dtemp, f0, farg, stran, darg,
          sigarg, statev_arg, v, sigspec):
    '''Objective function to be optimized by simplex

    '''
    d = darg.copy()
    f = farg.copy()
    sig = sigarg.copy()
    statev = copy(statev_arg)

    # initialize
    d[v] = x
    fp, ep = update_deformation(f, d, dt, 0)

    # store the best guesses
    sig, statev, stif = material.eval(t, dt, temp, dtemp,
                                      f0, fp, ep, d, sig, statev)

    # check the error
    error = 0.
    for i, j in enumerate(v):
        error += (sig[j] - sigspec[i]) ** 2
        continue

    return error

def numerical_jacobian(material, time, dtime, temp, dtemp, F0, F, stran, d,
                        stress, statev, v):
    '''Numerically compute material Jacobian by a centered difference scheme.

    Parameters
    ----------
    time : float
        Time at beginning of step
    dtime : float
        Time step length.  `time+dtime` is the time at the end of the step
    temp : float
        Temperature at beginning of step
    dtemp : float
        Temperature increment. `temp+dtemp` is the temperature at the end
        of the step
    F0, F : ndarray
        Deformation gradient at the beginning and end of the step
    strain : ndarray
        Strain at the beginning of the step
    d : ndarray
        Symmetric part of the velocity gradient at the middle of the step
    stress : ndarray
        Stress at the beginning of the step
    statev : ndarray
        State variables at the beginning of the step
    v : ndarray
        Array of subcomponents of Jacobian to return

    Returns
    -------
    Js : array_like
        Jacobian of the deformation J = dsig / dE

    Notes
    -----
    The submatrix returned is the one formed by the intersections of the
    rows and columns specified in the vector subscript array, v. That is,
    Js = J[v, v]. The physical array containing this submatrix is
    assumed to be dimensioned Js[nv, nv], where nv is the number of
    elements in v. Note that in the special case v = [1,2,3,4,5,6], with
    nv = 6, the matrix that is returned is the full Jacobian matrix, J.

    The components of Js are computed numerically using a centered
    differencing scheme which requires two calls to the material model
    subroutine for each element of v. The centering is about the point eps
    = epsold + d * dt, where d is the rate-of-strain array.

    History
    -------
    This subroutine is a python implementation of a routine by the same
    name in Tom Pucick's MMD driver.

    Authors
    -------
    Tom Pucick, original fortran implementation in the MMD driver
    Tim Fuller, Sandial National Laboratories, tjfulle@sandia.gov

    '''
    # local variables
    nv = len(v)
    deps =  np.sqrt(np.finfo(np.float64).eps)
    Jsub = np.zeros((nv, nv))
    dtime = 1 if dtime < 1.e-12 else dtime

    for i in range(nv):
        # perturb forward
        Dp = d.copy()
        Dp[v[i]] = d[v[i]] + (deps / dtime) / 2.
        Fp, Ep = update_deformation(F, Dp, dtime, 0)
        sigp = stress.copy()
        xp = copy(statev)
        sigp = material.eval(time, dtime, temp, dtemp,
                             F0, Fp, Ep, Dp, sigp, xp)[0]

        # perturb backward
        Dm = d.copy()
        Dm[v[i]] = d[v[i]] - (deps / dtime) / 2.
        Fm, Em = update_deformation(F, Dm, dtime, 0)
        sigm = stress.copy()
        xm = copy(statev)
        sigm = material.eval(time, dtime, temp, dtemp,
                             F0, Fm, Em, Dm, sigm, xm)[0]

        # compute component of jacobian
        Jsub[i, :] = (sigp[v] - sigm[v]) / deps

        continue

    return Jsub
