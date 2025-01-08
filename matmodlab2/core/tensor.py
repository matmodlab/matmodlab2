"""tensor.py

Constants and functions having to deal with 3D symmetric and nonsymmetric
second order tensors fourth-order tensors with minor and major symmetries.

Since there are a limited number of shapes allowed for input arrays, many of
the functions are hard coded for speed (no need to call a generalized inverse
function, when the inverse of a 3x3 can be hard coded easily.)

All of the functions are written with the following assumptions:

o Symmetric second-order tensors are stored as arrays of length 6 with the
  following component ordering

  [XX, YY, ZZ, XY, YZ, XZ]

o Nonsymmetric second-order tensors are stored as arrays of length 9 with the
  following component ordering

  [XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ]

  or

  [[XX, XY, XZ], [YX, YY, YZ], [ZX, ZY, ZZ]]

o Fourth-order tensors are stored as 6x6 matrices using the same component
  transformations as second-order symmetric tensors

"""
import numpy as np
import matmodlab2.core.linalg as la

# Fourth-order "identities"
# II1: I[i,j] I[k,l]
II1 = np.array(
    [
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
).reshape((6, 6))
# II2: I[i,k] I[j,l]
II2 = np.array(
    [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
).reshape((6, 6))
# II3: I[i,l] I[j,k]
II3 = np.array(
    [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
).reshape((6, 6))
II4 = (II2 + II3) / 2
# II5 = (I[i,k] I[j,l] + I[i,l] I[j,k]) / 2
II5 = np.array(
    [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5,
    ]
).reshape((6, 6))

# Second-order identities
Z6 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
I6 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
I9 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
I3x3 = np.eye(3)

SYMMETRIC_COMPONENTS = ["XX", "YY", "ZZ", "XY", "YZ", "XZ"]
TENSOR_COMPONENTS = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]
VOIGT = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

epsilon = np.finfo(float).eps


def has_valid_shape(A):
    return is_valid_shape(A.shape)


def is_valid_shape(shape):
    """Return whether the shape is valid for these set of functions"""
    return shape in ((6,), (9,), (3, 3), (6, 6))


def identity(n):
    """Return an identity tensor according to n"""
    if n not in (3, 6, 9):
        raise ValueError("Unknown identity size {0}".format(n))
    if n == 3:
        return np.eye(3)
    if n == 6:
        return np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    if n == 9:
        return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])


def identity_like(A):
    """Return an identity matrix like A"""
    A = np.asarray(A)
    assert has_valid_shape(A)
    return identity(A.shape[0])


def trace(A, metric=None):
    """Return trace of A"""
    A = np.asarray(A)
    assert has_valid_shape(A)
    if metric is None:
        metric = identity_like(A)
        X = np.array(metric)
    else:
        X = inv(metric)
    return double_dot(A, X)


def isotropic_part(A, metric=None):
    """Return isotropic part of A"""
    A = np.asarray(A)
    assert has_valid_shape(A)
    if metric is None:
        metric = identity_like(A)
        X = np.array(metric)
    else:
        X = inv(metric)
    return trace(A, metric=metric) / 3.0 * X


def deviatoric_part(A, metric=None):
    """Return deviatoric part of A"""
    return A - isotropic_part(A, metric=metric)


def is_symmetric_rep(A):
    return A.shape == (6,)


def symmetric_dyad(A, B):
    """Compute the symmetric dyad AB_ij = A_i B_j"""
    A = np.asarray(A)
    B = np.asarray(B)
    assert A.shape == (3,)
    assert A.shape == B.shape
    return np.array(
        [A[0] * B[0], A[1] * B[1], A[2] * B[2], A[0] * B[1], A[1] * B[2], A[0] * B[2]]
    )


def root_j2(A):
    """Return the square root of the second invariant of the
    deviatoric part of the matrix.
    """
    return magnitude(deviatoric_part(A)) / np.sqrt(2.0)


def invariants(A, type=None, n=None):
    """Return the invariants of a tensor A

    The type parameter is one of 'default' (None), 'mechanics', 'lode',
    'directed'. Multiple types of invariants can be returned by joining
    different types with &, ie, to get the mechanics and lode invariants do
    type='lode&mechanics'

    """
    A = np.asarray(A)
    assert has_valid_shape(A)
    anyin = lambda a, b: any([x in b for x in a])
    valid_types = ("default", "mechanics", "lode", "directed")

    if type is None:
        type = "default"
    types = [x.strip() for x in type.split("&") if x.split()]
    assert [x in valid_types for x in types]

    if "directed" in types and n in None:
        raise ValueError("type=directed requires n be defined")

    dikt = {}
    if anyin(("mechanics", "lode"), types):
        dikt["i1"] = trace(A)
        dikt["rootj2"] = magnitude(deviatoric_part(A)) / np.sqrt(2.0)
        dikt["j3"] = det(deviatoric_part(A))

    if "lode" in types:
        dikt["r"] = np.sqrt(2.0) * dikt["rootj2"]
        dikt["z"] = dikt["i1"] / np.sqrt(3.0)
        if abs(dikt["rootj2"]) < epsilon:
            dikt["lode"] = 0.0
        else:
            dikt["lode"] = dikt["j3"] / 2.0 * 3.0**1.5 / dikt["rootj2"] ** 3.0
        dikt["theta"] = np.arcsin(max(-1.0, min(dikt["lode"], 1.0))) / 3.0

    if anyin(("default", "directed"), types):
        A = matrix_rep(A, 0)
        asq = np.dot(A, A)
        deta = la.det(A)
        tra = trace(A)

        dikt["i1"] = tra
        dikt["i2"] = 0.5 * (tra**2 - trace(asq))
        dikt["i3"] = deta

    if "directed" in types:
        dikt["i4"] = np.dot(np.dot(n, A), n)
        dikt["i4"] = np.dot(np.dot(n, asq), n)

    if len(types) > 1:
        # For composite types, just return the dictionary
        return dikt

    if types[0] == "default":
        return dikt["i1"], dikt["i2"], dikt["i3"]

    if types[0] == "directed":
        return dikt["i1"], dikt["i2"], dikt["i3"], dikt["i4"], dikt["i5"]

    if types[0] == "mechanics":
        return dikt["i1"], dikt["rootj2"], dikt["j3"]

    if types[0] == "lode":
        return dikt["z"], dikt["r"], dikt["theta"], dikt["lode"]

    return None


def magnitude(A):
    """Return magnitude of A"""
    return np.sqrt(double_dot(A, A))


def dot(A, B):
    """Dot product of A and B"""
    A, B = np.asarray(A), np.asarray(B)
    assert has_valid_shape(A)
    assert has_valid_shape(B)
    if A.shape == (6,) and B.shape == (6,):
        return A * B * VOIGT
    if A.shape == (6,) and B.shape == (3, 3):
        return np.dot(matrix_rep(A, 0), B)
    if A.shape == (3, 3) and B.shape == (6,):
        return np.dot(A, matrix_rep(B, 0))
    if A.shape == (3, 3) and B.shape == (3, 3):
        return np.dot(A, B)
    if A.shape == (6, 6) and B.shape == (6,):
        return np.dot(A, B)
    if A.shape == (6,) and B.shape == (6, 6):
        return np.dot(A, B)
    if A.shape == (9,) and B.shape == (9,):
        return np.dot(A.reshape(3, 3), B.reshape(3, 3)).flatten()
    print(A.shape, B.shape)
    raise ValueError("Unknown dot combination")


def double_dot(A, B):
    """Return A:B"""
    A, B = np.asarray(A), np.asarray(B)
    assert has_valid_shape(A)
    assert has_valid_shape(B)
    if A.shape == (6, 6) and B.shape == (6,):
        return np.dot(A, B)
    if A.shape == (6,) and B.shape == (6, 6):
        return np.dot(A, B)
    A, B = A.reshape(-1), B.reshape(-1)
    if B.shape == (6,) and A.shape == (9,):
        A, B = B, A
    if A.shape == (6,) and B.shape == (6,):
        X = (
            A[0] * B[0]
            + A[3] * B[3]
            + A[5] * B[5]
            + A[3] * B[3]
            + A[1] * B[1]
            + A[4] * B[4]
            + A[5] * B[5]
            + A[4] * B[4]
            + A[2] * B[2]
        )
    elif A.shape == (6,) and B.shape == (9,):
        X = (
            A[0] * B[0]
            + A[3] * B[1]
            + A[5] * B[2]
            + A[3] * B[3]
            + A[1] * B[4]
            + A[4] * B[5]
            + A[5] * B[6]
            + A[4] * B[7]
            + A[2] * B[8]
        )
    elif A.shape == (9,) and B.shape == (9,):
        X = (
            A[0] * B[0]
            + A[1] * B[1]
            + A[2] * B[2]
            + A[3] * B[3]
            + A[4] * B[4]
            + A[5] * B[5]
            + A[6] * B[6]
            + A[7] * B[7]
            + A[8] * B[8]
        )
    return X


def det(A):
    """Computes the determininant of A"""
    A = np.asarray(A)
    assert has_valid_shape(A)
    if A.shape == (6,):
        X = (
            A[0] * A[1] * A[2]
            - A[0] * A[4] ** 2
            - A[1] * A[5] ** 2
            - A[2] * A[3] ** 2
            + 2 * A[3] * A[4] * A[5]
        )
    elif A.shape in ((3, 3), (9,)):
        A = A.reshape(-1)
        X = (
            A[0] * A[4] * A[8]
            - A[0] * A[5] * A[7]
            - A[1] * A[3] * A[8]
            + A[1] * A[5] * A[6]
            + A[2] * A[3] * A[7]
            - A[2] * A[4] * A[6]
        )
    else:
        raise ValueError("Unknown shape")
    return X


def inv(A):
    """Computes the inverse of A"""
    A = np.asarray(A)
    assert has_valid_shape(A)
    orig_shape = A.shape
    if A.shape == (3, 3):
        if is_symmetric(A):
            A = array_rep(A, (6,))
        else:
            A = A.reshape(-1)
    if A.shape == (6,):
        X = np.zeros(6)
        X[0] = (
            -(A[0] * A[1] - A[3] ** 2)
            * (
                -A[3] * (A[4] - A[3] * A[5] / A[0]) / (A[0] * (A[1] - A[3] ** 2 / A[0]))
                + A[5] / A[0]
            )
            * (
                A[3] * (A[4] - A[3] * A[5] / A[0]) / (A[0] * (A[1] - A[3] ** 2 / A[0]))
                - A[5] / A[0]
            )
            / (
                A[0] * A[1] * A[2]
                - A[0] * A[4] ** 2
                - A[1] * A[5] ** 2
                - A[2] * A[3] ** 2
                + 2 * A[3] * A[4] * A[5]
            )
            + 1 / A[0]
            + A[3] ** 2 / (A[0] ** 2 * (A[1] - A[3] ** 2 / A[0]))
        )
        X[1] = 1 / (A[1] - A[3] ** 2 / A[0]) + (A[4] - A[3] * A[5] / A[0]) ** 2 * (
            A[0] * A[1] - A[3] ** 2
        ) / (
            (A[1] - A[3] ** 2 / A[0]) ** 2
            * (
                A[0] * A[1] * A[2]
                - A[0] * A[4] ** 2
                - A[1] * A[5] ** 2
                - A[2] * A[3] ** 2
                + 2 * A[3] * A[4] * A[5]
            )
        )
        X[2] = (A[0] * A[1] - A[3] ** 2) / (
            A[0] * A[1] * A[2]
            - A[0] * A[4] ** 2
            - A[1] * A[5] ** 2
            - A[2] * A[3] ** 2
            + 2 * A[3] * A[4] * A[5]
        )
        X[3] = (A[4] - A[3] * A[5] / A[0]) * (A[0] * A[1] - A[3] ** 2) * (
            -A[3] * (A[4] - A[3] * A[5] / A[0]) / (A[0] * (A[1] - A[3] ** 2 / A[0]))
            + A[5] / A[0]
        ) / (
            (A[1] - A[3] ** 2 / A[0])
            * (
                A[0] * A[1] * A[2]
                - A[0] * A[4] ** 2
                - A[1] * A[5] ** 2
                - A[2] * A[3] ** 2
                + 2 * A[3] * A[4] * A[5]
            )
        ) - A[
            3
        ] / (
            A[0] * (A[1] - A[3] ** 2 / A[0])
        )
        X[4] = (
            -(A[4] - A[3] * A[5] / A[0])
            * (A[0] * A[1] - A[3] ** 2)
            / (
                (A[1] - A[3] ** 2 / A[0])
                * (
                    A[0] * A[1] * A[2]
                    - A[0] * A[4] ** 2
                    - A[1] * A[5] ** 2
                    - A[2] * A[3] ** 2
                    + 2 * A[3] * A[4] * A[5]
                )
            )
        )
        X[5] = (
            -(A[0] * A[1] - A[3] ** 2)
            * (
                -A[3] * (A[4] - A[3] * A[5] / A[0]) / (A[0] * (A[1] - A[3] ** 2 / A[0]))
                + A[5] / A[0]
            )
            / (
                A[0] * A[1] * A[2]
                - A[0] * A[4] ** 2
                - A[1] * A[5] ** 2
                - A[2] * A[3] ** 2
                + 2 * A[3] * A[4] * A[5]
            )
        )
    elif A.shape == (9,):
        X = np.zeros(9)
        X[0] = (
            -(A[0] * A[4] - A[1] * A[3])
            * (
                -A[1]
                * (A[5] - A[2] * A[3] / A[0])
                / (A[0] * (A[4] - A[1] * A[3] / A[0]))
                + A[2] / A[0]
            )
            * (
                A[3]
                * (A[7] - A[1] * A[6] / A[0])
                / (A[0] * (A[4] - A[1] * A[3] / A[0]))
                - A[6] / A[0]
            )
            / (
                A[0] * A[4] * A[8]
                - A[0] * A[5] * A[7]
                - A[1] * A[3] * A[8]
                + A[1] * A[5] * A[6]
                + A[2] * A[3] * A[7]
                - A[2] * A[4] * A[6]
            )
            + 1 / A[0]
            + A[1] * A[3] / (A[0] ** 2 * (A[4] - A[1] * A[3] / A[0]))
        )
        X[1] = (A[7] - A[1] * A[6] / A[0]) * (A[0] * A[4] - A[1] * A[3]) * (
            -A[1] * (A[5] - A[2] * A[3] / A[0]) / (A[0] * (A[4] - A[1] * A[3] / A[0]))
            + A[2] / A[0]
        ) / (
            (A[4] - A[1] * A[3] / A[0])
            * (
                A[0] * A[4] * A[8]
                - A[0] * A[5] * A[7]
                - A[1] * A[3] * A[8]
                + A[1] * A[5] * A[6]
                + A[2] * A[3] * A[7]
                - A[2] * A[4] * A[6]
            )
        ) - A[
            1
        ] / (
            A[0] * (A[4] - A[1] * A[3] / A[0])
        )
        X[2] = (
            -(A[0] * A[4] - A[1] * A[3])
            * (
                -A[1]
                * (A[5] - A[2] * A[3] / A[0])
                / (A[0] * (A[4] - A[1] * A[3] / A[0]))
                + A[2] / A[0]
            )
            / (
                A[0] * A[4] * A[8]
                - A[0] * A[5] * A[7]
                - A[1] * A[3] * A[8]
                + A[1] * A[5] * A[6]
                + A[2] * A[3] * A[7]
                - A[2] * A[4] * A[6]
            )
        )
        X[3] = -(A[5] - A[2] * A[3] / A[0]) * (A[0] * A[4] - A[1] * A[3]) * (
            A[3] * (A[7] - A[1] * A[6] / A[0]) / (A[0] * (A[4] - A[1] * A[3] / A[0]))
            - A[6] / A[0]
        ) / (
            (A[4] - A[1] * A[3] / A[0])
            * (
                A[0] * A[4] * A[8]
                - A[0] * A[5] * A[7]
                - A[1] * A[3] * A[8]
                + A[1] * A[5] * A[6]
                + A[2] * A[3] * A[7]
                - A[2] * A[4] * A[6]
            )
        ) - A[
            3
        ] / (
            A[0] * (A[4] - A[1] * A[3] / A[0])
        )
        X[4] = 1 / (A[4] - A[1] * A[3] / A[0]) + (A[5] - A[2] * A[3] / A[0]) * (
            A[7] - A[1] * A[6] / A[0]
        ) * (A[0] * A[4] - A[1] * A[3]) / (
            (A[4] - A[1] * A[3] / A[0]) ** 2
            * (
                A[0] * A[4] * A[8]
                - A[0] * A[5] * A[7]
                - A[1] * A[3] * A[8]
                + A[1] * A[5] * A[6]
                + A[2] * A[3] * A[7]
                - A[2] * A[4] * A[6]
            )
        )
        X[5] = (
            -(A[5] - A[2] * A[3] / A[0])
            * (A[0] * A[4] - A[1] * A[3])
            / (
                (A[4] - A[1] * A[3] / A[0])
                * (
                    A[0] * A[4] * A[8]
                    - A[0] * A[5] * A[7]
                    - A[1] * A[3] * A[8]
                    + A[1] * A[5] * A[6]
                    + A[2] * A[3] * A[7]
                    - A[2] * A[4] * A[6]
                )
            )
        )
        X[6] = (
            (A[0] * A[4] - A[1] * A[3])
            * (
                A[3]
                * (A[7] - A[1] * A[6] / A[0])
                / (A[0] * (A[4] - A[1] * A[3] / A[0]))
                - A[6] / A[0]
            )
            / (
                A[0] * A[4] * A[8]
                - A[0] * A[5] * A[7]
                - A[1] * A[3] * A[8]
                + A[1] * A[5] * A[6]
                + A[2] * A[3] * A[7]
                - A[2] * A[4] * A[6]
            )
        )
        X[7] = (
            -(A[7] - A[1] * A[6] / A[0])
            * (A[0] * A[4] - A[1] * A[3])
            / (
                (A[4] - A[1] * A[3] / A[0])
                * (
                    A[0] * A[4] * A[8]
                    - A[0] * A[5] * A[7]
                    - A[1] * A[3] * A[8]
                    + A[1] * A[5] * A[6]
                    + A[2] * A[3] * A[7]
                    - A[2] * A[4] * A[6]
                )
            )
        )
        X[8] = (A[0] * A[4] - A[1] * A[3]) / (
            A[0] * A[4] * A[8]
            - A[0] * A[5] * A[7]
            - A[1] * A[3] * A[8]
            + A[1] * A[5] * A[6]
            + A[2] * A[3] * A[7]
            - A[2] * A[4] * A[6]
        )
    else:
        raise ValueError("Unknown shape")

    if X.shape == orig_shape:
        return X

    return matrix_rep(X, 0)


def expm(A):
    """Compute the matrix exponential of a 3x3 matrix"""
    mat, orig_shape = matrix_rep(A)
    mat2 = la.expm(mat)
    return array_rep(mat2, orig_shape)


def logm(A):
    """Compute the matrix logarithm of a 3x3 matrix"""
    mat, orig_shape = matrix_rep(A)
    mat2 = la.logm(mat)
    return array_rep(mat2, orig_shape)


def powm(A, t):
    """Compute the matrix power of a 3x3 matrix"""
    mat, orig_shape = matrix_rep(A)
    mat2 = la.powm(mat)
    return array_rep(mat2, orig_shape)


def sqrtm(A):
    """Compute the square root of a 3x3 matrix"""
    mat, orig_shape = matrix_rep(A)
    mat2 = la.sqrtm(mat)
    return array_rep(mat2, orig_shape)


def matrix_rep(A, disp=1):
    """Convert array to matrix"""
    A = np.asarray(A)
    assert has_valid_shape(A)
    orig_shape = A.shape
    if orig_shape == (6,):
        ix1 = ([0, 1, 2, 0, 1, 0, 1, 2, 2], [0, 1, 2, 1, 2, 2, 0, 0, 1])
        ix2 = [0, 1, 2, 3, 4, 5, 3, 5, 4]
        mat = np.zeros((3, 3))
        mat[ix1] = A[ix2]
    elif orig_shape == (9,):
        mat = np.reshape(A, (3, 3))
    elif orig_shape == (3, 3):
        mat = np.array(A)
    else:
        raise ValueError("Unknown shape")
    if not disp:
        return mat
    return mat, orig_shape


def array_rep(mat, shape):
    """Reverse of matrix_rep"""
    mat = np.asarray(mat)
    if mat.shape == (6,):
        return mat
    if shape == (6,):
        mat = 0.5 * (mat + mat.T)
        return mat[([0, 1, 2, 0, 1, 0], [0, 1, 2, 1, 2, 2])]
    if shape == (9,):
        return np.ndarray.flatten(mat)
    if shape == (3, 3):
        return np.array(mat)
    raise ValueError("Unknown shape")


def symmetric_part(A):
    """Symmetric part of A"""
    A = np.asarray(A)
    assert A.shape in ((6,), (9,), (3, 3))
    if A.shape == (6,):
        return A
    elif A.shape == (9,):
        A = A.reshape((3, 3))
    symA = 0.5 * (A + A.T)
    return symA[([0, 1, 2, 0, 1, 0], [0, 1, 2, 1, 2, 2])]


def isdiag(A):
    """Determines if a matrix is diagonal."""
    A = np.asarray(A)
    assert has_valid_shape(A)
    if A.shape == (6.0):
        return np.all(np.abs(A[3:]) <= epsilon)
    elif A.shape == (9,):
        return np.all(np.abs(A[[0, 4, 8]]) <= epsilon)
    elif A.shape == (3, 3):
        return np.all(np.abs(A[([0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1])]) <= epsilon)
    raise ValueError("Unknown shape")


def symsq(F):
    """Computes dot(F.T, F)"""
    X = np.zeros(6)
    F = np.asarray(F).reshape(-1)
    assert F.shape == (9,)
    X[0] = F[0] ** 2 + F[3] ** 2 + F[6] ** 2
    X[1] = F[1] ** 2 + F[4] ** 2 + F[7] ** 2
    X[2] = F[2] ** 2 + F[5] ** 2 + F[8] ** 2
    X[3] = F[0] * F[1] + F[3] * F[4] + F[6] * F[7]
    X[4] = F[1] * F[2] + F[4] * F[5] + F[7] * F[8]
    X[5] = F[0] * F[2] + F[3] * F[5] + F[6] * F[8]
    return X


def unrotate(R, A):
    return push(R.T, A)


def rotate(R, A):
    return push(R, A)


def pull(F, A):
    return push(inv(F), A)


def push(F, A):
    """Computes the push operation F A F.T / J"""
    F = np.asarray(F).reshape(-1)
    assert F.shape == (9,)
    A = np.asarray(A)
    assert A.shape in ((3, 3), (6,), (6, 6))
    if A.shape == (3, 3):
        assert is_symmetric(A)
        A = array_rep(A, (6,))
    if A.shape == (6,):
        return push6(F, A)
    elif A.shape == (6, 6):
        return push66(F, A)
    raise ValueError("Unknown shape")


def push6(F, A):
    X = np.zeros(6)
    X[0] = (
        F[0] * (A[0] * F[0] + A[3] * F[1] + A[5] * F[2])
        + F[1] * (A[1] * F[1] + A[3] * F[0] + A[4] * F[2])
        + F[2] * (A[2] * F[2] + A[4] * F[1] + A[5] * F[0])
    )
    X[1] = (
        F[3] * (A[0] * F[3] + A[3] * F[4] + A[5] * F[5])
        + F[4] * (A[1] * F[4] + A[3] * F[3] + A[4] * F[5])
        + F[5] * (A[2] * F[5] + A[4] * F[4] + A[5] * F[3])
    )
    X[2] = (
        F[6] * (A[0] * F[6] + A[3] * F[7] + A[5] * F[8])
        + F[7] * (A[1] * F[7] + A[3] * F[6] + A[4] * F[8])
        + F[8] * (A[2] * F[8] + A[4] * F[7] + A[5] * F[6])
    )
    X[3] = (
        F[3] * (A[0] * F[0] + A[3] * F[1] + A[5] * F[2])
        + F[4] * (A[1] * F[1] + A[3] * F[0] + A[4] * F[2])
        + F[5] * (A[2] * F[2] + A[4] * F[1] + A[5] * F[0])
    )
    X[4] = (
        F[6] * (A[0] * F[3] + A[3] * F[4] + A[5] * F[5])
        + F[7] * (A[1] * F[4] + A[3] * F[3] + A[4] * F[5])
        + F[8] * (A[2] * F[5] + A[4] * F[4] + A[5] * F[3])
    )
    X[5] = (
        F[6] * (A[0] * F[0] + A[3] * F[1] + A[5] * F[2])
        + F[7] * (A[1] * F[1] + A[3] * F[0] + A[4] * F[2])
        + F[8] * (A[2] * F[2] + A[4] * F[1] + A[5] * F[0])
    )
    return X / det(F)


def push66(F, A):
    Q = symleaf(F)
    X = np.dot(np.dot(Q, A), Q.T)
    return X / det(F)


def polar_decomp(F):
    return la.polar_decomp(F)


def is_symmetric(A):
    A = np.asarray(A)
    if A.shape == (6,):
        return True
    if A.shape == (9,):
        A = A.reshape((3, 3))
    return np.allclose([A[0, 1], A[1, 2], A[0, 2]], [A[1, 0], A[2, 1], A[2, 0]])


def dyad(A, B):
    """Computes the outer product of A and B"""
    A, B = np.asarray(A), np.asarray(B)
    assert has_valid_shape(A)
    assert has_valid_shape(B)
    if A.shape == (3, 3) and is_symmetric(A):
        A = array_rep(A, (6,))
    if B.shape == (3, 3) and is_symmetric(B):
        B = array_rep(B, (6,))
    if A.shape == (6,) and B.shape == (6,):
        X = np.zeros((6, 6))
        X[0, 0] = A[0] * B[0]
        X[0, 1] = A[0] * B[1]
        X[0, 2] = A[0] * B[2]
        X[0, 3] = A[0] * B[3]
        X[0, 4] = A[0] * B[4]
        X[0, 5] = A[0] * B[5]
        X[1, 0] = A[1] * B[0]
        X[1, 1] = A[1] * B[1]
        X[1, 2] = A[1] * B[2]
        X[1, 3] = A[1] * B[3]
        X[1, 4] = A[1] * B[4]
        X[1, 5] = A[1] * B[5]
        X[2, 0] = A[2] * B[0]
        X[2, 1] = A[2] * B[1]
        X[2, 2] = A[2] * B[2]
        X[2, 3] = A[2] * B[3]
        X[2, 4] = A[2] * B[4]
        X[2, 5] = A[2] * B[5]
        X[3, 0] = A[3] * B[0]
        X[3, 1] = A[3] * B[1]
        X[3, 2] = A[3] * B[2]
        X[3, 3] = A[3] * B[3]
        X[3, 4] = A[3] * B[4]
        X[3, 5] = A[3] * B[5]
        X[4, 0] = A[4] * B[0]
        X[4, 1] = A[4] * B[1]
        X[4, 2] = A[4] * B[2]
        X[4, 3] = A[4] * B[3]
        X[4, 4] = A[4] * B[4]
        X[4, 5] = A[4] * B[5]
        X[5, 0] = A[5] * B[0]
        X[5, 1] = A[5] * B[1]
        X[5, 2] = A[5] * B[2]
        X[5, 3] = A[5] * B[3]
        X[5, 4] = A[5] * B[4]
        X[5, 5] = A[5] * B[5]
    elif A.shape == (3,) and B.shape == (3,):
        X = np.zeros(6)
        X[0] = A[0] * B[0]
        X[1] = A[1] * B[1]
        X[2] = A[2] * B[2]
        X[3] = A[0] * B[1]
        X[4] = A[1] * B[2]
        X[5] = A[0] * B[2]
    else:
        raise ValueError("Unknown shape")
    return X


def symshuffle(A, B):
    """Computes the product Xijkl = .5 (Aik Bjl + Ail Bjk)"""
    A, B = np.asarray(A), np.asarray(B)
    assert has_valid_shape(A)
    assert has_valid_shape(B)
    if A.shape == (3, 3) and is_symmetric(A):
        A = array_rep(A, (6,))
    if B.shape == (3, 3) and is_symmetric(B):
        B = array_rep(B, (6,))
    X = np.zeros((6, 6))
    if A.shape == (6,) and B.shape == (6,):
        X[0, 0] = (A[0] * B[0] + A[0] * B[0]) / 2.0
        X[0, 1] = (A[3] * B[3] + A[3] * B[3]) / 2.0
        X[0, 2] = (A[5] * B[5] + A[5] * B[5]) / 2.0
        X[0, 3] = (A[0] * B[3] + A[3] * B[0]) / 2.0
        X[0, 4] = (A[3] * B[5] + A[5] * B[3]) / 2.0
        X[0, 5] = (A[0] * B[5] + A[5] * B[0]) / 2.0
        X[1, 0] = (A[3] * B[3] + A[3] * B[3]) / 2.0
        X[1, 1] = (A[1] * B[1] + A[1] * B[1]) / 2.0
        X[1, 2] = (A[4] * B[4] + A[4] * B[4]) / 2.0
        X[1, 3] = (A[3] * B[1] + A[1] * B[3]) / 2.0
        X[1, 4] = (A[1] * B[4] + A[4] * B[1]) / 2.0
        X[1, 5] = (A[3] * B[4] + A[4] * B[3]) / 2.0
        X[2, 0] = (A[5] * B[5] + A[5] * B[5]) / 2.0
        X[2, 1] = (A[4] * B[4] + A[4] * B[4]) / 2.0
        X[2, 2] = (A[2] * B[2] + A[2] * B[2]) / 2.0
        X[2, 3] = (A[5] * B[4] + A[4] * B[5]) / 2.0
        X[2, 4] = (A[4] * B[2] + A[2] * B[4]) / 2.0
        X[2, 5] = (A[5] * B[2] + A[2] * B[5]) / 2.0
        X[3, 0] = (A[0] * B[3] + A[0] * B[3]) / 2.0
        X[3, 1] = (A[3] * B[1] + A[3] * B[1]) / 2.0
        X[3, 2] = (A[5] * B[4] + A[5] * B[4]) / 2.0
        X[3, 3] = (A[0] * B[1] + A[3] * B[3]) / 2.0
        X[3, 4] = (A[3] * B[4] + A[5] * B[1]) / 2.0
        X[3, 5] = (A[0] * B[4] + A[5] * B[3]) / 2.0
        X[4, 0] = (A[3] * B[5] + A[3] * B[5]) / 2.0
        X[4, 1] = (A[1] * B[4] + A[1] * B[4]) / 2.0
        X[4, 2] = (A[4] * B[2] + A[4] * B[2]) / 2.0
        X[4, 3] = (A[3] * B[4] + A[1] * B[5]) / 2.0
        X[4, 4] = (A[1] * B[2] + A[4] * B[4]) / 2.0
        X[4, 5] = (A[3] * B[2] + A[4] * B[5]) / 2.0
        X[5, 0] = (A[0] * B[5] + A[0] * B[5]) / 2.0
        X[5, 1] = (A[3] * B[4] + A[3] * B[4]) / 2.0
        X[5, 2] = (A[5] * B[2] + A[5] * B[2]) / 2.0
        X[5, 3] = (A[0] * B[4] + A[3] * B[5]) / 2.0
        X[5, 4] = (A[3] * B[2] + A[5] * B[4]) / 2.0
        X[5, 5] = (A[0] * B[2] + A[5] * B[5]) / 2.0
    else:
        raise ValueError("Unknown shape")
    return X


def symleaf(F):
    """COMPUTE A 6X6 MANDEL MATRIX THAT IS THE SYM-LEAF TRANSFORMATION OF THE
    INPUT 3X3 MATRIX F.

    Parameters
    ----------
    F : ANY 3X3 MATRIX (IN CONVENTIONAL 3X3 STORAGE)

    Returns
    -------
    X : 6X6 MANDEL MATRIX FOR THE SYM-LEAF TRANSFORMATION MATRIX

    Notes
    -----
    IF A IS ANY SYMMETRIC TENSOR, AND IF {A} IS ITS 6X1 MANDEL ARRAY, THEN THE
    6X1 MANDEL ARRAY FOR THE TENSOR B=F.A.TRANSPOSE[F] MAY BE COMPUTED BY

                          {B}=[FF]{A}

    IF F IS A DEFORMATION F, THEN B IS THE "PUSH" (SPATIAL) TRANSFORMATION OF
    THE REFERENCE TENSOR A IF F IS Inverse[F], THEN B IS THE "PULL"
    (REFERENCE) TRANSFORMATION OF THE SPATIAL TENSOR A, AND THEREFORE B WOULD
    BE Inverse[FF]{A}.

    IF F IS A ROTATION, THEN B IS THE ROTATION OF A, AND
    FF WOULD BE BE A 6X6 ORTHOGONAL MATRIX, JUST AS IS F

    """
    F = np.asarray(F).reshape(-1)
    assert F.shape == (9,)
    X = np.zeros((6, 6))
    X[0, 0] = F[0] * F[0]
    X[0, 1] = F[1] * F[1]
    X[0, 2] = F[2] * F[2]
    X[0, 3] = F[0] * F[1] + F[1] * F[0]
    X[0, 4] = F[1] * F[2] + F[2] * F[1]
    X[0, 5] = F[0] * F[2] + F[2] * F[0]
    X[1, 0] = F[3] * F[3]
    X[1, 1] = F[4] * F[4]
    X[1, 2] = F[5] * F[5]
    X[1, 3] = F[3] * F[4] + F[4] * F[3]
    X[1, 4] = F[4] * F[5] + F[5] * F[4]
    X[1, 5] = F[3] * F[5] + F[5] * F[3]
    X[2, 0] = F[6] * F[6]
    X[2, 1] = F[7] * F[7]
    X[2, 2] = F[8] * F[8]
    X[2, 3] = F[6] * F[7] + F[7] * F[6]
    X[2, 4] = F[7] * F[8] + F[8] * F[7]
    X[2, 5] = F[6] * F[8] + F[8] * F[6]
    X[3, 0] = F[0] * F[3] + F[3] * F[0]
    X[3, 1] = F[1] * F[4] + F[4] * F[1]
    X[3, 2] = F[2] * F[5] + F[5] * F[2]
    X[3, 3] = F[0] * F[4] + F[1] * F[3]
    X[3, 4] = F[1] * F[5] + F[2] * F[4]
    X[3, 5] = F[0] * F[5] + F[2] * F[3]
    X[4, 0] = F[3] * F[6] + F[6] * F[3]
    X[4, 1] = F[4] * F[7] + F[7] * F[4]
    X[4, 2] = F[5] * F[8] + F[8] * F[5]
    X[4, 3] = F[3] * F[7] + F[4] * F[6]
    X[4, 4] = F[4] * F[8] + F[5] * F[7]
    X[4, 5] = F[3] * F[8] + F[5] * F[6]
    X[5, 0] = F[0] * F[6] + F[6] * F[0]
    X[5, 1] = F[1] * F[7] + F[7] * F[1]
    X[5, 2] = F[2] * F[8] + F[8] * F[2]
    X[5, 3] = F[0] * F[7] + F[1] * F[6]
    X[5, 4] = F[1] * F[8] + F[2] * F[7]
    X[5, 5] = F[0] * F[8] + F[2] * F[6]
    return X
