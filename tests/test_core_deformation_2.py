import numpy as np
import pytest
import matmodlab2 as mml

tol=1.0e-12

def val_is_close(val1, val2, tol):
    if np.isclose(val1, val2, atol=tol, rtol=tol):
        return True
    print("val1:\n{0}".format(repr(val1)))
    print("val2:\n{0}".format(repr(val2)))
    print("diff:\n{0}".format(repr(val1 - val2)))
    return False


def arr_is_close(arr1, arr2, tol):
    if np.allclose(arr1, arr2, atol=tol, rtol=tol):
        return True
    print("arr1:\n{0}".format(repr(arr1)))
    print("arr2:\n{0}".format(repr(arr2)))
    print("diff:\n{0}".format(repr(arr1 - arr2)))
    return False


@pytest.mark.unit
@pytest.mark.parametrize('kappa', np.linspace(-2.0, 2.0, 9))
def test_strain_from_stretch_1(kappa):

    # Setup
    components = np.eye(3).flatten()

    # Test
    comp = mml.core.deformation.strain_from_stretch(components, kappa)
    assert arr_is_close(comp, np.zeros(9), tol)  # No strain

    # Teardown
    pass


@pytest.mark.unit
@pytest.mark.parametrize('kappa', np.linspace(-2.0, 2.0, 9))
@pytest.mark.parametrize('stretch', np.linspace(0.5, 1.5, 9))
def test_strain_from_stretch_2(kappa, stretch):

    # Setup
    components = np.eye(3).flatten()
    components[0] = stretch

    if kappa == 0.0:
        strain = np.log(stretch)
    else:
        strain = (stretch ** kappa - 1.0) / kappa

    # Test
    comp = mml.core.deformation.strain_from_stretch(components, kappa)

    assert np.isclose(comp[0], strain, atol=tol, rtol=tol)
    assert np.allclose(comp[1:], np.zeros(8), atol=tol, rtol=tol)  # No strain

    # Teardown
    pass




@pytest.mark.unit
@pytest.mark.parametrize('kappa', [2.0, 1.0, 0.0, -1.0, -2.0])
def test_strain_from_defgrad_1(kappa):

    # Setup
    components = np.eye(3).flatten()

    # Test
    comp, rot = mml.core.deformation.strain_from_defgrad(components, kappa)
    print(components)
    print(comp, rot)

    assert np.allclose(rot, np.eye(3), atol=tol, rtol=tol)  # No rotation
    assert np.allclose(comp, np.zeros(6), atol=tol, rtol=tol)  # No strain

    # Teardown
    pass


@pytest.mark.unit
@pytest.mark.parametrize('kappa', np.linspace(-2.0, 2.0, 9))
@pytest.mark.parametrize('stretch', np.linspace(0.5, 1.5, 9))
def test_strain_from_defgrad_2(kappa, stretch):

    # Setup
    components = np.eye(3).flatten()
    components[0] = stretch

    if kappa == 0.0:
        strain = np.log(stretch)
    else:
        strain = (stretch ** kappa - 1.0) / kappa

    # Test
    comp, rot = mml.core.deformation.strain_from_defgrad(components, kappa)

    assert arr_is_close(rot, np.eye(3), tol)  # No rotation
    assert val_is_close(comp[0], strain, tol)
    assert arr_is_close(comp[1:], np.zeros(5), tol)  # No strain

    # Teardown
    pass


@pytest.mark.unit
@pytest.mark.parametrize('kappa', np.linspace(-2.0, 2.0, 9))
@pytest.mark.parametrize('stretch', np.linspace(0.5, 1.5, 9))
def test_strain_from_defgrad_3(kappa, stretch):
    """
    Axial deformation with superimposed random rotation
    """

    # Setup

    def random_unit_vector():
        theta = np.arccos(1.0 - 2.0 * np.random.random())
        phi = 2.0 * np.pi * np.random.random()
        return np.array([np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta)])

    def random_rotation_tensor():
        alpha = np.random.uniform(0.0, 2.0 * np.pi)
        veca = random_unit_vector()

        dyadaa = np.outer(veca, veca)
        axiala = np.array([[0.0, -veca[2], veca[1]],
                           [veca[2], 0.0, -veca[0]],
                           [-veca[1], veca[0], 0.0]])

        return (np.cos(alpha) * (np.eye(3) - dyadaa) +
                dyadaa + np.sin(alpha) * axiala)

    R = random_rotation_tensor()
    components = np.eye(3)
    components[0, 0] = stretch
    components = np.dot(R, components).flatten()

    if kappa == 0.0:
        strain = np.log(stretch)
    else:
        strain = (stretch ** kappa - 1.0) / kappa

    # Test
    comp, rot = mml.core.deformation.strain_from_defgrad(components, kappa)

    assert arr_is_close(rot, R, tol)  # No rotation
    assert val_is_close(comp[0], strain, tol)
    assert arr_is_close(comp[1:], np.zeros(5), tol)  # No strain

    # Teardown
    pass




if __name__ == '__main__':
    test_defgrad_basic(0.0)
