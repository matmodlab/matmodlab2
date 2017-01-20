"""Test of the Tresca material model.  This model is courtesy of Scot Swan.

The model and plots from these verification tests support the publication:

<CITATION>

All attribution goes to Scot Swan for these tests. Any mistakes should be
assumed to be errors on my part (Tim Fuller).

"""
import pytest
import numpy as np
from matmodlab2 import *

def assym(A):
    return np.array([A[0,0], A[1,1], A[2,2], A[0,1], A[1,2], A[0,2]])

def f(x):
    fac = -1.0 + 2.0 * x
    A = np.array([[1.0, 0.0, 0.0],
                  [0.0, fac, 0.0],
                  [0.0, 0.0,-1.0]])
    return A

def test_tresca_yield_function_1():
    tresca = TrescaMaterial(K=1, G=1, A1=1.)
    assert np.isclose(-1., tresca.yield_function(np.zeros(6)))
    assert np.isclose(-1., tresca.yield_function(np.zeros((3,3))))

def test_tresca_yield_function_gradient_1():
    tresca = TrescaMaterial(K=1, G=1, A1=1.)
    X = np.array([[1,0,0],[0,0,0],[0,0,-1]])/2.0
    A = f(0.5)
    x = assym(X)
    a = assym(A)
    assert np.allclose(X, tresca.yield_function_gradient(A))
    assert np.allclose(x, tresca.yield_function_gradient(x))

def test_tresca_yield_function_num():
    """ Test the yield_function_gradient() against numerical derivatives
    using yield_function() as the source."""
    for _ in range(10):
        tresca = TrescaMaterial(K=1, G=1, A1=1.)
        A = np.random.uniform(-10.0, 10.0, (3, 3))
        A = (A + A.T) / 2.0
        f0 = tresca.yield_function(A)
        dfdsig_calc = np.zeros((3,3))
        tol = 1.0e-6
        for idx in range(3):
            for jdx in range(idx, 3):
                tmp = np.copy(A)
                tmp[idx,jdx] += tol
                if idx != jdx:
                    tmp[jdx,idx] += tol
                f1 = tresca.yield_function(tmp)
                if idx == jdx:
                    dfdsig_calc[idx,jdx] = (f1-f0)/tol
                else:
                    dfdsig_calc[idx,jdx] = (f1-f0)/tol/2.0
                    dfdsig_calc[jdx,idx] = (f1-f0)/tol/2.0
        dfdsig = tresca.yield_function_gradient(A)
        assert np.allclose(dfdsig, dfdsig_calc, rtol=1.0e-6, atol=1.0e-6)

def test_tresca_radial_return_dev():
    A = np.array([[1, 0, 0], [0, 0, 0], [0, 0,-1]])
    tresca = TrescaMaterial(K=1, G=1, A1=1.)
    sig_ret = tresca.radial_return(2.0 * A)
    assert np.allclose(A, sig_ret)
    assert np.isclose(0.0, tresca.yield_function(sig_ret))

def test_tresca_radial_return_txe():
    A = np.array([[2, 0, 0], [0,-1, 0], [0, 0,-1]])
    tresca = TrescaMaterial(K=1, G=1, A1=1.)
    sig_ret = tresca.radial_return(A)
    assert np.allclose(2./3. * A, sig_ret)
    assert np.isclose(0.0, tresca.yield_function(sig_ret))

def test_tresca_radial_return_txeish_unit():
    fac = np.sqrt(2.0) / np.cos(np.pi/12.0)
    A = np.array([[(3.0 + np.sqrt(3.0))/6.0, 0, 0],
                  [0, (np.sqrt(3.0)-3.0)/6.0, 0],
                  [0, 0, -1.0 / np.sqrt(3.0)]])
    tresca = TrescaMaterial(K=1, G=1, A1=1.)
    sig_ret = tresca.radial_return(3 * A)
    assert np.allclose(fac * A, sig_ret)
    assert np.isclose(0.0, tresca.yield_function(sig_ret))

def test_tresca_simple_return_dev():
    A = np.array([[1, 0, 0], [0, 0, 0], [0, 0,-1]])
    tresca = TrescaMaterial(K=1, G=1, A1=1.)
    sig_ret = tresca.simple_return(2. * A)
    assert np.allclose(A, sig_ret)
    assert np.isclose(0.0, tresca.yield_function(sig_ret))

def test_tresca_simple_return_txe():
    A = np.array([[2, 0, 0], [0,-1, 0], [0, 0,-1]])
    tresca = TrescaMaterial(K=1, G=1, A1=1.)
    sig_ret = tresca.simple_return(A)
    assert np.allclose(2./3.*A, sig_ret)
    assert np.isclose(0.0, tresca.yield_function(sig_ret))

def test_tresca_simple_return_1():
    A = np.array([[13,0,0],[0,-2,0],[0,0,-11]]) / 6.0
    tresca = TrescaMaterial(K=1, G=1, A1=1.)
    sig_ret = tresca.simple_return(A)
    assert np.allclose(np.array([[7,0,0],[0,-2,0],[0,0,-5]]) / 6.0, sig_ret)
    assert np.isclose(0.0, tresca.yield_function(sig_ret))

def test_tresca_eval():
    tresca = TrescaMaterial(K=8., G=3., A1=1.)
    A = np.array([[1, 0, 0], [0, 0, 0], [0, 0,-1]])
    I3, Z3x3 = np.eye(3), np.zeros((3,3))
    s, x, C = tresca.eval(1., 1., 0., 0., I3, I3, Z3x3, Z3x3, A, None)
    assert np.allclose(A, s)

def test_tresca_multi_step_verification():
    # Multi-step verification test
    I3, Z3x3 = np.eye(3), np.zeros((3,3))
    rt3 = np.sqrt(3.0)
    kmod = 8.0
    gmod = 1.0e9
    ys = 1.0e6
    fac = ys / (6.0 * gmod)
    eta = (3.0 - rt3) / 6.0
    stress = np.zeros((3,3))
    table_head = ["time", "eps1", "eps2", "eps3", "sig1", "sig2", "sig3"]
    table_data = [
        # t          e1                 e2                e3        s1       s2       s3
        [0.0,                0.0,            0.0,            0.0,     0.0,     0.0,     0.0],
        [1.0,      fac*(3      ),  fac*(0      ), -fac*(3      ),      ys,     0.0,     -ys],
        [1.0+eta, -fac*(1-2*rt3),  fac*(2      ), -fac*(1+2*rt3),  2*ys/3,  2*ys/3, -4*ys/3],
        [2.0,     -fac*(3-2*rt3),  fac*(6+2*rt3), -fac*(3+4*rt3),     0.0,      ys,     -ys],
        [2.0+eta, -fac*(5-2*rt3),  fac*(4+4*rt3),  fac*(1-6*rt3), -2*ys/3,  4*ys/3, -2*ys/3],
        [3.0,     -fac*(9      ),  fac*(6+6*rt3),  fac*(3-6*rt3),     -ys,      ys,     0.0],
        [3.0+eta, -fac*(7+2*rt3),  fac*(2+8*rt3),  fac*(5-6*rt3), -4*ys/3,  2*ys/3,  2*ys/3],
        [4.0,     -fac*(9+4*rt3),  fac*(  8*rt3),  fac*(9-4*rt3),     -ys,     0.0,      ys],
        [4.0+eta, -fac*(5+6*rt3), -fac*(2-8*rt3),  fac*(7-2*rt3), -2*ys/3, -2*ys/3,  4*ys/3],
        [5.0,     -fac*(3+6*rt3), -fac*(6-6*rt3),  fac*(9      ),     0.0,     -ys,      ys],
        [5.0+eta, -fac*(1+6*rt3), -fac*(4-4*rt3),  fac*(5+2*rt3),  2*ys/3, -4*ys/3,  2*ys/3],
        [6.0,      fac*(3-4*rt3), -fac*(6-2*rt3),  fac*(3+2*rt3),      ys,     -ys,     0.0],
        [6.0+eta,  fac*(1-2*rt3), -fac*(2      ),  fac*(1+2*rt3),  4*ys/3, -2*ys/3, -2*ys/3],
        [7.0,      fac*(3      ), -fac*(0      ), -fac*(3      ),      ys,     0.0,     -ys],
        [8.0,                0.0,            0.0,            0.0,     0.0,     0.0,     0.0],
    ]

    # Drive the TrescaModel using prescribed strain from the above table
    N = 20
    mps = MaterialPointSimulator('Tresca')
    tresca = TrescaMaterial(K=kmod, G=gmod, A1=ys)
    mps.material = tresca
    for row in table_data[1:]:
        mps.add_step('EEE', row[1:4], frames=N, time_whole=row[0])
    mps.run()
    simdata = mps.get2('Time', 'E.XX', 'E.YY', 'E.ZZ', 'S.XX', 'S.YY', 'S.ZZ')

    output = [table_data[0]]
    for idx in range(0, len(table_data)-1):
        t0 = np.array(table_data[idx][0])
        t1 = np.array(table_data[idx+1][0])
        e0 = np.array(table_data[idx][1:4])
        e1 = np.array(table_data[idx+1][1:4])
        dt = t1 - t0
        d = np.eye(3) * (e1 - e0) / N / dt
        for jdx in range(N):
            # Update the stress
            stress, _, _ = tresca.eval(t0, dt, 0., 0., I3, I3, Z3x3, d, stress, None)

            # Format for output
            t = t0 + (jdx+1) * (t1 - t0) / N
            e = e0 + (jdx+1) * (e1 - e0) / N
            s = stress.sum(axis=0)  # convert from 3x3 to 3
            output.append([t,] + e.tolist() + s.tolist())

    output = np.asarray(output)
    table_data = np.asarray(table_data)

    # Compare each column with the others. Do not use np.allclose() because
    # small errors (~machine precision) are introduced when dealing with large
    # stresses (~1.0e6) and are propagated to when the stress approaches zero.
    # np.allclose() has no memory of the 'average magnitude' of the stress over
    # the test, and subsequently flags what we know to be small deviations as
    # errors.
    for idx in range(len(table_data[0])):
        col_gold = table_data[:, idx]
        col_comp = output[::N, idx]
        col_sim = simdata[::N, idx]

        diff_comp = np.linalg.norm(col_gold - col_comp)
        diff_sim = np.linalg.norm(col_gold - col_sim)

        diff_mag = max(1.0, np.linalg.norm(col_gold))
        assert np.isclose(0.0, diff_comp / diff_mag)
        assert np.isclose(0.0, diff_sim / diff_mag)

    return

if __name__ == '__main__':
    test_tresca_multi_step_verification()
