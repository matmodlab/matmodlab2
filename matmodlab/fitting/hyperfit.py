from numpy import *
from numpy.linalg import inv
try:
    import sympy
    from sympy import symbols, Symbol, sqrt as Sqrt, Rational, lambdify
except ImportError:
    sympy = None

__all__ = ['POLYNOMIAL', 'MOONEY_RIVLIN', 'NEO_HOOKE',
           'UNIAXIAL', 'BIAXIAL', 'SHEAR', 'HyperFit']

POLYNOMIAL = 'Polynomial'
MOONEY_RIVLIN = 'Mooney Rivlin'
NEO_HOOKE = 'Neo Hooke'
UNIAXIAL = 'Uniaxial'
BIAXIAL = 'Biaxial'
SHEAR = 'Shear'

def HyperFit(model=POLYNOMIAL, **kwargs):
    """Factory method that returns a fitter object"""
    if sympy is None:
        raise RuntimeError('HyperFit requires sympy')
    if model == POLYNOMIAL:
        return PolynomialHyperFit(**kwargs)
    elif model == MOONEY_RIVLIN:
        kwargs['n'] = 2
        return PolynomialHyperFit(**kwargs)
    elif model == NEO_HOOKE:
        kwargs['n'] = 1
        return PolynomialHyperFit(**kwargs)
    raise ValueError('unknown HyperFit model {0}'.format(model))

def lstsq(A, b):
    """Least squares fit to

        A.x = b

    from which

       Transpose[A].A.x = Transpose[A].b
                      x = Inverse[Transpose[A].A].Transpose[A].b

    """
    A = asarray(A)
    b = asarray(b)
    return dot(dot(inv(dot(A.T, A)), A.T), b)

def moving_average(a, n=3):
    ret = cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class PolynomialHyperFit:
    ij = ((1,0), (0,1), (2,0), (1,1), (0,2),
          (3,0), (2,1), (1,2), (0,3))

    def __init__(self, n=3, i2_dep=True):
        """Expand the hyperelastic energy function to give the axial stress

        The list of the hyperelastic coefficients and the associated stress
        stored in coeffs and stress_diff. The actual axial stress would be
        given by S=coeffs.stress_diff. Note that the terms sent back are
        symbolic.

        n is the order of the expansion

        """
        self.n = n
        self.x = None

        # expanded hyperelastic model
        lam, l1, l2, l3 = symbols('lambda lambda_1 lambda_2 lambda_3')
        I1 = l1 ** 2 + l2 ** 2 + l3 ** 2
        I2 = (l1 * l2) ** 2 + (l2 * l3) ** 2 + (l3 * l1) ** 2
        J = l1 * l2 * l3

        I1b = I1 / (J ** Rational(2,3))
        I2b = I2 / (J ** Rational(4,3))

        # energy function and coefficients
        W, C = [], []
        k = m = 0
        while k < self.n:
            i, j = self.ij[m]
            m += 1
            if not i2_dep and j:
                continue
            C.append(Symbol('C_{{{0}{1}}}'.format(i,j)))
            W.append((I1b - 3) ** i * (I2b - 3) ** j)
            k += 1

        self.coeffs = C
        self.energy = W

        # stress difference
        self.stress_diff = [(l1 * W[i].diff(l1) - l3 * W[i].diff(l3))
                            for i in range(len(self.ij[:self.n]))]

    def fit(self, xy, type=UNIAXIAL):
        """Fit the stress vs strain curve with a nth order hyperelastic
        model

        """
        xy = asarray(xy)

        lam, l1, l2, l3 = symbols('lambda lambda_1 lambda_2 lambda_3')
        if type == UNIAXIAL:
            # uniaxial tension, incompressible.
            c = {l1: lam, l2: 1/Sqrt(lam), l3: 1/Sqrt(lam)}

        elif type == BIAXIAL:
            # biaxial tension, incompressible.
            c = {l1: lam, l2: lam, l3: 1/lam/lam}

        elif type == SHEAR:
            # biaxial tension, incompressible.
            c = {l1: lam, l2: 1/lam, l3: 1}

        else:
            raise RuntimeError('unrecognized data type')

        # The /lam term converts stress to engineering stress
        S1 = [s.subs(c) / lam for s in self.stress_diff]
        self.fun = [lambdify(lam, s) for s in S1]

        A = []
        u = xy[:,0] + 1
        for l in u:
            A.append([f(l) for f in self.fun])

        self.x = lstsq(A, xy[:,1])

        fi = self.eval(xy[:,0])
        self.fiterr = sqrt(mean((xy[:,1] - fi) ** 2))

        return self.x

    def eval(self, strain, x=None, fac=None):
        """Evaluate the nth order hyperelastic model.

        x is a list of hyperelastic coeficients (found with hyperfit).

        """
        if x is None:
            x = self.x.copy()

        if fac is not None:
            x = array(x)
            x *= fac

        assert len(x) == self.n
        A = []
        for e in strain:
            A.append([f(e+1) for f in self.fun])
        return dot(A, x).flatten()

    def pprint(self, x=None):
        if x is None:
            x = self.x
        y = ['C{0}{1}={2:.8f}'.format(i,j,x[k])
             for k, (i, j) in enumerate(self.ij[:self.n])]
        print ', '.join(y)

    def todict(self, x=None):
        if x is None:
            x = self.x
        keys = ['C{0}{1}'.format(i,j) for (i, j) in self.ij[:self.n]]
        return dict(zip(keys, x))

    def gendata(self, x, filename='data.csv'):
        strain = linspace(-.25, 3., 100)
        s = self.eval(strain, x)
        noise = random.normal(0, .03*amax(s), 100)
        with open(filename, 'w') as fh:
            for row in zip(strain, s+noise):
                x, y = [float(_) for _ in row]
                fh.write('{0:.18f},{1:.18f}\n'.format(x, y))

    def bp_plot(self, xy, x=None, plot=None):

        if x is None:
            if self.x is None:
                return
            x = self.x.copy()

        import bokeh.plotting as bp

        if plot is None:
            plot = bp.figure()

        if xy is not None:
            plot.circle(xy[:,0], xy[:,1])

        xp = linspace(amin(xy[:,0]), amax(xy[:,0]))
        yp = self.eval(xp, x)
        plot.line(xp, yp, color='black', line_width=1.5)

        return plot

if __name__ == '__main__':
    a = PolynomialHyperFit(n=3)
    a.gendata([11e6, .75e5, -1e3])
