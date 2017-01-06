import numpy as np
from matmodlab import MaterialPointSimulator, ElasticMaterial, PlasticMaterial, \
    environ

def test_1():
    """Test the elastic and plastic materials for equivalence"""
    environ.SQA = True
    E = 10.
    Nu = .1
    G = E / 2. / (1. + Nu)
    K = E / 3. / (1. - 2. * Nu)

    jobid = 'Job-El'
    mps_el = MaterialPointSimulator(jobid)
    material = ElasticMaterial(E=E, Nu=Nu)
    mps_el.assign_material(material)
    mps_el.add_step('E'*6, [1,0,0,0,0,0], scale=.1, frames=1)
    mps_el.add_step('S'*6, [0,0,0,0,0,0], frames=5)
    mps_el.run()
    df_el = mps_el.df

    jobid = 'Job-Pl'
    mps_pl = MaterialPointSimulator(jobid)
    material = PlasticMaterial(K=K, G=G)
    mps_pl.assign_material(material)
    mps_pl.add_step('E'*6, [1,0,0,0,0,0], scale=.1, frames=1)
    mps_pl.add_step('S'*6, [0,0,0,0,0,0], frames=5)
    mps_pl.run()
    df_pl = mps_pl.df

    for key in ('S.XX', 'S.YY', 'S.ZZ', 'E.XX', 'E.YY', 'E.ZZ'):
        assert np.allclose(df_el[key], df_pl[key])
