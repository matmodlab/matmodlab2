from .core.matmodlab import MaterialPointSimulator
from .core.material import Material
from .core.materials import *
from .core.database import DatabaseFile
from .core.environ import environ
try:
    get_ipython()
    print('Setting up the Matmodlab notebook environment')
    environ.notebook = True
except NameError:
    pass
from .core.logio import logger

__all__ = ['MaterialPointSimulator',
           'Material', 'ElasticMaterial', 'AnisotropicElasticMaterial',
           'ElasticMaterialTotal',
           'PlasticMaterial', 'VonMisesMaterial',
           'NonhardeningPlasticMaterial', 'HardeningPlasticMaterial',
           'PolyHyperMaterial', 'MooneyRivlinMaterial',
           'DatabaseFile', 'environ', 'logger']
