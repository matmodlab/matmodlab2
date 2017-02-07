from setuptools import setup

setup(name='matmodlab2',
      version='1.0.0',
      description='A material point simulator for testing and evaluating material models',
      long_description=('The material model laboratory (Matmodlab2) '
                        'is a material point simulator developed as '
                        'a tool for developing and analyzing material '
                        'models for use in larger finite element codes.'),
      classifiers=[  # Classifier list:  https://pypi.python.org/pypi?:action=list_classifiers
                   "Development Status :: 3 - Alpha",
                   "Environment :: Console",
                   "Intended Audience :: Developers",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: BSD License",
                   "Natural Language :: English",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python :: 3",
                   "Topic :: Scientific/Engineering",
                   "Topic :: Scientific/Engineering :: Physics",
                   "Topic :: Software Development :: Testing",
                  ],
      url='https://github.com/matmodlab/matmodlab2',
      author='Tim Fuller and Scot Swan',
      author_email='timothy.fuller@utah.edu',
      license='BSD-3-Clause',
      packages=['matmodlab2',
                'matmodlab2.core',
                'matmodlab2.ext_helpers',
                'matmodlab2.fitting',
                'matmodlab2.materials',
                'matmodlab2.optimize',
                'matmodlab2.umat',
               ],
      package_dir={
                   'matmodlab2':'matmodlab2',
                  },
      package_data={
                    'matmodlab2':[
                                  'ext_helpers/*',
                                  'umat/*',
                                  'umat/umats/*',
                                 ],
                   },
      install_requires=['numpy', 'scipy'],
      zip_safe=False)
