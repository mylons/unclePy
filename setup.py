from setuptools import setup

requires = [
    'pandas',
    'numpy',
    'matplotlib',
    'scipy',
    'scikit-learn',
    'pyyaml',
    'h5py',
    'sqlalchemy'
]

setup(
    name = 'unclePy',
    version = '1.0',
    packages = ['unclePy'],
    install_requires = requires,
    url = 'https://github.com/eric-hunt/unclePy',
    license = 'GNU Affero General Public License v3.0',
    author = 'Jacob Miller',
    author_email = 'jmiller@neb.com',
    description = 'A parser for Unchained Labs Uncle HDF5 binary files written in unclePy. '
)
