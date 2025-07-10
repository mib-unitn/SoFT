from setuptools import setup, find_packages
from os import path
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='solar_ft',
    version='1.0.6',
    description='A Python lib for Solar Feature Tracking',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mib-unitn/SoFT',
    author='Michele Berretti',
    author_email='michele.berretti@unitn.it',
    license='GPL-3.0',
    packages=['soft'],
    install_requires=['pandas',
                      'numpy',
                      'numba',
                      'astropy',  
                      'scipy',
                      'scikit-image',
                      'matplotlib',   
                      'tqdm',
                      'pathos',
                      'typing',
                      ],

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',  
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
)
