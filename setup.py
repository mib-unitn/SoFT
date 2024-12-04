from setuptools import setup, find_packages

setup(
    name='SoFT',
    version='0.3_rc3',    
    description='A Python lib for Solar Feature Tracking',
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
        'DDevelopment Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',  
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
)
