from setuptools import setup, find_packages
import io
import versioneer

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.rst')

setup(
    name='captest',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    url='http://github.com/bt-/pvcaptest',
    license='MIT',
    author='Ben Taylor',
    python_requires='>=3.5',
    install_requires=['pandas>=0.24',
                      'numpy>=1.13.0',
                      'python-dateutil>=2.5',
                      'matplotlib>=2',
                      'statsmodels>=0.8',
                      'scikit-learn>=0.19',
                      'bokeh>=1',
                      ],
    extras_require={'viz': ['holoviews>=1.11'],
                    'csky': ['pvlib>=0.7.0',
                             'tables'],
                    'all': ['holoviews>=1.11',
                            'pvlib>0.7.0',
                            'tables']
                    },
    author_email='benjaming.taylor@gmail.com',
    description=('Framework and methods to facilitate photovoltaic '
    'facility capacity testing following ASTM E2848.'),
    long_description=long_description,
    packages=['captest'],
    include_package_data=True,
    platforms='any',
    classifiers=['Programming Language :: Python :: 3',
                 'Development Status :: 2 - Pre-Alpha',
                 'Natural Language :: English',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering'
                 ]
)
