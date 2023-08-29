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

INSTALL_REQUIRES=[
    'pandas>=1',
    'numpy>=1.13.0',
    'python-dateutil>=2.5',
    'matplotlib>=2',
    'statsmodels>=0.8',
    'scikit-learn>=0.19',
    'bokeh>=3.0.0',
    'colorcet',
    'param',
]

EXTRAS_REQUIRE={
    'optional': ['holoviews>=1.14.8', 'panel', 'pvlib>=0.9.0', 'openpyxl', ],
}
EXTRAS_REQUIRE['test'] = EXTRAS_REQUIRE['optional'] + [
    'coveralls',
    'pytest',
    'pytest-cov',
    'pytest-mock',
    'pytest-timeout',
]
EXTRAS_REQUIRE['build'] = EXTRAS_REQUIRE['optional'] + [
    'build',
    'twine',
]
EXTRAS_REQUIRE['docs'] = EXTRAS_REQUIRE['optional'] + [
    'docutils==0.18.1',
    'nbsphinx==0.9.1',
    'notebook',
    'recommonmark==0.7.1',
    'sphinx==6.1.3',
    'sphinx_rtd_theme==1.2.0',
]
EXTRAS_REQUIRE['all'] = sorted(set(sum(EXTRAS_REQUIRE.values(), [])))

setup(
    name='captest',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    url='http://github.com/bt-/pvcaptest',
    license='MIT',
    author='Ben Taylor',
    python_requires='>=3.8',
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    author_email='benjaming.taylor@gmail.com',
    description=(
        'Framework and methods to facilitate photovoltaic '
        'facility capacity testing following ASTM E2848.'),
    long_description=long_description,
    long_description_content_type='text/x-rst',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
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
