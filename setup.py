from setuptools import setup, find_packages

setup(
    name='moppy',
    version='0.1.0',
    description='Use ProMPs to model and generate human-like movements.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Matthias Ebner & Paul Prünster',
    # qauthor_email='your.email@example.com',
    # url='https://github.com/yourusername/example_module',
    packages=['moppy',
              'moppy.deep_promp',
              'moppy.trajectory',
              'moppy.interfaces',
              'moppy.mp_types'],
    python_requires='>=3.12',
    install_requires=[
        'torch',
        'numpy',
        'matplotlib',
    ],
)