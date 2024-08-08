"""Setup script."""

from setuptools import setup, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='ltu',
    version='0.0.1',
    packages=find_packages(include=['ltu', 'ltu.*']),
    package_data={'fmnist' :['fmnist/*.pickle']},
    include_package_data=True,
    install_requires=required,
)
