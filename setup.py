#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

install_requires = [
    'numpy>=1.16.0',
    'pandas>=0.23.4',
    'scikit-learn>=0.20.2',
    'tensorflow>=1.13.0, <2.0',
    'tensorpack==0.9.4',
]

setup_requires = [
    'pytest-runner>=2.11.1',
]

test_require = [
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
]

development_requirements = [
    # general
    'bumpversion>=0.5.3',
    'pip>=9.0.1',
    'watchdog>=0.8.3',

    # docs
    'm2r>=0.2.0',
    'Sphinx>=1.7.1',
    'sphinx_rtd_theme>=0.2.4',

    # style check
    'flake8>=3.5.0',
    'isort>=4.3.4',

    # fix style issues
    'autoflake>=1.1',
    'autopep8>=1.3.5',

    # distribute on PyPI
    'twine>=1.10.0',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1',
    'tox>=2.9.1',
]

extras_require = {
    'test': test_require,
    'dev': development_requirements + test_require,
}

setup(
    author="MIT Data To AI Lab",
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Generative adversarial training for synthesizing tabular data",
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'tgan=tgan.cli:main'
        ]
    },
    install_package_data=True,
    install_requires=install_requires,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='tgan',
    name='tgan',
    packages=find_packages(include=['tgan', 'tgan.*']),
    python_requires='>=3.5',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=test_require,
    url='https://github.com/sdv-dev/TGAN',
    version='0.1.1-dev',
    zip_safe=False,
)
