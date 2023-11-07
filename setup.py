from setuptools import setup, find_packages

setup(
    name='timepulse',
    version='0.0.1',
    description='A set of tools to help with timeseries flow',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='SQUAREDEV BV',
    author_email='hello@squaredev.io',
    url='https://github.com/squaredev-io/timepulse',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        # Dependencies go here
    ],
    extras_require={
        'dev': [
            'pytest',
        ]
    },
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',  # Specify which py version you support
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.9',  # Minimum version requirement of the package
    keywords='timeseries tools ml flow python machine learning',  # Short descriptors for your package
    package_data={
        'my_library': ['data/*.dat'],  # Include any package data in a subdirectory
    },
    entry_points={
        'console_scripts': [
            'my_library_cli=my_library.cli:main',  # This allows you to create a command-line script
        ],
    },
    include_package_data=True,
    zip_safe=False
)

setup(**setup_info)