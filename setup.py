from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="timepulse",
    version="0.2.0",
    description="A set of tools to help with timeseries flow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SQUAREDEV BV",
    author_email="hello@squaredev.io",
    url="https://github.com/squaredev-io/timepulse",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=[
        "holidays>=0.38",
        "joblib>=1.1.1",
        "numpy>=1.24.3",
        "pandas>=2.0.2",
        "pytest>=7.4.3",
        "scikit_learn>=1.3.2",
        "setuptools>=65.5.0",
        "tensorflow>=2.14.0",
        # "tensorflow_macos>=2.14.0",
        "typer>=0.9.0",
        "xgboost>=2.0.2",
    ],
    extras_require={
        "dev": [
            "pytest",
        ]
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",  # Specify which py version you support
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",  # Minimum version requirement of the package
    keywords="timeseries tools ml flow python machine learning",  # Short descriptors for your package
    package_data={},
    entry_points={
        "console_scripts": [
            "timepulse_cli=timepulse.cli:main",  # This allows you to create a command-line script
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
