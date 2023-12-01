from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="timepulse",
    version="0.0.1",
    description="A set of tools to help with timeseries flow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SQUAREDEV BV",
    author_email="hello@squaredev.io",
    url="https://github.com/squaredev-io/timepulse",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=[
        "absl-py==2.0.0",
        "astunparse==1.6.3",
        "cachetools==5.3.2",
        "exceptiongroup==1.2.0",
        "flatbuffers==23.5.26",
        "gast==0.5.4",
        "geographiclib==2.0",
        "geopy==2.4.0",
        "google-auth==2.23.4",
        "google-auth-oauthlib==1.1.0",
        "google-pasta==0.2.0",
        "grpcio==1.59.3",
        "h5py==3.10.0",
        "holidays==0.37",
        "idna==3.4",
        "importlib-metadata==6.8.0",
        "iniconfig==2.0.0",
        "joblib==1.3.2",
        "keras==2.15.0",
        "libclang==16.0.6",
        "Markdown==3.5.1",
        "MarkupSafe==2.1.3",
        "ml-dtypes==0.2.0",
        "numpy==1.26.2",
        "oauthlib==3.2.2",
        "opt-einsum==3.3.0",
        "packaging==23.2",
        "pandas==2.1.3",
        "pluggy==1.3.0",
        "protobuf==4.23.4",
        "pyasn1==0.5.1",
        "pyasn1-modules==0.3.0",
        "pytest==7.4.3",
        "pytest-cov==4.1.0",
        "python-dateutil==2.8.2",
        "pytz==2023.3.post1",
        "requests==2.31.0",
        "requests-oauthlib==1.3.1",
        "rsa==4.9",
        "scikit-learn==1.3.2",
        "scipy==1.11.4",
        "six==1.16.0",
        "tensorboard==2.15.1",
        "tensorboard-data-server==0.7.2",
        "tensorflow==2.15.0",
        "tensorflow-estimator==2.15.0",
        "tensorflow-io-gcs-filesystem==0.34.0",
        "tensorflow-macos==2.15.0",
        "termcolor==2.3.0",
        "threadpoolctl==3.2.0",
        "tomli==2.0.1",
        "typer==0.9.0",
        "typing_extensions==4.8.0",
        "tzdata==2023.3",
        "urllib3==2.1.0",
        "Werkzeug==3.0.1",
        "weather_data_retriever==1.1",
        "wrapt==1.14.1",
        "xgboost==2.0.2",
        "zipp==3.17.0",
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
    package_data={
        "timepulse": ["data/datasets/*.csv"],  # Include any package data in a subdirectory
    },
    entry_points={
        "console_scripts": [
            "my_library_cli=my_library.cli:main",  # This allows you to create a command-line script
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
