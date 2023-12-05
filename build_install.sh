#!/bin/bash

# Remove the 'dist' and 'build' folders
rm -rf dist build timepulse.egg-info

# Build source and wheel distributions using setup.py
python setup.py sdist bdist_wheel

# Uninstall the existing 'timepulse'
pip uninstall -y timepulse

# Install the updated version of 'timepulse'
pip install .
