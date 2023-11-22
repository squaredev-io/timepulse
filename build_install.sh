#!/bin/bash

# Specify the repository URL
repo_url="https://github.com/stavrostheocharis/weather_data_retriever.git"

# Specify the folder name
folder_name="weather_data_retriever"

# Specify the package name
package_name="weather_data_retriever"

# Create the folder
mkdir -p $folder_name

# Clone the repository into the folder
git clone $repo_url $folder_name

# Optional: Navigate into the cloned folder
cd $folder_name

# Remove existing build artifacts
rm -rf dist build $package_name.egg-info

# Build source and wheel distributions using setup.py
python setup.py sdist bdist_wheel

# Uninstall the existing package
pip uninstall -y $package_name

# Install the updated version of the package
pip install .

# Navigate back to the original directory
cd ..

# Remove the cloned folder
rm -rf $folder_name

# Update the repository
git pull

# Remove the 'dist' and 'build' folders
rm -rf dist build timepulse.egg-info

# Build source and wheel distributions using setup.py
python setup.py sdist bdist_wheel

# Uninstall the existing 'timepulse'
pip uninstall -y timepulse

# Install the updated version of 'timepulse'
pip install .
