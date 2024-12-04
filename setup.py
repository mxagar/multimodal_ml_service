"""Setup script for the multimodal_ml_service package.

Author: Mikel Sagardia
Date: 2024-12-04
"""
from setuptools import setup, find_packages

setup(
    name="multimodal_ml_service",
    version="0.0.1",
    description="Blueprint for building a Machine Learning Service with multiple models and modalities.",
    author="Mikel Sagardia",
    author_email="sagardia.mikel@gmail.com",
    packages=find_packages(include=["src*", "src.*"]), # Include the `src` namespace
    package_dir={"src": "src"}, # Map the `src` directory to the `src` namespace
    python_requires=">=3.11",
    install_requires=[], # Use pip-tools for dependency management
)
