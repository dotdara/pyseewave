from setuptools import setup, find_packages

setup(
    name="signal_processing_toolkit",
    version="0.1.0",
    author="Dara",
    author_email="daraaasuama1@gmail.com",
    description="A Python library for signal processing and spectral analysis.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dotdara/pyseewave",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
