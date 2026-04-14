from setuptools import setup, find_packages

setup(
    name="fellowship-of-the-ring",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib"
    ],
)