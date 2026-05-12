from setuptools import setup, find_packages

setup(
    name="fellowship-of-the-ring",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "pandas",
        "matplotlib",
        "scipy",
        "meshio",
        "pyvista",
        "h5py"
    ],
)