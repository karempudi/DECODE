
import os
import setuptools

from setuptools import setup

# pip needs requirements here; keep in sync with meta.yaml!
requirements = [
    "numpy",
    # require torch 1.7 because 1.8+ breaks the neural network internally somewhere
    #"torch@https://download.pytorch.org/whl/cu110/torch-1.7.1%2Bcu110cp37-cp37-linux_x86_64.whl",
    #"torchaudio==0.7.2", # needs to be compatible with torch version
    #"torchvision==0.8.2", # also needs to be compatible with torch version
    "click",
    "deprecated",
    "gitpython>=3.1",
    "h5py",
    "importlib_resources",
    "matplotlib",
    "pandas",
    "pytest",
    "pyyaml",
    "requests",
    "scipy",
    "seaborn==0.10",
    "scikit-image",
    "scikit-learn",
    "tensorboard",
    "tifffile",
    "tqdm",
    "opencv-python",
    "notexbook-theme",
    "notebook",
]

setup(
    name='decode',
    version='0.11.1',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    url='https://rieslab.de',
    license='GPL3',
    author='Lucas-Raphael Mueller',
    author_email='',
    description=''
)