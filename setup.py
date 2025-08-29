import os
import setuptools

script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

# Installs
setuptools.setup(
    name="flow_drive",
    version="1.0.0",
    author="anonymous",
    packages=["flow_drive"],
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: Free for non-commercial use",
    ]
)
