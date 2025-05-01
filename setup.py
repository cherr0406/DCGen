from setuptools import find_packages, setup

setup(
    name="DCGen",
    version="0.1.0",
    description="Unofficial pip-installable version of DCGen",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
)
