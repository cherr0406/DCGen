from setuptools import setup, find_packages

setup(
    name="DCGen",
    version="0.1.0",
    description="Unofficial pip-installable version of DCGen",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
