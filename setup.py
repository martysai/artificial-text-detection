from setuptools import setup

setup(
    name="detection",
    version="0.1.0",
    packages=["detection"],
    data_files=[("./", ["requirements.txt"])],
    test_suite="nose.collector",
    tests_require=["nose"],
)
