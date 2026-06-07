from setuptools import find_packages, setup

package_name = "world_model_datasets"

setup(
    name=package_name,
    version="0.3.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "pyarrow"],
    zip_safe=True,
    maintainer="Ryohei Sasaki",
    maintainer_email="rsasaki0109@gmail.com",
    description="Convert rosbag2 into robot-learning / world-model datasets.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "export_lerobot = world_model_datasets.converter:main",
        ],
    },
)
