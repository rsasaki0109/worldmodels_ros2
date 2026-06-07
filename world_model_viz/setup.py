import os
from glob import glob

from setuptools import find_packages, setup

package_name = "world_model_viz"

setup(
    name=package_name,
    version="0.4.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
        (os.path.join("share", package_name, "foxglove"), glob("foxglove/*.json")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Ryohei Sasaki",
    maintainer_email="rsasaki0109@gmail.com",
    description="RViz/Foxglove visualization of imagined World Model futures.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "occupancy_marker_node = world_model_viz.occupancy_marker_node:main",
        ],
    },
)
