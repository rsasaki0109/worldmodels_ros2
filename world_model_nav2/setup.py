import os
from glob import glob

from setuptools import find_packages, setup

package_name = "world_model_nav2"

setup(
    name=package_name,
    version="0.3.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Ryohei Sasaki",
    maintainer_email="rsasaki0109@gmail.com",
    description="Score Nav2 candidate trajectories with World Model risk.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "trajectory_scorer_node = world_model_nav2.trajectory_scorer_node:main",
            "score_demo = world_model_nav2.score_demo:main",
        ],
    },
)
