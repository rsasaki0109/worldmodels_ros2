from setuptools import find_packages, setup

package_name = "world_model_py"

setup(
    name=package_name,
    version="0.3.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Ryohei Sasaki",
    maintainer_email="rsasaki0109@gmail.com",
    description="Python adapter SDK and ROS 2 runtime for World Models.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # ROS 2 nodes
            "runtime_node = world_model_py.runtime_node:main",
            "monitor_node = world_model_py.monitor_node:main",
            "planning_node = world_model_py.planning_node:main",
            "sample_publisher = world_model_py.sample_publisher:main",
            "camera_sim = world_model_py.camera_sim:main",
            # standalone CLI + reference remote server (also usable without ROS)
            "world-model = world_model_py.cli:main",
            "world-model-server = world_model_py.server:main",
        ],
    },
)
