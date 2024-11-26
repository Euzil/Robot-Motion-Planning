from setuptools import find_packages, setup

package_name = 'panda_mujoco_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=['panda_mujoco_control', 'panda_mujoco_control.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'mujoco'],
    zip_safe=True,
    maintainer='youran',
    maintainer_email='youran@example.com',
    description='Panda robot control with MuJoCo and MoveIt2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mujoco_bridge = panda_mujoco_control.mujoco_bridge:main',
            'task_executor = panda_mujoco_control.task_executor:main',
        ],
    },
)
