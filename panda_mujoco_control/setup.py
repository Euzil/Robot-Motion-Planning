from setuptools import setup
from glob import glob

package_name = 'panda_mujoco_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Panda robot control with MuJoCo',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'task_executor = panda_mujoco_control.task_executor:main',
            'mujoco_bridge = panda_mujoco_control.mujoco_bridge:main',
        ],
    },
)
