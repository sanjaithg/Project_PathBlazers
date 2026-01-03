#!/usr/bin/env python3
import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'bots'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[package_name, package_name + ".*"]),
    data_files=[
        ('share/ament_index/resource_index/packages',
         [f'resource/{package_name}']),
        ('share/' + package_name, ['package.xml']),
        # Install all launch/config files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'world'),
         [f for f in glob('world/*') if os.path.isfile(f)]),
        (os.path.join('share', package_name, 'world', 'models', 'cardboard_box'), [
            'world/models/cardboard_box/model.sdf',
            'world/models/cardboard_box/model.config',
        ]),
        (os.path.join('share', package_name, 'world', 'models', 'cardboard_box', 'meshes'),
         glob('world/models/cardboard_box/meshes/*')),
    ],
    install_requires=[
        'setuptools',
        'torch',
        'numpy',
        'squaternion',
        'tensorboard',
    ],
    zip_safe=True,
    maintainer='hillman',
    maintainer_email='ed23b055@smail.iitm.ac.in',
    description='Bots RL package with TD3 training',
    license='TODO: License declaration',
    extras_require={
        'dev': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            "test_td3 = bots.td3_rl.test_td3:main",
            "train_td3 = bots.td3_rl.train_td3:main",
            "FakeOdomPublisher = bots.td3_rl.FakeOdomPublisher:main",
        ],
    },
)
