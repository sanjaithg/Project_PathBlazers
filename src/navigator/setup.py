from setuptools import find_packages, setup

package_name = 'navigator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ibot',
    maintainer_email='ibot@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
    'dev': ['pytest'],
    },
    entry_points={
        'console_scripts': [
          "joy = navigator.joy_to_vel:main",  
          "front = navigator.move_front_wheels:main",
          "rear = navigator.move_rear_wheels:main",
          "inv_kin = navigator.inv_kin:main",
          "viz = navigator.odom_vis:main",
        ],
    },
)
