import os
from setuptools import setup
from glob import glob

package_name = 'mr_goto'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        # (os.path.join('share', package_name), glob('launch/*.launch.*')),
        (os.path.join('share', package_name), glob('launch/*.launch.xml'))
    ],
    install_requires = ['setuptools'],
    zip_safe = True,
    maintainer = 'parallels',
    maintainer_email = 'stanusoiumihai@gmail.com',
    description = 'TODO: Package description',
    license = 'TODO: License declaration',
    tests_require = ['pytest'],
    entry_points = {
        'console_scripts': [
            'goto = mr_goto.goto:main',
            'planner = mr_goto.planner:main',
            'bspline = mr_goto.bspline:main'
        ],
    },
)
