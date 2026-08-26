import os
from glob import glob

from setuptools import find_packages
from setuptools import setup

package_name = 'teamd_janken_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='carrobo2026',
    maintainer_email='takeyama.ren796@mail.kyutech.jp',
    description='Car-Robo @Home tidy-up task state machine example',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'janken_sm = teamd_janken_pkg.janken_sm:main',
            'hand_recog = teamd_janken_pkg.nodes.hand_recog:main',
        ],
    },
)
