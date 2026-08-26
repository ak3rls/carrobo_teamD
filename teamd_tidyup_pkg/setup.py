import os
from glob import glob

from setuptools import find_packages
from setuptools import setup

package_name = 'teamd_tidyup_pkg'
setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    package_data={
        'teamd_tidyup_pkg.nodes': ['*.npz'],
    },
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
        (
            os.path.join('share', package_name, 'models'),
            glob('models/*'),
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
            'tidyup_sm = teamd_tidyup_pkg.tidyup_sm:main',
            'tidyup_sm1 = teamd_tidyup_pkg.tidyup_sm1:main',
            'open_drawer = teamd_tidyup_pkg.states.drawer_open:main',
            (
                'yoloe_detection_service = '
                'teamd_tidyup_pkg.nodes.yoloe_detection_service:main'
            ),
        ],
    },
)
