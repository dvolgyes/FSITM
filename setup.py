#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.util import get_platform

short_description = 'Feature similarity index for tone mapped images (FSITM) - revised'

with open('requirements.txt', 'rt') as f:
    reqs = list(map(str.strip, f.readlines()))

setup(
    name='fsitm-revised',
    version='0.9.0',
    description=short_description,
    long_description=short_description,
    author='David Volgyes',
    author_email='david.volgyes@ieee.org',
    url='https://github.com/dvolgyes/FSITM',
    packages=['FSITM'],
    package_dir={'FSITM': 'src'},
    scripts=["src/FSITM.py",],
    data_files=[],
    keywords=['tone-mapping', 'image quality', 'metrics'],
    classifiers=[],
    license='BSD',
    platforms=[get_platform()],
    require=reqs,
    download_url='https://github.com/dvolgyes/FSITM/archive/latest.tar.gz',
)
