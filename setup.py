
from setuptools import setup

import os

long_desc="easy_mpl"

fpath = os.path.join(os.getcwd(), "readme.md")
if os.path.exists(fpath):
    with open(fpath, "r") as fd:
        long_desc = fd.read()

setup(

    name='easy_mpl',

    version="0.21.4",

    description='one stop shop for matplotlib plots',
    long_description=long_desc,
    long_description_content_type="text/markdown",

    url='https://github.com/Sara-Iftikhar/easy_mpl',

    author='Sara Iftikhar',
    author_email='sara.rwpk@gmail.com',

    classifiers=[
        'Development Status :: 4 - Beta',

        'Natural Language :: English',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',

        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

    packages=['easy_mpl'],

    # 3.6.0 version has problem with parallel coords
    install_requires=['matplotlib>=3.3.0, <3.9.0',
                       ],
    extras_require = {"all": [
        "numpy>=1.16.2, <2.0.0",
        "matplotlib>=3.3.0, <3.9.0",
        "pandas>=0.23.3, <2.0.0",
        "scipy"]}
)