
from setuptools import setup

setup(

    name='easy_mpl',

    version="0.13",

    description='one stop shop for matplotlib plots',
    long_description="",
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

    install_requires=['numpy', 'matplotlib', 'pandas'],
)