Installation
*************

using pip
=========
The most easy way to install easy_mpl is using ``pip``
::
    pip install easy_mpl

However, if you are interested in installing all dependencies of easy_mpl, you can
choose to install all of them as well.
::
    pip install easy_mpl[all]

We can also specify the easy_mpl version that we want to install as below
::
    pip install easy_mpl==0.21.1


using github link
=================
You can also use github link to install easy_mpl.
::
    python -m pip install git+https://github.com/Sara-Iftikhar/easy_mpl.git

The latest code however (possibly with less bugs and more features) can be installed from ``dev`` branch instead
::
    python -m pip install git+https://github.com/Sara-Iftikhar/easy_mpl.git@dev

To install the latest branch (`dev`) with all requirements use ``all`` keyword
::
    python -m pip install "easy_mpl[all] @ git+https://github.com/Sara-Iftikhar/easy_mpl.git@dev"

using setup.py file
===================
go to folder where repository is downloaded
::
    python setup.py install