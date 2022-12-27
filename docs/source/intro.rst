.. _intro:

.. currentmodule:: easy_mpl

Introduction
*************

Matplotlib is great library which offers huge flexibility due to its object oriented
programming style. However, **most** of the times, we the users don't need that
much flexibiliy and just want to get things done as quickly as possible. For example
why should I write at least three lines to plot a simple array with legend when same
can be done in one line and my purpose is just to view the array. Why I can't simply
do ``plot(data)`` or ``imshow(img)``. This motivation gave birth to this library.
`easy_mpl` stands for easy maplotlib. The purpose of this is to ease the use of
matplotlib while keeping the flexibility of object oriented programming paradigm
of matplotlib intact. Using these one liners will save the time and will not hurt.
Moreover, you can swap every function of this library with that of matplotlib and
vice versa.


``easy_mpl`` contains two kinds of functions, one which are just wrappers around their
matplotlib alternatives. These include :func:`plot`, :func:`scatter`, :func:`bar_chart`, :func:`pie`,
:func:`hist`, :func:`imshow` and :func:`boxplot`. As the name suggests, these are just alternatives
to their matplotlib aliases. All of these functions take same input arguments as taken by
corresponding matplotlib functions. If these functions are given same input arguments as to their
matplotlib alternatives, these functions return same output as returned by matplotlib. Therefore,
we can consider them as alternative to matplotlib. All these functions take three further input
arguments. These are ``ax``, ``ax_kws`` and ``show``. The meanings of these three arguments are as below

 * If ``ax`` argument is given, then the plots are drawn on this otherwise either a new
   matplotlib axes is created or currently availabel axes is used.
 * The arguments to manipulate the x and y labels, ticklabels, title etc are called ``ax_kws``.
   These arguments are passed to :func:`easy_mpl.utils.process_axes` .
 * If ``show`` is set to False, then the axes is not *exhauseted*, which means, we can
   manipulate it if required and call `plt.show` or `plt.draw` after manipulating the axes.
   Otherwise, in default case (when ``show`` is True), the plot is drawn immediately after
   calling the corresponding function.

Moreover these wrapper functions also take some auxilliary input arguments
which can be used for further manipulation of these plots. For example the
:func:`imshow` function takes the ``whiten_grid`` argument. The other
kinds of functions are helper functions for data visualization and anlayis.
These include :func:`regplot`, :func:`dumbbell_plot`, :func:`ridge`, :func:`parallel_coordinates`,
:func:`taylor_plot`, :func:`lollipop_plot`, :func:`circular_bar_plot`,
:func:`violin_plot` and :func:`spider_plot`