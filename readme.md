[![Documentation Status](https://readthedocs.org/projects/easy-mpl/badge/?version=latest)](https://easy-mpl.readthedocs.io/en/latest/?badge=latest)

[![Downloads](https://pepy.tech/badge/easy-mpl)](https://pepy.tech/project/easy-mpl)

Matplotlib is great library which offers huge flexibility due to its object oriented
programming style. However, **most** of the times, we the users don't need that 
much flexibiliy and just want to get things done as quickly as possible. For example 
why should I write at least three lines to plot a simple array with legend when same 
can be done in one line and my purpose is just to view the array. Why I can't simply
do ```plot(data)``` or ```imshow(img)```. This motivation gave birth to this library.
`easy_mpl` stands for easy maplotlib. The purpose of this is to ease the use of 
matplotlib while keeping the flexibility of object oriented programming paradigm 
of matplotlib intact. Using these one liners will save the time and will not hurt. 
Moreover, you can swap every function of this library with that of matplotlib and 
vice versa.

# Installation

This package can be installed using pip from pypi using following command

    pip install easy_mpl

# Usage


## plot

```python
from easy_mpl import plot
import numpy as np
plot(np.random.random(100))
# use x and y
plot(np.arange(100), np.random.random(100))
# use x and y with marker style
plot(np.arange(100), np.random.random(100), '.')

plot(np.random.random(100), '.')
# use cutom marker
plot(np.random.random(100), '--*')

plot(np.random.random(100), '--*', label='label')
# log transform y-axis
plot(np.random.random(100), '--*', logy=True, label='label')
```


## bar_chart

```python
from easy_mpl import bar_chart
bar_chart([1,2,3,4,4,5,3,2,5])
# specifying labels
bar_chart([3,4,2,5,10], ['a', 'b', 'c', 'd', 'e'])
# sorting the data
bar_chart([1,2,3,4,4,5,3,2,5], sort=True)
```


## regplot

```python
import numpy as np
from easy_mpl import regplot
x_, y_ = np.random.random(100), np.random.random(100)
regplot(x_, y_)
```


## imshow

```python
import numpy as np
from easy_mpl import imshow
x = np.random.random((10, 5))
imshow(x, annotate=True)
# show colorbar
imshow(x, colorbar=True)
```


## hist

```python
from easy_mpl import hist
import numpy as np
hist(np.random.random((10, 1)))
```


## pie

```python
from easy_mpl import pie
import numpy as np
pie(np.random.randint(0, 3, 100))
# or by directly providing fractions
pie([0.2, 0.3, 0.1, 0.4])
```

## scatter

```python
import numpy as np
from easy_mpl import scatter
import matplotlib.pyplot as plt
x = np.random.random(100)
y = np.random.random(100)
scatter(x, y, show=False)
# show colorbar
scatter(x, y, colorbar=True, show=False)
# retrieve axes for further processing
ax, _ = scatter(x, y, show=False)
assert isinstance(ax, plt.Axes)
```


## contour

```python
from easy_mpl import contour
import numpy as np
x = np.random.uniform(-2, 2, 200)
y = np.random.uniform(-2, 2, 200)
z = x * np.exp(-x**2 - y**2)
contour(x, y, z, fill_between=True, show_points=True)
# show contour labels
contour(x, y, z, label_contours=True, show_points=True)
```


## dumbbell_plot

```python
import numpy as np
from easy_mpl import dumbbell_plot
st = np.random.randint(1, 5, 10)
en = np.random.randint(11, 20, 10)
dumbbell_plot(st, en)
# modify line color
dumbbell_plot(st, en, line_kws={'color':"black"})
```


## ridge

```python
import numpy as np
from easy_mpl import ridge
data_ = np.random.random((100, 3))
ridge(data_)
# specifying colormap
ridge(data_, cmap="Blues")
# using pandas DataFrame
import pandas as pd
ridge(pd.DataFrame(data_))
```


## parallel_coordinates

```python
import random
import numpy as np
import pandas as pd
from easy_mpl import parallel_coordinates

ynames = ['P1', 'P2', 'P3', 'P4', 'P5']
N1, N2, N3 = 10, 5, 8
N = N1 + N2 + N3
categories = ['a', 'b', 'c', 'd', 'e', 'f']
y1 = np.random.uniform(0, 10, N) + 7
y2 = np.sin(np.random.uniform(0, np.pi, N))
y3 = np.random.binomial(300, 1 / 10, N)
y4 = np.random.binomial(200, 1 / 3, N)
y5 = np.random.uniform(0, 800, N)
# combine all arrays into a pandas DataFrame
data = np.column_stack((y1, y2, y3, y4, y5))
data = pd.DataFrame(data, columns=ynames)
# using DataFrame
parallel_coordinates(data, names=ynames)
# using continuous values for categories
parallel_coordinates(data, names=ynames, categories=np.random.randint(0, 5, N))
# using categorical classes
parallel_coordinates(data, names=ynames, categories=random.choices(categories, k=N))
# using numpy array
parallel_coordinates(data.values, names=ynames)
# with customized tick labels
parallel_coordinates(data.values, ticklabel_kws={"fontsize": 8, "color": "red"})
# using straight lines instead of breziers
parallel_coordinates(data, linestyle="straight")
# with categorical class labels
data['P5'] = random.choices(categories, k=N)
parallel_coordinates(data, names=ynames)
# with categorical class labels and customized ticklabels
data['P5'] = random.choices(categories, k=N)
parallel_coordinates(data,  ticklabel_kws={"fontsize": 8, "color": "red"})
```