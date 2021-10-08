# Machine Learning Lab 2 - NumPy & SciPy

Time to leave the basic Python lists behind - welcome ``NumPy``!

## 2 - NumPy

NumPy provides **efficient** data structures for vectors (one-dimensional arrays), matrices (two-dimensional arrays), and functions for operations that exceeds what you can do with basic python lists.

* Website: https://numpy.org/
* Reference (API): https://numpy.org/devdocs/reference/index.html#reference

### 2.1 - Overview

Following this lab, you should be able to:

* Create empty arrays (vectors and matrices)
* Create populated arrays (vectors and matrices)
* Get information about the data structures you're working with
* Use common array functions
* Iterate over the arrays
* Calculate descriptive statistics

### 2.2 - Creating NumPy arrays

First, in the jupyter notebook provided, import numpy and create an empty NumPy array (and assign to a variable ``an_array``):

```python
import numpy as np

an_array = np.array([])
```

To create a pre-populated NumPy array, you can do the following:

```python
another_array = np.array([0,1,2,3,4,3,2,1,0,3])
```

Next, add the content of ``another_array`` to ``an_array``, and print out the content.

```python
print(an_array)
```

Adding arrays together can be done using the ``np.append`` function. You can look that up in the API reference link provided above.

For more information on how to use an API, have a look at [[this video](https://bournemouth.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=7b8a0bcf-da0d-4dfc-9dcf-ac4900fa0b88)].

Your arrays are 1-dimensional, which means they are **vectors**. Now, let's create a 2-dimensional array, _AKA_ a **matrix**.

```python
c = np.array([(1,3,3,7), (4,2,4,2)])
```

**QUESTION**: what shape is this matrix? (rows x columns)

### 2.3 - Accessing data in arrays

Let's start with our vectors, e.g., ``another_array``. To access the values at certain indices (positions), you can do the same as with basic python lists.

For example, to get the value of the 3rd element (index 2):

```python
print('3rd value (at index 2):',another_array[2])
```

For matrices, the principle remains, but need to be mindful of a few things.

If you want a specific element, you need to reference both the column and row. For example:

```python
print("Value at (0, 2) =", c[0][2])
```

You can also get an entire row, e.g.,

```python
print("1st row =", c[0])
```

But, to access a column, we need to work with ``slicing``. Slicing allows us to use a ``:`` to signify using all values along a certain dimension (axis).

For example, to get all row values along the 2nd column (index 1):

```python
print("2nd column =", c[:,1])
```

You can learn much more about indexing and slicing in this [[NumPy Refeerence API page](https://numpy.org/doc/stable/reference/arrays.indexing.html)], but the above should be sufficient for this unit.


### 2.4 - Iterating over arrays

You can do this like a python list, as a ``for each`` loop or ``by index``.

For the latter, we can essentially create a loop over a set of numbers, which will always start at ``0`` until some ``range`` given. In this case, using the size of our vector ``another_array``:

```python
for i in range(another_array.size):
    print(another_array[i])
```

As you can see, that ties in directly with accessing elements of arrays from above.

The alternative is as simple as this for a **vector**:

```python
for value in another_array:
    print(value)
```

But, what happens if you do that with the matrix?

```python
for row in c:
    print(row)
```

The above code will iterate over each row, which may be what you want to do some times. However, if you want to iterate over each element, consider the following two approaches:

```python
print("Iterating over each element, row by row")
for row in c:
    for cell in row:
        print(cell, end=' ')
```

Or, using the flatten function.

```python
print("Iterating over each element, using flatten function")
for cell in c.flatten():
    print(cell, end=' ')
```

### 2.5 - NumPy functions and attributes

We already used a few NumPy functions and attributes above, like ``size`` and ``flatten()``.

#### 2.5.1 - Shape

Another attribute that can be very useful if you work with a matrix you've loaded from a file, is ``.shape``. This will tell you how many rows and columns the data structure has got.

**PS**: attributes are called like functions, but without any brackets ().

#### 2.5.2 - Infinity & NaN

As we may be dealing with data that is missing or have values like infinity, NumPy conveniently supports that!

You can use ``np.inf`` for inifinity, and ``np.nan`` for Not A Number (NAN), respectively.

#### 2.5.3 - Array creation functions

The ``arrange`` function allows you to create an array of numbers from _0 - (n-1)_, e.g.,

```python
# create a vector with numbers 1-19
a = np.arange(20)
```

The ``zeros`` function allows you to create an array of... You might have guessed it... All _zeros_!

```python
# create a 2x2 matrix of zeros
b = np.zeros((2,2))
```

And if you need an array of ``ones``:

```python
# create a 4x4 matrix of ones
c = np.ones((4,4))
```

Or, if you need an array with a specific fill value:

```python
# create a 3x3 matrix of NaN
d = np.full((3,3), np.nan)
```

For some linear algebra, you may need an identity matrix. Most likely not in in this unit, but including this for completeness:

```python
# create a 5x5 identity matrix (1s down the diagonal)
e = np.eye(5)
```

If you need to initialise an array with random numbers in the range [0,1]:

```python
# create a 2x2 matrix of random numbers in the range [0,1]
f = np.random.random((2,2))
```

We'll be talking much more about random numbers later in the unit when we get to probability, so this is just planting a seed!

Did you get the geeky pun?! :joy:


#### 2.5.4 - Array arithmetics

In the same way as you can access elements in an array as discussed above, you can update the values, e.g.,

```python
# set the value of the first element in the array to 5
another_array[0] = 5
```

If you need to do vector and matrix operations like addition, subtraction, division, etc., on the entire data structure, it's actually really simple!

```python
v = np.arange(4)
print("v     =", v)
print("v + 5 =", v + 5)
print("v - 5 =", v - 5)
print("v * 2 =", v * 2)
print("v / 2 =", v / 2)
```

If you want to check out more array arithmetics possible, [[this page](https://jakevdp.github.io/PythonDataScienceHandbook/02.03-computation-on-arrays-ufuncs.html#Array-arithmetic)] gives a really nice overview.


#### 2.5.5 - More functions [OPTIONAL]

There are some more functions you may enjoy as you continue on your journey with data science. See, for example, [[this article](https://towardsdatascience.com/10-numpy-functions-you-should-know-1dc4863764c5)], which has got some useful use cases and illustrations.


### 2.6 - Calculating descriptive statistics

I know this is also about using functions, so it could have been covered under the previous heading. However, I wanted to give it a "special" place, and not simply giving you the functions...

Yes, you'll have to look at the [[API for NumPy](https://numpy.org/devdocs/reference/routines.statistics.html)] to find out how to calculate:

* min (smallest value)
* max (largest value)
* mean
* median
* mode
* variance
* standard deviation


## 3 - SciPy

SciPy is a Python module/library for numerical integration, interpolation,
optimisation, linear algebra, and statistics.

* Website: https://www.scipy.org/
* Reference (API): https://docs.scipy.org/doc/scipy/reference/

We're not going to deep into SciPy this week, but we will return to it.

For this, week, SciPy fills a gap where NumPy can't calculate the ``mode``.
So, have a look at the API reference for how you can do that with **SciPy**.
In particular, look at [scipy.stats](https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html).

You can also compute a range of statistics using the [describe()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.describe.html) function in SciPy.
