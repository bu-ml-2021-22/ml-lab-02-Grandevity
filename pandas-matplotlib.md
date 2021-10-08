# Machine Learning Lab 2 - Pandas & MatplotLib

So far, so good?

## 4 - Pandas

Next, we're going to get familiar with **pandas**.
* Website: https://pandas.pydata.org/
* API: https://pandas.pydata.org/pandas-docs/stable/reference/index.html

Pandas uses ``NumPy`` under the bonnet, providing high-performance, easy-to-use data structures and data analysis tools. It typically feeds statistical analysis in ``SciPy``, visualising data in ``Matplotlib`` and machine learning in ``Scikit-Learn``.


### 4.1 - Overview

Following this lab, you should be able to:

* Load datasets into Pandas data structures
* View the raw data in Jupyter Notebook
* Get descriptive statistics from the data
* Run queries over the data

Later in the unit, we'll look at using Pandas for cleaning and transforming data as well.


### 4.2 - Loading data into Pandas

First, let's just create some dummy data in a standard python data structure (a dictionary).

```python
import pandas as pd

# a hardcoded dictionary
data = {
    'apples': [3, 2, 0, 1],
    'oranges': [0, 3, 7, 2]
}

# load into a DataFrame
purchases_dataframe = pd.DataFrame(data)
purchases_dataframe # will print in a Jupyter Notebook with nice formatting (but only when the last line in the cell)
```

As you can see, the data has been loaded into a Pandas [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), which is like a 2-dimensional NumPy array (matrix) with meta-data and additional functions.

Each column in a Pandas DataFrame is a [Series](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html), with a whole host of attributes and functions/methods as well.

But, before we proceed to looking at some of these attributes and functions, let's load a proper dataset.

```python
import pandas as pd

dataset = pd.read_csv('data/ml-assessment-dataset.csv')
dataset.head() # displays the header and top half of the dataset
```

The features of the dataset are as follows:

 * ``L4 mean mark`` - the mean mark from Level 4 (based on the 6 taught units prior to this unit).
 * ``Did formative assessment`` - a 1 signifies that the student did the formative assessment in the unit, and a 0 signifies that they did not.
 * ``Submitted`` - a 1 signifies the student submitted their assignment, and a 0 signifies a non-submission.
 * ``Plagiarism`` - a 1 signifies that the student was caught plagiarising (and marks will have been affected), and a 0 signifying not caught plagiarism (and hopefully did not).
 * ``VLE content visited`` - the proportion of content visited on Brightspace (e.g.,lecture slides, recordings, additional resources, assignment brief, etc).
 * ``Lecture engagement`` - the proportion of lectures attended / engaged with (self-reported).
 * ``Lab engagement`` - the proportion of labs attended / engaged with (self-reported).
 * ``Classifier impl`` - the chosen classifier the student decided to implement from scratch (was a past requirement, but not this year).
 * ``Code mark`` - the mark for the code (classifier implemented from scratch, max 30%).
 * ``Report mark`` - the mark for the report (max 70%).
 * ``Total mark`` - the total / final mark (code + report).
 * ``Class`` - a 1 signifies that the student passed, and a 0 signifies a fail.

 This is a real dataset of anonymised data from past students doing this Machine Learning unit,
 though most of the data has been modified a bit to ensure additional protection of past students' data.
 The original data has also been sampled a bit differently, so the class distribution is not
 what it was in reality. However, the individual student trends are still the same, so what you will learn and
 observe is based on real data.
 

### 4.3 - Viewing properties of the dataset

You've already seen one function above, ``.head()``, which outputs the first five rows of your DataFrame by default. To output a different number of rows, just pass on the number you want as an argument, e.g.,

```python
dataset.head(10)
```

To see the bottom half of the dataset, use ``.tail()``, which also takes a number of rows as an argument, e.g.,

```python
dataset.tail(10)
```

In reality, you're unlikely to peruse the data in such a manner. You're more likely to call some functions for some key ``.info()`` and to ``.describe()`` the dataset, as well as the ``.shape`` attribute (just like for NumPy arrays).

```python
print(dataset.shape)
print(dataset.info())
print(dataset.describe())
```

``.info()`` gives you some overview information about the dataset, including the number of instances (rows), the number of features (columns), whether there's missing values/data, and the datatype for each feature.

``.describe()`` gives you **descriptive statistics** of each feature.

As ``.describe()`` can spam your notebook/command line, you can do this for specific features. For example:

```python
print("\nInfo about the class\n")
print(dataset['Class'].describe(),"\n")
```

Finally, for now, we can also get the numbers for the distribution of features, like the ``class`` by using the ``value_counts()`` function.

```python
print(dataset['Class'].value_counts())
```

We'll get into more details in the **Exploratory Data Analysis** section below. But first, let's just introduce the basics of visualisation (more on this next lab).


## 5 - Matplotlib

For visualisation, **matplotlib** is a very popular module/library, annd it integrates quite seamlessly with ``Pandas``.
* Website: https://matplotlib.org/
* API: https://matplotlib.org/api/index.html


### 5.1 - Overview

We're just introducing the topic of visualisation this week. The aim is simply to get familiar with using the ``.plot()`` function on a Pandas ``DataFrame``.

### 5.2 - Histogram plot

Histograms are useful for understanding the probability distribution of data. To plot a histogram for the ``L4 mean mark`` feature, we need to:

1. specify the ``L4 mean mark`` feature like you're accessing the elment of a NumPy array

```python
dataset['L4 mean mark']
```

2. call the ``.plot()`` function and specify the ``kind`` of plot (``hist``), and here also giving the plot a ``title``:

```python
.plot(kind='hist', title='Level 4 mean mark distribution');
```

3. putting the two together, here's the code that should work by copy-pasting:

```python
# a histogram for the 'Age' feature
dataset['L4 mean mark'].plot(kind='hist', title='Level 4 mean mark distribution');
```

### 5.2 - Scatter plot

The recipe is pretty much the same if you want to do a scatter plot, but in this case we need to specify the two features (columns) within the ``.plot()`` function (as arguments).

To do a scatter plot for ``Resting blood pressure`` and ``Maximum heart rate``, you can copy-paste the following into your Jupyter Notebook:

```python
# a simple scatter plot for 'Resting blood pressure' vs 'Maximum heart rate'
dataset.plot(kind='scatter', x='L4 mean mark', y='Total mark', title='Level 4 mean mark vs unit mark');
```

### 5.3 - Bar chart for the class distribution

To plot the distribution of a feature, like the class, we can do something similar to the examples above. However, we need to get call a function to get the ``.value_counts()`` first.

And, a typical way of visualising the class distribution is a ``bar chart``:

```python
# a bar chart for the 'class' distribution
dataset['Class'].value_counts().plot(kind='bar', title='Class distribution');
```
