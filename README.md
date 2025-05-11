# Programming and Scripting Project

The following project contains an analysis of the Fischer's iris data set. The dataset was originally published by Ronald Fisher in 1936 and has been used for the testing and development of classification methods and machine learning.

The analysis is part of the Programming and Scripting module of the higher diploma in computing at ATU.

We were asked to research the dataset and investigate it, as if we were asked to give a presentation on the dataset where we explain what investigating a data set entails and how Python can be used to do it, presenting both our code and its output.

## Contents

### Introduction

I introduce the content and the dataset.

### Imports

I import the packages needed for the project. A list of the requirements can be found in the repository.

### Data Loading

In this section I import the data for analysis by reading from a file found in the iris folder of the repository.

### Storing Variables

In this section I create python variables separating the data into more managable slices or sections.

### Describing the Data

In this section I generate statistics describing the data.

### Creating Histograms

In this section I present histograms and the amount of customisation available to the user.

### Creating KDE Plots With Seaborn

This next section contains similar analysis to the histograms section but improves upon the same idea using kernel density plots from seaborn.

### Creating Box Plots

Using MatPlotLib, I create highly customised box plots of the features.

### Removing Fliers

In this section I explore removing fliers.

### Creating a Scatter Plot

In this section I create scatter plots of the data.

### Using Seaborn

In this section I create a pairgrid using seaborn that represents the data using KDE and scatter plots. It aims to show the power of seaborn using very little code to gain full effect.

### Heatmap of Pearson Correlation Coefficients

In this section I use numpy and MatPlotLib to create heatmaps of the correlation coefficients of the features of each class or species of flower.

### Lines of Best Fit

Using both seaborn and MatPlotLib, I create a mixed scatter KDE and line plot on one axis. NumPy is used to determine the line of best fit for the data.

### Machine Learning

Using a support vector machine to approximate hyperplanes that serve to predict classification of the data.

## Setup

The code can be run in codespaces with no trouble but you can download the code and run it locally if needed.

If you choose to run this in your own local environment, there is a requirements file that contains the necessary packages needed to execute all cells in the notebook. This can be imported as an environment using conda:

> $ conda create --name <env> --file requirements_conda.txt

And using pip you can import to your current working environment using:

> $ pip install -r requirements_pip.txt

## References used:

- [pandas.read_csv Api Reference](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)
- [Geeks For Geeks - How to export Pandas DataFrame to a CSV file?](https://www.geeksforgeeks.org/how-to-export-pandas-dataframe-to-a-csv-file/)
- [Pyplot Tutorial](https://matplotlib.org/stable/tutorials/pyplot.html#sphx-glr-tutorials-pyplot-py)
- [Histograms Examples](https://matplotlib.org/stable/gallery/statistics/hist.html)
- [Histogram Examples - Updating histograms colors section](https://matplotlib.org/stable/gallery/statistics/hist.html#updating-histogram-colors)
- [API Reference - matplotlib.pyplot.hist](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html)
- [Choosing Colormaps in Matplotlib](https://matplotlib.org/stable/users/explain/colors/colormaps.html)
- [Stack Overflow question - Why does my xlabel not show up?](https://stackoverflow.com/questions/30019671/why-does-my-xlabel-not-show-up-its-not-getting-cut-off)
- [API Reference matplotlib.pyplot.subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots)
- [matplotlib Subplots Axes and Figures Example](https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/figure_title.html)
- [seaborn.kdeplot Documentation](https://seaborn.pydata.org/generated/seaborn.kdeplot.html)
- [Statology - How to Create Subplots in Seaborn (With Examples)](https://www.statology.org/seaborn-subplots/)
- [Stack Overflow Question - Hide legend from seaborn pairplot](https://stackoverflow.com/questions/54781243/hide-legend-from-seaborn-pairplot)
- [API Reference - matplotlib.pyplot.boxplot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html)
- [numpy.array Documentation](https://numpy.org/doc/stable/reference/generated/numpy.array.html)
- [matplotlib Box Plots Examples](https://matplotlib.org/stable/gallery/statistics/boxplot_color.html)
- [Stack Overflow Thread: Pandas boxplot: set color and properties for box, median, mean](https://stackoverflow.com/questions/35160956/pandas-boxplot-set-color-and-properties-for-box-median-mean)
- [Stack Overflow Thread: Flier colors in boxplot with matplotlib](https://stackoverflow.com/questions/43342564/flier-colors-in-boxplot-with-matplotlib)
- [How to Add a Title to Seaborn Plots \(With Examples\)](https://www.statology.org/seaborn-title/)
- [Pandas API Reference - pandas.DataFrame.copy()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.copy.html)
- [Pandas User Guide - Indexing and selecting data](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy)
- [Stack Exchange: Removing outliers renders a new distribution that has its own outliers](https://stats.stackexchange.com/questions/492995/removing-outliers-renders-a-new-distribution-that-has-its-own-outliers)
- [matplotlib Examples - List of Named Colors](https://matplotlib.org/stable/gallery/color/named_colors.html)
- [seaborn.pairplot API Reference](https://seaborn.pydata.org/generated/seaborn.pairplot.html)
- [seaborn.PairGrid Api Reference](https://seaborn.pydata.org/generated/seaborn.PairGrid.html)
- [Stack Overflow Question: How to show the title for the diagram of Seaborn pairplot() or PridGrid()](https://stackoverflow.com/questions/36813396/how-to-show-the-title-for-the-diagram-of-seaborn-pairplot-or-pridgrid)
- [Seaborn Tutorial: Axis Grids](https://seaborn.pydata.org/tutorial/axis_grids.html)
- [Wikipedia - Pearson Correlation Coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
- [What is Six Sigma - Confidence Intervals: Why n>30 is Acceptable as Population Representative?](https://whatissixsigma.net/confidence-intervals-why-n30-is-acceptable-as-population-representative/)
- [API Reference - pandas.DataFrame.corr](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html)
- [Annotated Heatmap Example](https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html)
- [Stack Abuse - Calculating Pearson Correlation Coefficient in Python with Numpy](https://stackabuse.com/calculating-pearson-correlation-coefficient-in-python-with-numpy/)
- [API Reference - matplotlib.pyplot.colorbar](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html)
- [Data Flair - Iris Flower Classification](https://data-flair.training/blogs/iris-flower-classification/)
- [Duck.ai prompt - 'precision recall f1 score' and 'Tell me more'](https://duckduckgo.com/?q=precision+recall+f1+score&ia=chat)
