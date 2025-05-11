# Imports

# Numpy for dealing with arrays

import numpy as np

# Pandas for dealing with dataframes

import pandas as pd

# MatPlotLib for creating plots

import matplotlib.pyplot as plt

# Colors from MatPlotLib for more control over colour schemes

from matplotlib import colors

# Seaborn for creating plots

import seaborn as sns

# Scikit-learn for machine learning

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Loading data from file assigning names to the columns.

data = pd.read_csv(
    './iris/iris.data',
    names=[
        'Sepal Length in cm',
        'Sepal Width in cm',
        'Petal Length in cm',
        'Petal Width in cm',
        'Class'
    ]
)

# Print first 5 lines of the data

data.head()

# Getting the dataframe without the class column to perform analysis later.

data_no_class = data.drop(columns=['Class'])

# Separating the data by class.

setosa_data = data_no_class[0:50]
versicolor_data = data_no_class[50:100]
virginica_data = data_no_class[100:150]

# Separating by feature

sepal_length = data['Sepal Length in cm']
sepal_width = data['Sepal Width in cm']
petal_length = data['Petal Length in cm']
petal_width = data['Petal Width in cm']

# Separating by feature and species.

setosa_petal_length = setosa_data['Petal Length in cm']
setosa_petal_width = setosa_data['Petal Width in cm']
setosa_sepal_length = setosa_data['Sepal Length in cm']
setosa_sepal_width = setosa_data['Sepal Width in cm']

versicolor_petal_length = versicolor_data['Petal Length in cm']
versicolor_petal_width = versicolor_data['Petal Width in cm']
versicolor_sepal_length = versicolor_data['Sepal Length in cm']
versicolor_sepal_width = versicolor_data['Sepal Width in cm']

virginica_petal_length = virginica_data['Petal Length in cm']
virginica_petal_width = virginica_data['Petal Width in cm']
virginica_sepal_length = virginica_data['Sepal Length in cm']
virginica_sepal_width = virginica_data['Sepal Width in cm']

# Getting data description.

description = data.groupby('Class').describe().T

# Exporting to csv file.

description.to_csv('iris.describe', sep=',')

# Print out the description of the data below.

description

# Creating a 2x2 grid of plots

fig, ax = plt.subplots(2, 2)

# Creating basic histograms with no customisation.

ax[0, 0].hist(sepal_length)
ax[0, 1].hist(sepal_width)
ax[1, 0].hist(petal_length)
ax[1, 1].hist(petal_width)

# Show figure.

plt.show()

# Creating a 2 by 2 grid of plots that share a y axis
# with the padding around the plots adjusted.

fig, ax = plt.subplots(2, 2, tight_layout=True, sharey=True)

# Creating histograms in each of the 4 plots.
# The hist function returns 3 elements:
# N: The values of the histogram bins.
# bins: The edges of the bins.
# patches: Container of individual artists used to create the histogram.

N, bins, patches = ax[0, 0].hist(x=sepal_length)

# Setting label for the axis.

ax[0, 0].set_xlabel('Sepal Length in cm', fontsize=12)

# Getting a fraction of the maximum for each bin
# so they can be colour coded later

fracs = N/N.max()

# Setting the colour map to span the full range of values
# by linearly mapping the values in the selected interval
# to values between 0 and 1.

norm = colors.Normalize(fracs.min(), fracs.max())

# Looping through the objects and setting colours for each
# based on the fractions calculater earlier.
# The fraction has been slightly altered to take just a
# section of the colour map so they aren't too dark or too light.

for thisfrac, thispatch in zip(fracs, patches):
    colour = plt.cm.Blues(norm(thisfrac)*0.3+0.3)
    thispatch.set_facecolor(colour)

# Process repeated for each axis.

N, bins, patches = ax[0, 1].hist(x=sepal_width)
ax[0, 1].set_xlabel('Sepal Width in cm', fontsize=12)

fracs = N/N.max()
norm = colors.Normalize(fracs.min(), fracs.max())
for thisfrac, thispatch in zip(fracs, patches):
    colour = plt.cm.Oranges(norm(thisfrac)*0.3+0.3)
    thispatch.set_facecolor(colour)

N, bins, patches = ax[1, 0].hist(x=petal_length)
ax[1, 0].set_xlabel('Petal Length in cm', fontsize=12)

fracs = N/N.max()
norm = colors.Normalize(fracs.min(), fracs.max())
for thisfrac, thispatch in zip(fracs, patches):
    colour = plt.cm.Greens(norm(thisfrac)*0.3+0.3)
    thispatch.set_facecolor(colour)

N, bins, patches = ax[1, 1].hist(x=petal_width)
ax[1, 1].set_xlabel('Petal Width in cm', fontsize=12)

fracs = N/N.max()
norm = colors.Normalize(fracs.min(), fracs.max())
for thisfrac, thispatch in zip(fracs, patches):
    colour = plt.cm.Purples(norm(thisfrac)*0.3+0.3)
    thispatch.set_facecolor(colour)

# Creating a title and setting fontsize and placement.

fig.suptitle('Distribution of Iris Data\nBy Feature', fontsize=16)

# Exporting to png file.

plt.savefig('histogram.png')
plt.show()

# Create 2x2 grid of plots

fig, axs = plt.subplots(2, 2, tight_layout=True)

# Create KDE plots in each quadrant of the figure,
# assigning x values,
# assigning hue by class,
# choosing to fill the plot and
# choosing which colour palette to use.

sns.kdeplot(ax=axs[1, 1], data=data, x='Petal Width in cm',
            hue='Class', fill=True, palette='tab10')
sns.kdeplot(ax=axs[1, 0], data=data, x='Petal Length in cm',
            hue='Class', fill=True, palette='tab10')
sns.kdeplot(ax=axs[0, 1], data=data, x='Sepal Width in cm',
            hue='Class', fill=True, palette='tab10')
sns.kdeplot(ax=axs[0, 0], data=data, x='Sepal Length in cm',
            hue='Class', fill=True, palette='tab10')

# Disabling legend for 3 of the 4 plots

for ax in [axs[0, 0], axs[0, 1], axs[1, 0]]:
    ax.get_legend().set_visible(False)

# Adding a title to the figure

fig.suptitle("Iris Data")

# Creating figure.

fig, ax = plt.subplots(tight_layout=True)

# Creating NumPy array for boxplot.

petal_length_by_species = np.array(
    [setosa_petal_length, versicolor_petal_length, virginica_petal_length]
    )

# Creating boxplot, using transposed array and coloured boxes.

bp = ax.boxplot(petal_length_by_species.T, patch_artist=True)

# Individual boxplot colour settings in returned dictionary.

bp['medians'][0].set_color('tab:blue')
bp['boxes'][0].set_color('tab:blue')
bp['boxes'][0].set_alpha(0.25)
bp['whiskers'][0].set_color('tab:blue')
bp['whiskers'][1].set_color('tab:blue')
bp['caps'][0].set_color('tab:blue')
bp['caps'][1].set_color('tab:blue')
bp['fliers'][0].set_markeredgecolor('tab:blue')
bp['medians'][1].set_color('tab:orange')
bp['boxes'][1].set_color('tab:orange')
bp['boxes'][1].set_alpha(0.25)
bp['whiskers'][2].set_color('tab:orange')
bp['whiskers'][3].set_color('tab:orange')
bp['caps'][2].set_color('tab:orange')
bp['caps'][3].set_color('tab:orange')
bp['fliers'][1].set_markeredgecolor('tab:orange')
bp['medians'][2].set_color('tab:green')
bp['boxes'][2].set_color('tab:green')
bp['boxes'][2].set_alpha(0.25)
bp['whiskers'][4].set_color('tab:green')
bp['whiskers'][5].set_color('tab:green')
bp['caps'][4].set_color('tab:green')
bp['caps'][5].set_color('tab:green')

# Setting labels and titles.

plt.suptitle('Iris Data', fontsize=20, x=0.52)
ax.set_title("Petal Length (cm)", fontsize=12, x=0.51)
ax.set_xlabel("Species", fontsize=12)
ax.set_ylabel("Petal Length (cm)", fontsize=12)

# Labelling each group.

ax.set_xticks([1, 2, 3], ["Setosa", "Versicolor", "Virginica"], fontsize=10)

# Adding grid lines.

ax.grid(axis="y", linestyle="--", alpha=0.7)

# Show figure.

plt.show()

bp = sns.boxplot(data=setosa_petal_length).set(title='Iris Setosa')

# Create a copy of the data to be adjusted

setosa_petal_length_adjusted = setosa_petal_length.copy()

# Creating the necessary variables

median = setosa_petal_length.median()
quantile = setosa_petal_length.quantile([0.25, 0.75])
quantile_25 = quantile[0.25]
quantile_75 = quantile[0.75]
IQR = quantile_75 - quantile_25
min_value = quantile_25 - 1.5 * (IQR)
max_value = quantile_75 + 1.5 * (IQR)

# Getting upper values storing as variable

upper_values = setosa_petal_length[setosa_petal_length >= max_value]

# Removing values from data

for x in upper_values.index:
    setosa_petal_length_adjusted.drop(x, inplace=True)

# Repeat as above for lower values

lower_values = setosa_petal_length[setosa_petal_length <= min_value]

for x in lower_values.index:
    setosa_petal_length_adjusted.drop(x, inplace=True)

# Creating boxplot for new values

sns.boxplot(setosa_petal_length_adjusted).set(title='Iris Setosa')


def remove_outliers(data):
    median = data.median()
    quantile = data.quantile([0.25, 0.75])
    IQR = quantile[0.75] - quantile[0.25]
    min_value = quantile[0.25] - 1.5 * (IQR)
    max_value = quantile[0.75] + 1.5 * (IQR)
    upper_values = data[data >= max_value]
    for x in upper_values.index:
        data.drop(x, inplace=True)
    lower_values = data[data <= min_value]
    for x in lower_values.index:
        data.drop(x, inplace=True)
    return data

# Creating box plot to identify fliers

sns.boxplot(data=versicolor_petal_length).set(title='Iris Versicolor')

# Creating a copy of the dataset

versicolor_petal_length_adjusted = versicolor_petal_length.copy()

# Running script to remove outliers

remove_outliers(versicolor_petal_length_adjusted)

# Plotting boxplot to check if outliers are removed

sns.boxplot(versicolor_petal_length_adjusted).set(title='Iris Versicolor')

# Creating a single plot with adjusted padding.

fig, ax = plt.subplots(tight_layout=True)

# Creating scatter plots for each of the target
# variables in different colours on the same axis.
# Using c= to set marker colour and label for each
# variable rather than using colour mapping.

ax.scatter(setosa_sepal_length, setosa_sepal_width,
           c='tab:blue', label='Setosa', alpha=0.35)
ax.scatter(versicolor_sepal_length, versicolor_sepal_width,
           c='tab:orange', label='Versicolor', alpha=0.35)
ax.scatter(virginica_sepal_length, virginica_sepal_width,
           c='tab:green', label='Virginica', alpha=0.35)

# Setting labels for the axis.

ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Sepal Width (cm)')

# Displaying legend.

ax.legend()

# Setting title for figure and plot.

plt.suptitle('Iris Data', fontsize=20, x=0.535)
plt.title('Sepal Length by Sepal Width', x=0.49)
plt.show()

grid = sns.PairGrid(data, hue='Class', diag_sharey=False, layout_pad=0.5)
grid.map_upper(sns.kdeplot, alpha=0.5)
grid.map_lower(sns.scatterplot, alpha=0.5)
grid.map_diag(sns.kdeplot, fill=True)
grid.add_legend()
grid.figure.suptitle('Iris Flowers by Feature', y=1)

grid = sns.pairplot(data, hue='Class', palette="GnBu_d")

# Finding the pearson correlation values for each feature.

setosa_corr = setosa_data.corr()
versicolor_corr = versicolor_data.corr()
virginica_corr = virginica_data.corr()

# Creating feature name list.

feature_names = ["Sepal Width", "Sepal Length", "Petal Length", "Petal Width"]

# Turning correlation coefficient dataframes into arrays and rounding to 2 decimal places.

array_setosa = np.array(setosa_corr.round(2))
array_versicolor = np.array(versicolor_corr.round(2))
array_virginica = np.array(virginica_corr.round(2))

# Creating 3 subplots.

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 14))

# Setting figure title and layout.

fig.suptitle("Correlation Coefficient Between Features")

# Setting x and y labels for all axes.

for ax in [ax1, ax2, ax3]:
    ax.set_xticks(range(len(feature_names)), labels=feature_names,
                rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(feature_names)), labels=feature_names)

# Creating heatmap and colourbar.
# Setting colourmap, min and max values for colourbar.
# Setting colourbar size and label

im1 = ax1.imshow(array_setosa, cmap='Blues', vmin=-1, vmax=1)
cb1 = ax1.figure.colorbar(im1, fraction=0.046, pad=0.04, label='Pearson Correlation\n Coefficient')

# Labeling heatmap sections.

for i in range(len(feature_names)):
    for j in range(len(feature_names)):
        text = ax1.text(j, i, array_setosa[i, j],
                       ha="center", va="center", color="w")
        
# Adding title for axis.

ax1.set_title("Iris Setosa")

# Repeating above for each species.

im2 = ax2.imshow(array_versicolor, cmap='Oranges', vmin=-1, vmax=1)
cb2 = ax2.figure.colorbar(im2, fraction=0.046, pad=0.04, label='Pearson Correlation\n Coefficient')

for i in range(len(feature_names)):
    for j in range(len(feature_names)):
        text = ax2.text(j, i, array_versicolor[i, j],
                       ha="center", va="center", color="w")
        
ax2.set_title("Iris Versicolor")

im3 = ax3.imshow(array_virginica, cmap='Greens', vmin=-1, vmax=1)
cb3 = ax3.figure.colorbar(im3, fraction=0.046, pad=0.04, label='Pearson Correlation\n Coefficient')

for i in range(len(feature_names)):
    for j in range(len(feature_names)):
        text = ax3.text(j, i, array_virginica[i, j],
                       ha="center", va="center", color="w")
        
ax3.set_title("Iris Virginica")

fig.tight_layout()
plt.show()

# Creating figure and axis

fig, ax = plt.subplots(tight_layout=True)

# Adding Seaborn kernel density plot

sns.kdeplot(data=data, x='Sepal Length in cm', y='Sepal Width in cm',
            hue='Class', alpha=0.15)

# Setting title and axis labels

fig.suptitle('Iris Flowers')
ax.set_title('Sepal Length by Sepal Width')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')

# Setting y limits for a cleaner graph

ax.set_ylim(1.75, 4.75)

# Scatter plots

ax.scatter(setosa_sepal_length, setosa_sepal_width,
           color='tab:blue', alpha=0.5, marker='o')
ax.scatter(versicolor_sepal_length, versicolor_sepal_width,
           color='tab:orange', alpha=0.5, marker='o')
ax.scatter(virginica_sepal_length, virginica_sepal_width,
           color='tab:green', alpha=0.5, marker='o')

# Getting line of best fit polynomial

setosa_sl_sw_poly = np.polynomial.polynomial.Polynomial.fit(
    setosa_sepal_length, setosa_sepal_width, 1
    )
versicolor_sl_sw_poly = np.polynomial.polynomial.Polynomial.fit(
    versicolor_sepal_length, versicolor_sepal_width, 1
    )
virginica_sl_sw_poly = np.polynomial.polynomial.Polynomial.fit(
    virginica_sepal_length, virginica_sepal_width, 1
    )

# Plotting line of best fit

range_sepal_length_sepal_width = np.linspace(4, 8.5, 6)
range_sepal_length_sepal_width_setosa = np.linspace(4, 6.25, 6)

ax.plot(range_sepal_length_sepal_width_setosa,
        setosa_sl_sw_poly(range_sepal_length_sepal_width_setosa),
        label='Setosa', alpha=0.5)
ax.plot(range_sepal_length_sepal_width,
        versicolor_sl_sw_poly(range_sepal_length_sepal_width),
        label='Versicolor', alpha=0.5)
ax.plot(range_sepal_length_sepal_width,
        virginica_sl_sw_poly(range_sepal_length_sepal_width),
        label='Virginica', alpha=0.5)

# Show legend and plot

plt.legend()
plt.show()

sns.boxplot(data=setosa_sepal_width).set(title='Iris Setosa')

# Creating a copy of the data

setosa_sepal_width_adjusted = setosa_sepal_width.copy()

index = setosa_sepal_width[setosa_sepal_width < 2.5].index

setosa_sepal_width_adjusted.drop(index, inplace=True)

setosa_sepal_width_adjusted.shape

sns.boxplot(data=setosa_sepal_width_adjusted).set(title='Iris Setosa')

setosa_sepal_length_adjusted = setosa_sepal_length.copy()

setosa_sepal_length_adjusted.drop(index, inplace=True)

setosa_sepal_length_adjusted.shape

data_adjusted = data.copy()

data_adjusted.drop(index, inplace=True)

data_adjusted.shape

# Creating figure and axis

fig, ax = plt.subplots(tight_layout=True)

# Adding Seaborn kernel density plot

sns.kdeplot(data=data_adjusted, x='Sepal Length in cm', y='Sepal Width in cm',
            hue='Class', alpha=0.15)

# Setting title and axis labels

fig.suptitle('Iris Flowers')
ax.set_title('Sepal Length by Sepal Width')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')

# Setting y limits for a cleaner graph

ax.set_ylim(1.75, 4.75)

# Scatter plots

ax.scatter(setosa_sepal_length_adjusted, setosa_sepal_width_adjusted,
           color='tab:blue', alpha=0.5, marker='o')
ax.scatter(versicolor_sepal_length, versicolor_sepal_width,
           color='tab:orange', alpha=0.5, marker='o')
ax.scatter(virginica_sepal_length, virginica_sepal_width,
           color='tab:green', alpha=0.5, marker='o')

# Getting line of best fit polynomial

setosa_sl_sw_poly = np.polynomial.polynomial.Polynomial.fit(
    setosa_sepal_length_adjusted, setosa_sepal_width_adjusted, 1
    )
versicolor_sl_sw_poly = np.polynomial.polynomial.Polynomial.fit(
    versicolor_sepal_length, versicolor_sepal_width, 1
    )
virginica_sl_sw_poly = np.polynomial.polynomial.Polynomial.fit(
    virginica_sepal_length, virginica_sepal_width, 1
    )

# Plotting line of best fit

range_sepal_length_sepal_width = np.linspace(4, 8.5, 6)
range_sepal_length_sepal_width_setosa = np.linspace(4, 6.25, 6)

ax.plot(range_sepal_length_sepal_width_setosa,
        setosa_sl_sw_poly(range_sepal_length_sepal_width_setosa),
        label='Setosa', alpha=0.5)
ax.plot(range_sepal_length_sepal_width,
        versicolor_sl_sw_poly(range_sepal_length_sepal_width),
        label='Versicolor', alpha=0.5)
ax.plot(range_sepal_length_sepal_width,
        virginica_sl_sw_poly(range_sepal_length_sepal_width),
        label='Virginica', alpha=0.5)

# Show legend and plot

plt.legend()
plt.show()

# Separating the features and target classes

data_values = data.values
X = data_values[:, 0:4]
Y = data_values[:, 4]

# Split the data to train and test dataset.

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Support vector machine algorithm

svn = SVC()
svn.fit(X_train, y_train)

# Predict from the test dataset

predictions = svn.predict(X_test)

# Calculate the accuracy

accuracy_score(y_test, predictions)

# Print the accuracy score

print(classification_report(y_test, predictions))

# Inputting test values to predict the species.
# The values are in the same order as the columns in the dataset.

X_new = np.array([[3, 2, 1, 0.2], [4.9, 2.2, 3.8, 1.1], [5.3, 2.5, 4.6, 1.9]])

# Prediction of the species from the input vector

prediction = svn.predict(X_new)
print(f"Prediction of Species: {prediction}")