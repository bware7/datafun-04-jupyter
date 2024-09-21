# datafun-04-jupyter
Project 4 uses a combination of Python and Markdown to create an initial data story in a Jupyter Notebook. The project includes a project virtual environment with popular libraries for data analytics including pandas, matplotlib, and seaborn, and introduces a common process for starting exploratory data analysis projects.

# Create and Activate Project Virtual Environment
```
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r "requirements.txt"

git add .
git commit -m "initial commit"
git push -u origin main
```

# Install Dependencies
```
Add the following to requirements.txt

jupyterlab 
numpy 
pandas
matplotlib 
seaborn 
scipy

Install the packages listed in the requirements file with this command:

py -m pip install -r requirements.txt
```

# Edit and Execute Notebook

Created FirstNotebook.ipynb and TestDrive.ipynb with some basic fuctions and cells as an example.

# Install and Set Up Jupyter in VS Code
Follow these steps to set up Jupyter Notebook in Visual Studio Code (VS Code):

Install the Jupyter Extension:

Open VS Code.
Go to the Extensions view by clicking on the square icon on the left sidebar or pressing Ctrl+Shift+X.
Search for "Jupyter" and install the official Jupyter extension.

Open the Project Folder:

Open your root project repository folder in VS Code (e.g., datafun-04-jupyter).

Select the Python Interpreter:

Press Ctrl+Shift+P to open the command palette.
Type "Python: Select Interpreter" and press Enter.
Choose the interpreter from your virtual environment (.venv).

Create and Open the Notebook:

In the VS Code Explorer, create a new file named yourname_eda.ipynb (e.g., BinWare_eda.ipynb).
Ensure the file has the .ipynb extension.
Double-click the notebook file to open it for editing.

Add an Introduction:

Add a Markdown cell at the top of your notebook with the title, author, date, and purpose of the project.

# Steps for Exploratory Data Analysis (EDA)
Perform exploratory data analysis using pandas and other tools as needed. Below are the steps followed in this project:

1. Import Dependencies
```
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import scipy
    import pyarrow
```

2. Data Acquisition

```
# Load the Iris dataset into a DataFrame
df = sns.load_dataset('iris')
```

3. Initial Data Inspection

```
# Display the first 10 rows of the DataFrame
print(df.head(10))

# Check the shape of the DataFrame (rows, columns)
print(f"Shape of the DataFrame: {df.shape}")

# Display the data types of each column
print("Data types of each column:")
print(df.dtypes)
```

4. Initial Descriptive Statistics

```
# Display summary statistics
print(df.describe())
```

5. Initial Data Distribution for Numerical Columns

```
# Plot histograms for all numerical columns
df.hist(figsize=(12, 10), bins=15)
plt.suptitle('Histograms of Numerical Columns')
plt.show()
```

6. Initial Data Distribution for Categorical Columns

```
# Inspect value counts for the 'species' column
print(df['species'].value_counts())

# Visualize the distribution using a count plot
sns.countplot(x='species', data=df)
plt.title('Distribution of Species')
plt.show()
```

7. Initial Data Transformation and Feature Engineering

```
# Rename a column for better readability
df.rename(columns={'sepal_length': 'Sepal Length'}, inplace=True)

# Add a new column (e.g., Sepal Area)
df['Sepal Area'] = df['Sepal Length'] * df['sepal_width']

# Display the first few rows to show the new column
print(df.head())
```

8. Initial Visualizations

Pair Plot:

```
# Pair plot to show relationships between features, colored by species
sns.pairplot(df, hue='species')
plt.show()
```

Correlation Heatmap:

```
# Compute correlation matrix on numerical columns only
corr = df.select_dtypes(include='number').corr()

# Generate a heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

Box Plot:

```
# Box plot of sepal length by species
plt.figure(figsize=(8,6))
sns.boxplot(x='species', y='Sepal Length', data=df)
plt.title('Sepal Length by Species')
plt.show()
```

Violin Plot:

```
# Violin plot of petal width by species
plt.figure(figsize=(8,6))
sns.violinplot(x='species', y='petal_width', data=df)
plt.title('Petal Width by Species')
plt.show()
```

9. Initial Storytelling and Presentation

# Observations:

The dataset is balanced with 50 samples for each species.
Petal measurements are key differentiators among species.
Strong correlations exist between certain features.
The new feature Sepal Area may aid in species classification.
Visualizations confirm that species can be distinguished using feature combinations.

# Conclusion:

The EDA indicates that the Iris dataset is suitable for predictive modeling. Petal dimensions are particularly useful for classifying species. Future work could involve applying machine learning algorithms to develop classification models.
