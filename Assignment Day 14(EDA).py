
#EDA Assignment - Day 14

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def locate_csv():
    
    r"C: \Users\Sarvesh pal\OneDrive\Desktop\INTERNSHIP\Cars_ data.csv"
    if len(sys.argv) > 1:
        path = Path(sys.argv[1]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")
        return path

    folder = Path(__file__).resolve().parent
    candidates = [
        folder / "data.csv",
        folder / "cars.csv",
        folder / "Car_dataset.csv",
        folder / "car data.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path

    csv_files = list(folder.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    raise FileNotFoundError(
        "Could not locate the Cars dataset CSV. Put the CSV beside this script "
        "or run: python EDA_Assignment_Day_14.py path/to/cars.csv"
    )


def main():
    
    # Load the dataset and inspect it
    
    csv_path = locate_csv()
    df = pd.read_csv(csv_path)

    print(f"\nLoaded dataset: {csv_path}")
    print("\nFirst five rows:")
    print(df.head())

    print("\nData types and non-null counts:")
    df.info()


    # Drop irrelevant columns
    
    cols_to_drop = [
        "Engine Fuel Type",
        "Market Category",
        "Vehicle Style",
        "Popularity",
        "Number of Doors",
        "Vehicle Size",
    ]

    # Drop only columns that are present, so minor dataset variations work.
    existing_drop_cols = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_drop_cols)

    print("\nDataframe after dropping irrelevant columns:")
    print(df.head())

    
    # Rename columns
    
    rename_cols = {
        "Engine HP": "HP",
        "Engine Cylinders": "Cylinders",
        "Transmission Type": "Transmission",
        "Driven_Wheels": "Drive Mode",
        "highway MPG": "Highway MPG",
        "city mpg": "City MPG",
        "MSRP": "Price",
    }
    df = df.rename(columns=rename_cols)

    print("\nDataframe after renaming columns:")
    print(df.head())

    
    # Remove duplicate rows
    
    rows_before = len(df)
    print(f"\nNumber of rows before removing duplicates: {rows_before}")

    df = df.drop_duplicates()

    print("\nDataframe after removing duplicates:")
    print(df.head())
    print(f"Number of rows after removing duplicates: {len(df)}")
    print(f"Duplicate rows removed: {rows_before - len(df)}")

    
    # Remove missing values
    
    print("\nMissing values in each column before removal:")
    print(df.isnull().sum())

    df = df.dropna()

    print("\nMissing values after removal:")
    print(df.isnull().sum())

    print("\nDescriptive statistics:")
    print(df.describe())

    
    # Outlier exploration: boxplots for Price and HP
    
    if "Price" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df["Price"])
        plt.title("Boxplot of Price")
        plt.xlabel("Price")
        plt.tight_layout()
        plt.show()

    if "HP" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df["HP"])
        plt.title("Boxplot of HP")
        plt.xlabel("HP")
        plt.tight_layout()
        plt.show()

    # Print numeric columns and save their names in list l.
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    print("\nNumeric columns:")
    print(numeric_df)
    l = numeric_df.columns.tolist()
    print("\nList l (numeric column names):")
    print(l)

    
    # Outlier removal using the IQR method
    
    if not l:
        raise ValueError("No numeric columns found for IQR outlier detection.")

    Q1 = df[l].quantile(0.25)
    Q3 = df[l].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Keep rows that are within the IQR bounds for every numeric feature.
    df2 = df[~((df[l] < lower_bound) | (df[l] > upper_bound)).any(axis=1)].copy()

    print("\nQ1:")
    print(Q1)
    print("\nQ3:")
    print(Q3)
    print("\nIQR:")
    print(IQR)
    print(f"\nShape of df before IQR outlier removal: {df.shape}")
    print(f"Shape of df2 after IQR outlier removal: {df2.shape}")

    # Unique values and their counts for each column in df.
    print("\nUnique value counts for each column:")
    for column in df.columns:
        print(f"\n--------------- {column} ----------------")
        print(df[column].value_counts())

    
    # Univariate distributions
    
    if "HP" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x="HP", kde=True)
        plt.title("Distribution of HP")
        plt.tight_layout()
        plt.show()

    # Plot numeric columns in a 2x3 subplot grid, as requested.
    plot_cols = l[:6]
    if plot_cols:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for index, column in enumerate(plot_cols):
            sns.histplot(data=df, x=column, kde=True, ax=axes[index])
            axes[index].set_title(f"Distribution of {column}")

        for index in range(len(plot_cols), len(axes)):
            axes[index].set_visible(False)

        plt.tight_layout()
        plt.show()

    
    # Bar chart: most frequent car makes
    
    if "Make" in df.columns:
        make_counts = df["Make"].value_counts().nlargest(20)
        plt.figure(figsize=(12, 8))
        make_counts.plot(kind="bar")
        plt.title("Top 20 Car Makes by Number of Cars")
        plt.xlabel("Make")
        plt.ylabel("Number of Cars")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    
    # Count plot: Transmission with Drive Mode as hue
    
    if {"Transmission", "Drive Mode"}.issubset(df.columns):
        plt.figure(figsize=(15, 5))
        sns.countplot(data=df, x="Transmission", hue="Drive Mode")
        plt.title("Transmission Count by Drive Mode")
        plt.xlabel("Transmission")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    
    # Bivariate distribution: HP vs Price
    
    if {"HP", "Price"}.issubset(df.columns):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="HP", y="Price")
        plt.title("Relationship between HP and Price")
        plt.tight_layout()
        plt.show()

        print("\nCorrelation between HP and Price:")
        print(df[["HP", "Price"]].corr().iloc[0, 1])

    
    # Aggregated values across categories
    
    if {"Cylinders", "Price"}.issubset(df.columns):
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Cylinders", y="Price")
        plt.title("Mean Price by Number of Cylinders")
        plt.xlabel("Cylinders")
        plt.ylabel("Mean Price")
        plt.tight_layout()
        plt.show()

    # Plot Transmission categories on the y-axis.
    if "Transmission" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, y="Transmission")
        plt.title("Number of Cars by Transmission")
        plt.xlabel("Count")
        plt.ylabel("Transmission")
        plt.tight_layout()
        plt.show()

    
    # Multivariate analysis: correlation matrix and heatmap
    
    corr = df.select_dtypes(include=[np.number]).corr()
    print("\nCorrelation matrix:")
    print(corr)

    if not corr.empty:
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, cmap="BrBG", annot=True, fmt=".2f")
        plt.title("Correlation Heatmap of Numeric Features")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
