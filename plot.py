import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import numpy as np


def plot_line_chart(
    data: pd.DataFrame,
    column_to_plot: str,
    index_col: str,
    title: str,
):
    plt.figure(figsize=(20, 10))
    plt.plot(data[column_to_plot], label=column_to_plot)
    plt.title(title)
    plt.xlabel(column_to_plot)
    plt.ylabel(index_col)
    plt.legend()
    plt.show()


def plot_boxplot(data: pd.DataFrame, column_to_plot: str, title: str):
    plt.figure(figsize=(20, 10))
    sns.boxplot(x=data[column_to_plot])
    plt.title(title)
    plt.show()


def plot_scatterplot(data: pd.DataFrame, x_col: str, y_col: str):
    plt.figure(figsize=(20, 10))
    plt.scatter(x=data[x_col], y=data[y_col], alpha=0.5)
    plt.title(f"{x_col} vs {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()


def plot_histogram(data: pd.DataFrame, main_column: str, transformed_column: str):
    fig, ax = plt.subplots(2, 2, figsize=(20, 10))

    # Q-Q plot for original data
    stats.probplot(data[main_column], dist="norm", plot=ax[0, 0])
    ax[0, 0].set_title("Q-Q Plot of Original Data")

    # Plot histogram of original data with KDE
    sns.histplot(
        data[main_column],
        bins=30,
        kde=False,
        color="skyblue",
        stat="density",
        ax=ax[0, 1],
    )
    sns.kdeplot(data[main_column], color="darkblue", linewidth=2, ax=ax[0, 1])
    ax[0, 1].set_title("Histogram and KDE of Original Data")

    # Q-Q plot for transformed data
    stats.probplot(data[transformed_column], dist="norm", plot=ax[1, 0])
    ax[1, 0].set_title("Q-Q Plot of Transformed Data")

    # Plot histogram of transformed data with KDE
    sns.histplot(
        data[transformed_column],
        bins=30,
        kde=False,
        color="skyblue",
        stat="density",
        ax=ax[1, 1],
    )
    sns.kdeplot(data[transformed_column], color="darkblue", linewidth=2, ax=ax[1, 1])
    ax[1, 1].set_title(f"Histogram and KDE of {transformed_column}")

    plt.tight_layout()
    plt.show()


def visual_inspection(
    data: pd.DataFrame,
    column_to_plot: str,
    index_col: str,
    title: str,
    comparison_col: str,
):
    data.set_index(index_col, inplace=True)

    plot_line_chart(
        data=data,
        column_to_plot=column_to_plot,
        index_col=index_col,
        title=title,
    )

    plot_boxplot(
        data=data,
        column_to_plot=column_to_plot,
        title=title,
    )

    plot_scatterplot(
        data=data,
        x_col=comparison_col,
        y_col=column_to_plot,
    )
