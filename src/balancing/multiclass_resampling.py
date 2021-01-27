from multi_imbalance.resampling.global_cs import GlobalCS
from multi_imbalance.resampling.soup import SOUP
from multi_imbalance.utils.plot import plot_visual_comparision_datasets
import pandas as pd

import matplotlib.pyplot as plt


def global_cs_optimized():
    return GlobalCS()


def soup_optimized():
    return SOUP()


def draw_plot(X: pd.DataFrame, y: pd.DataFrame, X_resampled: pd.DataFrame, y_resampled: pd.DataFrame, name: str,
              save_plot: bool = False):
    plot_visual_comparision_datasets(X, y, X_resampled, y_resampled, 'CriteoSD', 'Resampled CriteoSD with ' + name)
    plt.show()
    if save_plot:
        plt.savefig("/graphs" + name + ".png")
