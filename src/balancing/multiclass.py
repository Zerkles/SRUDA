from multi_imbalance.utils.plot import plot_visual_comparision_datasets

from src.balancing.utilities import resample_and_write_to_csv, path_balanced_csv, \
    check_if_new_categorical_features_generated
import matplotlib.pyplot as plt


def global_cs_optimized(X, y):
    from multi_imbalance.resampling.global_cs import GlobalCS

    obj = GlobalCS()
    resample_and_write_to_csv(obj, X, y, "GlobalCS" + str(obj.get_params()))


def mdo_optimized(X, y):
    from multi_imbalance.resampling.mdo import MDO

    obj = MDO()
    resample_and_write_to_csv(obj, X, y, "MDO" + str(obj.get_params()))


def soup_optimized(X, y):
    from multi_imbalance.resampling.soup import SOUP

    obj = SOUP(maj_int_min={
        'maj': [0],  # indices of majority classes
        'min': [1],  # indices of minority classes
    })
    # resample_and_write_to_csv(obj, X, y, "SOUP" + str(obj.get_params()))
    resample_and_write_to_csv(obj, X, y, "SOUP" + str({"k": obj.k, "shuffle": obj.shuffle}))


def spider3_optimized(X, y):
    from multi_imbalance.resampling.spider import SPIDER3

    obj = SPIDER3(k=5)
    resample_and_write_to_csv(obj, X, y, "SPIDER3" + str(obj.get_params()))


def static_smote_optimized(X, y):
    from multi_imbalance.resampling.static_smote import StaticSMOTE
    # it has no parameters to set and no fit_resample()

    obj = StaticSMOTE()

    X['Sales'] = y
    X_resampled, y_resampled = obj.fit_transform(X, y)
    check_if_new_categorical_features_generated(X, X_resampled)

    write_df = X_resampled
    write_df["Sales"] = y_resampled
    write_df.to_csv(path_balanced_csv + "/" + "StaticSMOTE" + ".csv", index=False)
    print("Balanced:", "StaticSMOTE", '\n')
    # TODO: Coś się z danymi odpierdala i złe csv wychodzą

    return X_resampled, y_resampled


def draw_plot(X, y, X_resampled, y_resampled, name):
    plot_visual_comparision_datasets(X, y, X_resampled, y_resampled, 'CriteoCS', 'Resampled CriteoCS with ' + name)
    plt.show()
    # plt.savefig("/graphs"+name+".png")


def balance_all_multiclass(X, y):
    # import warnings
    # warnings.filterwarnings("ignore")
    print("Multi-imbalanced methods comparsion:")

    global_cs_optimized(X, y)
    # mdo_optimized(X, y) # UWAGA! Generuje nowe dane dla cech kategorycznych!
    soup_optimized(X, y)
    # spider3_optimized(X, y) # UWAGA! Generuje nowe dane dla cech kategorycznych!
    # static_smote_optimized(X, y)  # Nie działa to i nie wiem czemu
