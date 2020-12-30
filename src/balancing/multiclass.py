from src.balancing.utilities import resample_and_write_to_csv, path_balanced_csv


def global_cs_variations(X, y):
    from multi_imbalance.resampling.global_cs import GlobalCS

    obj = GlobalCS()
    # print(obj)
    # print(obj.fit_resample(X, y))
    resample_and_write_to_csv(obj, X, y, "GlobalCS" + str(obj.get_params()))

    obj = GlobalCS(shuffle=False)
    # print(obj)
    # print(obj.fit_resample(X, y))
    resample_and_write_to_csv(obj, X, y, "GlobalCS" + str(obj.get_params()))


def mdo_variations(X, y):
    from multi_imbalance.resampling.mdo import MDO

    obj = MDO()
    resample_and_write_to_csv(obj, X, y, "MDO" + str(obj.get_params()))


def soup_variations(X, y):
    from multi_imbalance.resampling.soup import SOUP

    obj = SOUP(maj_int_min={
        'maj': [0],  # indices of majority classes
        'min': [1],  # indices of minority classes
    })
    # resample_and_write_to_csv(obj, X, y, "SOUP" + str(obj.get_params()))
    resample_and_write_to_csv(obj, X, y, "SOUP" + str({"k": obj.k, "shuffle": obj.shuffle}))


def spider3_variations(X, y):
    from multi_imbalance.resampling.spider import SPIDER3

    obj = SPIDER3(k=5)
    resample_and_write_to_csv(obj, X, y, "SPIDER3" + str(obj.get_params()))


def static_smote_variations(X, y):
    from multi_imbalance.resampling.static_smote import StaticSMOTE
    # it has no parameters to set and no fit_resample()

    X.append(y)
    X_resampled, y_resampled = StaticSMOTE().fit_transform(X, y)

    write_df = X_resampled.append(y_resampled)
    write_df.to_csv(path_balanced_csv + "/StaticSMOTE.csv")
    print("Balanced:", "StaticSMOTE")

    return X_resampled, y_resampled


def balance_all_multiclass(X, y):
    import warnings
    warnings.filterwarnings("ignore")
    print("Multi-imbalanced methods comparsion:")

    global_cs_variations(X, y)
    mdo_variations(X, y)
    soup_variations(X, y)
    spider3_variations(X, y)
    # static_smote_variations(X, y) Nie działa to i nie wiem czemu
    # TODO: Coś się z danymi odpierdala i złe csv wychodzą

    # plot_visual_comparision_datasets(X, y, X_resampled, y_resampled, 'CriteoCS', 'Resampled CriteoCS with ' + name)
    # plt.show()
    # # plt.savefig("/graphs"+name+".png")
