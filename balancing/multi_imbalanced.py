from balancing.utilities import resample_and_write_to_csv_multiclass


def global_cs_variations(X, y):
    from multi_imbalance.resampling.global_cs import GlobalCS

    obj = GlobalCS
    print(obj)
    print(obj.fit_resample(X,y))
    resample_and_write_to_csv_multiclass(obj, X, y, "GlobalCS" + str(obj.get_params()))


def mdo_variations(X, y):
    from multi_imbalance.resampling.mdo import MDO

    obj = MDO()
    resample_and_write_to_csv_multiclass(obj, X, y, "MDO" + str(obj.get_params()))


def soup_variations(X, y):
    from multi_imbalance.resampling.soup import SOUP

    obj = SOUP(maj_int_min={
        'maj': y,  # indices of majority classes
        'min': y  # indices of minority classes
    })

    resample_and_write_to_csv_multiclass(obj, X, y, "SOUP" + str(obj.get_params()))


def spider3_variations(X, y):
    from multi_imbalance.resampling.spider import SPIDER3

    obj = SPIDER3(k=5)
    resample_and_write_to_csv_multiclass(obj, X, y, "SPIDER3" + str(obj.get_params()))


def static_smote_variations(X, y):
    from multi_imbalance.resampling.static_smote import StaticSMOTE

    obj = StaticSMOTE()
    resample_and_write_to_csv_multiclass(obj, X, y, "StaticSMOTE")  # it has no parameters to set


def balance_all_multiclass(X, y, cores_count):
    import warnings
    warnings.filterwarnings("ignore")
    print("Multi-imbalanced methods comparsion:")

    global_cs_variations(X, y)
    mdo_variations(X, y)
    soup_variations(X, y)
    # spider3_variations(X, y)
    static_smote_variations(X, y)
