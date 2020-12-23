def global_cs(X, y):
    from multi_imbalance.resampling.global_cs import GlobalCS

    global_cs_obj = GlobalCS()

    return global_cs_obj.fit_resample(X, y)


def mdo(X, y):
    from multi_imbalance.resampling.mdo import MDO

    mdo_obj = MDO()

    return mdo_obj.fit_resample(X, y)


def soup(X, y):
    from multi_imbalance.resampling.soup import SOUP

    soup_obj = SOUP(maj_int_min={
        'maj': y,  # indices of majority classes
        'min': y  # indices of minority classes
    })

    return soup_obj.fit_resample(X, y)


def spider3(X, y):
    from multi_imbalance.resampling.spider import SPIDER3

    spider_obj = SPIDER3(k=5)

    return spider_obj.fit_resample(X, y)


def static_smote(X, y):
    from multi_imbalance.resampling.static_smote import StaticSMOTE

    static_smote_obj = StaticSMOTE()

    return static_smote_obj.fit_transform(X, y)
