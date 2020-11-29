from train_score_test import train_and_score, percent_change


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


def spider(X, y):
    from multi_imbalance.resampling.spider import SPIDER3

    spider_obj = SPIDER3(k=5)

    return spider_obj.fit_resample(X, y)


def static_smote(X, y):
    from multi_imbalance.resampling.static_smote import StaticSMOTE

    static_smote_obj = StaticSMOTE()

    return static_smote_obj.fit_transform(X, y)


def compare_multi_class_methods(X, y, score_original):
    print("GlobalCs:")
    X_resampled, y_resampled = global_cs(X, y)
    # print(sorted(Counter(y_resampled).items()))
    score = train_and_score(X_resampled, y_resampled)
    print("Score:", score)
    print("Percent Change:", percent_change(score_original, score), "\n")

    print("MDO:")
    X_resampled, y_resampled = mdo(X, y)
    # print(sorted(Counter(y_resampled).items()))
    score = train_and_score(X_resampled, y_resampled)
    print("Score:", score)
    print("Percent Change:", percent_change(score_original, score), "\n")

    print("SOUP:")
    X_resampled, y_resampled = soup(X, y)
    # print(sorted(Counter(y_resampled).items()))
    score = train_and_score(X_resampled, y_resampled)
    print("Score:", score)
    print("Percent Change:", percent_change(score_original, score), "\n")

    print("SPIDER3:")
    X_resampled, y_resampled = spider(X, y)
    # print(sorted(Counter(y_resampled).items()))
    score = train_and_score(X_resampled, y_resampled)
    print("Score:", score)
    print("Percent Change:", percent_change(score_original, score), "\n")

    print("StaticSMOTE:")
    X_resampled, y_resampled = static_smote(X, y)
    # print(sorted(Counter(y_resampled).items()))
    score = train_and_score(X_resampled, y_resampled)
    print("Score:", score)
    print("Percent Change:", percent_change(score_original, score), "\n")
