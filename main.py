import sklearn


def do_preprocessing():
    pass


def main():
    guwno = None    # TODO:DUPA
    # preprocessing
    # preprocessing w sumie jest tylko raz, więc jak Julek zbalansuje i zapisze do plików to tylko ścieżki plików są nam potrzebne
    # ---------------------
    # balancing
    # potrzebuję pliku csv z danymi
    # zwracam kilka plików csv ze zbalansowanymi danymi, żeby nie trzeba było tego uczyć od zera

    # ---------------------

    # model building
    # ja potrzebuje pliku csv z separatorem
    # będę zwracał to co będę mógł do późniejszej oceny

    d = {
        "filename": "data_set_1",
        "time": 24.3,
        "TP": 100,
        "TN": 100,
        "FP": 10,
        "FN": 10,
        "score": (0, 2, 4, 4)
    }

    # ---------------------

    # scores
    # dla mnie truth table jak dostanę to jestem w domu
    # auROC zobacze co tam jeszcze do zrobienia
    return guwno


if __name__ == "__main__":
    for i in range(69):
        print("dupa")

    main()

#     SCENARIUSZ 1 - GÓWNO BEZ BALANSOWANIA
#     julek nic nie processuje, nieprzerobiony dataset leci do Lulka
#     elo lulek też nic nie balansuje
#     pioter odpala learning
#     ja robie scoring i sie okazuje, że balancing jest potrzebny, bo inaczej mamy straty

#     SCENARIUSZ 2 - BALANSING ALE BEZ WYBIERANA FICZURÓW
#     julek nic nie processuje, nieprzerobiony dataset leci do Lulka
#     lulek balansuje to wszystko
#     pioter odpala learning
#     ja robie scoring


#     SCENARIUSZ 3 - ALL INCLUDED
#     julek processuje
#     lulek balansuje
#     wszystko se robimy
#     happy ending
