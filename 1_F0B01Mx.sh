#ODPALIĆ Z MODELEM BEZ PARAMETRÓW
python3 -W ignore main.py -m xgb -m cat -m reg -m forest \
 -b none -b smotenc -b ros -b rus -b nearmiss -b enn -b renn -b allknn -b onesided \
 -b ncr -b iht -b globalcs -b soup \
 -i data/criteo/100k_no_feature_importance.csv -uf data/criteo/60k_no_feature_importance.csv -bd data/balanced_csv
#możliwe, że trzeba będzie wyrzucić smotenc