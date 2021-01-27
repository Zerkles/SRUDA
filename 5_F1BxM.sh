#ODPALIĆ Z MODELEM Z PARAMETRAMI, MODEL BALANSOWANIA BEZ PARAMETRÓW (INNY BRANCH)
python3 -W ignore main.py -m xgb -m cat -m reg -m forest -b none \
 -b ros -b rus -b smotenc -b ros -b rus -b nearmiss -b enn -b renn -b allknn -b onesided \
 -b ncr -b iht -b globalcs -b soup -i data/no_price_feature_selected/imbalance_set_no_price.csv \
 -uf data/no_price_feature_selected/test_set_no_price.csv \
 -bd data/balanced_csv