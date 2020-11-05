import fileinput

for line in fileinput.input("Criteo_Conversion_Search/CriteoSearchData", inplace=True):
    print(str(line).replace(' ', '\t'), end='')
