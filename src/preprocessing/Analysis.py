import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


class Analysis:
    """description of class"""
    def calculate_mean(self,row):
        print("avg")
        data=(pd.to_numeric(row, errors='coerce'))
        print(np.mean(data))
        data=data[(data!=-1.0) & (data!=-0.0)] 
        mean=np.mean(data)
        return mean

    def calculate_median(self,row):
        print("median")
        data=(pd.to_numeric(row, errors='coerce'))
        print(np.median(data))
        data=data[(data!=-1.0) & (data!=-0.0)] 
        mean=np.median(data)
        return mean
    def count_unique(self,row):
        print("number of unique values")
        data=(pd.to_numeric(row, errors='coerce'))
        
        return len(np.unique(data))


    def analyze_data(self,data):
        matrix=np.transpose(np.delete(data,0,0))
        #print(matrix)
        print("saleamountineuro")
        print(self.count_unique(matrix[2]))
        print(selif.calculate_median(matrix[2]))
        print(selif.calculate_mean(matrix[2]))
        print("time_delay_for_conversion")
        print(self.count_unique(matrix[3]))
        print(selif.calculate_median(matrix[3]))
        print(selif.calculate_mean(matrix[3]))
        print("nb_lcicks_1week")
        print(self.count_unique(matrix[5]))
        print(selif.calculate_median(matrix[5]))
        print(selif.calculate_mean(matrix[5]))
        print("product_price")
        print(self.count_unique(matrix[6]))
        print(selif.calculate_median(matrix[6]))
        print(selif.calculate_mean(matrix[6]))

    def displot(self,matrix,columns):
        matrix=np.delete(matrix,0,0)
        print(matrix)
        plot_data=(pd.to_numeric([matrix[:,columns[0]]], errors='coerce'))
        print(plot_data[0])
        df = pd.DataFrame(data=plot_data[0],columns=["Sale"])
        #print(df)
        sns.displot(data=df, x="Sale")
        plt.show()


