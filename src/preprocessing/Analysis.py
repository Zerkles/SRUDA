import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


class Analysis:
    """description of class"""
    def calculate_mean(self,row):
        print("avg")
        data=(pd.to_numeric(row, errors='coerce'))
        print(np.mean(data))
        data=data[(data!=-1.0) & (data!=-0.0)&(data!=-1) & (data!=-0)] 
        mean=np.mean(data)
        return mean

    def calculate_median(self,column):
        print("median")
        data=(pd.to_numeric(column, errors='coerce'))
        print(np.median(data))
        data=data[(data!=-1.0) & (data!=-0.0)] 
        mean=np.median(data)
        return mean
    def count_unique(self,column):
        print("number of unique values")
        data=(pd.to_numeric(column, errors='coerce'))
        
        return len(np.unique(data))
    def count_mode(self,column):
        print("Mode:")
        m = stats.mode(column)
        print(m)
        return m


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
        matrix_temp=np.delete(matrix,0,0)
        print(matrix_temp)
        plot_data=(pd.to_numeric(matrix_temp[:,columns[0]], errors='coerce'))
        df = pd.DataFrame(data=plot_data,columns=["Sale"])
        print(df)
        sns.displot(data=df, x="Sale",bins=2)
        plt.show()

    def sale_displot(self,matrix):
        matrix_temp=np.delete(matrix,0,0)
        plot_data=(pd.to_numeric(matrix_temp[:,1], errors='coerce'))
        df = pd.DataFrame(data=plot_data,columns=["Sale"])
        sns.displot(data=df, x="Sale",bins=2)
        plt.show()

    def SalesAmountInEuro_displot(self,matrix):
        matrix_temp=np.delete(matrix,0,0)
        matrix_temp=np.delete(matrix_temp,np.where(matrix_temp[:,2]=='-1.0')[0],0)
        print(matrix_temp)
        plot_data=(pd.to_numeric(matrix_temp[:,2], errors='coerce'))
        df = pd.DataFrame(data=plot_data,columns=["SalesAmountInEuro"])
        print(df)
        sns.displot(data=df, x="SalesAmountInEuro")
        plt.xlim(1,1200)
        plt.ylim(0,50000)
        plt.show()

    def time_delay_for_conversion_displot(self,matrix):
        matrix_temp=np.delete(matrix,0,0)
        matrix_temp=np.delete(matrix_temp,np.where(matrix_temp[:,3]=='-1')[0],0)
        #print(matrix_temp)
        plot_data=(pd.to_numeric(matrix_temp[:,3], errors='coerce'))
        df = pd.DataFrame(data=plot_data,columns=["time_delay_for_conversion"])
        #print(df)
        sns.displot(data=df, x="time_delay_for_conversion",bins=100000)
        plt.xlim(1,6000)
        plt.ylim(0,30000)
        plt.show()

    def click_timestamp_displot(self,matrix):
        matrix_temp=np.delete(matrix,0,0)
        #print(matrix_temp)
        plot_data=(pd.to_numeric(matrix_temp[:,4], errors='coerce'))
        df = pd.DataFrame(data=plot_data,columns=["click_timestamp"])
        #print(df)
        sns.displot(data=df, x="click_timestamp",bins=200)
        plt.show()

    def nb_clicks_1week_displot(self,matrix):
        matrix_temp=np.delete(matrix,0,0)
        matrix_temp=np.delete(matrix_temp,np.where(matrix_temp[:,5]=='-1')[0],0)
        #print(matrix_temp)
        plot_data=(pd.to_numeric(matrix_temp[:,5], errors='coerce'))
        df = pd.DataFrame(data=plot_data,columns=["nb_clicks_1week"])
        print(df)
        sns.displot(data=df, x="nb_clicks_1week",bins=6000)
        plt.xlim(0,600)
        #plt.ylim(0,800000)
        plt.show()

    def product_price_displot(self,matrix):
        matrix_temp=np.delete(matrix,0,0)
        #matrix_temp=np.delete(matrix_temp,np.where(matrix_temp[:,6]=='0.0')[0],0)
        #print(matrix_temp)
        plot_data=(pd.to_numeric(matrix_temp[:,6], errors='coerce'))
        df = pd.DataFrame(data=plot_data,columns=["product_price"])
        print(df)
        sns.displot(data=df, x="product_price",bins=4000)
        plt.xlim(0,800)
        #plt.ylim(0,20)
        plt.show()

    def correlation(self, matrix):
         matrix_temp=matrix
         
         dataset=pd.DataFrame(matrix_temp[:,1:],matrix_temp[:,0],matrix_temp[0,1:])
         #dataset=pd.DataFrame(matrix_temp[:,[1,2,3,4,5,6]],matrix_temp[:,0],matrix_temp[0,[1,2,3,4,5,6]])
         dataset = dataset.apply(pd.to_numeric, errors='coerce', axis=1)
         #print(dataset)
         corr = dataset.corr()
         sns.heatmap(corr)
         plt.tight_layout()
         plt.show()

    def check_Sale_and_product_price(self,matrix):
      unique, counts = np.unique(matrix[:,1], return_counts=True)
      unique2, counts2 = np.unique(matrix[:,6], return_counts=True)
      print(dict(zip(unique, counts)))
      print(dict(zip(unique2, counts2)))

    def check_Sale_Product_price_diffrence(self, matrix):
        count_0_in_sale_and_product_price=0
        count_1_in_sale_and_0_in_product_price=0
        for x in range(0,15995634):
            if (matrix[x,1]=="0" and matrix[x,6]=="0.0"):
                count_0_in_sale_and_product_price+=1
            elif(matrix[x,1]=="1" and matrix[x,6]=="0.0"):
                count_1_in_sale_and_0_in_product_price+=1
        print("sale 0 and product 0:"+str(count_0_in_sale_and_product_price))
        print("sale 1 and product 0:"+str(count_1_in_sale_and_0_in_product_price))

    def check_if_Sale_0_price_0(self, matrix):
        price_equals_0_when_Sale_0=0
        price_not_equals_0_when_Sale_0=0

        for x in range(0,15995634):
            if (matrix[x,1]=="0" ):
                if(matrix[x,6]=="0.0"):
                    price_equals_0_when_Sale_0+=1
                else:
                    price_not_equals_0_when_Sale_0
      
        print("price_equals_0_when_Sale_0:"+str(price_equals_0_when_Sale_0))
        print("price_not_equals_0_when_Sale_0:"+str(price_not_equals_0_when_Sale_0))
