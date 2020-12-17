import numpy as np
from Analysis import Analysis
from Features import Features
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import datetime

columns=["Sale","SalesAmountInEuro","time_delay_for_conversion","click_timestamp","nb_clicks_1week","product_price","product_age_group","device_type,audience_id","product_gender","product_brand","prod_category1","prod_category2","prod_category3","prod_category4","prod_category5","prod_category6","prod_category7","product_country","product_id","product_title","partner_id","user_id"]

class PreProcessing:
    file_length = 15995634;
    number_of_columns = 23
    fmt = "%s"
    labelEncoders = {'product_age_group': LabelEncoder(),
     'device_type': LabelEncoder(),
     'audience_id': LabelEncoder(),
     'product_gender': LabelEncoder(),
     'product_brand': LabelEncoder(),
     'prod_category1': LabelEncoder(),
     'prod_category2': LabelEncoder(),
     'prod_category3': LabelEncoder(),
     'prod_category4': LabelEncoder(),
     'prod_category5': LabelEncoder(),
     'prod_category6': LabelEncoder(),
     'prod_category7': LabelEncoder(),
     'product_country': LabelEncoder(),
     'product_id': LabelEncoder(),
     'product_title': LabelEncoder(),
     'partner_id': LabelEncoder(),
     'user_id': LabelEncoder()}

    def load_data (self,file_path , load_fraction ):
        print (" Data loading started ... ")
        lines_to_read = round (15995634 * load_fraction )
        lines = np . empty ([ lines_to_read , self . number_of_columns+1 ] , dtype = object )
        with open ( file_path ) as file :
            for x in range ( lines_to_read ) :
                    if x%100000==0:
                        print(x)
                    lines [x] = file . readline (). rstrip("\n") . split (",")
        print (" Data loading finished . Loaded rows")
        #print(type(lines))
        #print (lines_to_read)
        #print(lines[0])
        #print(lines[1])
        #lines.tofile("test",' ',"'%s'")
        #file = open("file2.txt", "w+")
        #content = str(lines) 
        #file.write(content) 
        #file.close()
        return lines

    def load_data_by_chunks ( self , file_path , chunk_fraction ):
        chunksize = round ( self . file_length * chunk_fraction ) + 1
        return pd . read_csv ( file_path , chunksize = chunksize , delimiter ="\t", dtype =
        str )
    def get_X_and_Y(self, matrix):
        
        matrix=np.delete(np.delete(matrix,0,0),0,1)
        X=matrix[:,list(range(3, 22))]
        time_stamp_column=X[:,0]
        time_stamp_column=[datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S') for date_time_str in time_stamp_column]
        time_stamp_column=[date_time_str.timestamp() for date_time_str in time_stamp_column]
        X[:,0]=time_stamp_column
        Y=matrix[:,0]        
        Y=[int(y == 'True') for y in Y] 
        Y=np.array(Y)
        Y=Y.astype(float)
        return X,Y

    def preprocess_data (self, data):
        print (" Data preprocessing started ... ")

            
       ## encoder = np.LabelEncoder()
        ##for x in range (3 , np.size(data ,1)):
          ## data [: , x] = encoder.fit_transform( data [: , x ])

        print (" Data preprocessing finished . Loaded dataset of {0} rows and {1} columns ".format (data.shape [0],data.shape[1]))
        data.tofile("test",' ',"'%s'")

        return data

features=Features()
test=PreProcessing()
analysis=Analysis()
##labelEncoders = pickle.load(open("E:\inz\criteo\criteo\lablencoder.pickle","rb"))

#test.analyze_data(test.load_data("E:\inz\criteo\criteo\csv\criteoCategorized_as_category.csv",1))

X,Y=test.get_X_and_Y(test.load_data("E:\inz\criteo\criteo\csv\criteoCategorized_as_category.csv",0.01))
Features.select_fetures_select_from_model_linearsvc(X,Y,columns[3:22],1000000)