import numpy as np
from Analysis import Analysis
from Features import Features
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import datetime
from src.balancing import data_controller

columns = ["Sale", "SalesAmountInEuro", "time_delay_for_conversion", "click_timestamp", "nb_clicks_1week",
           "product_price", "product_age_group", "device_type","audience_id", "product_gender", "product_brand",
           "prod_category1", "prod_category2", "prod_category3", "prod_category4", "prod_category5", "prod_category6",
           "prod_category7", "product_country", "product_id", "product_title", "partner_id", "user_id"]


class PreProcessing:
    file_length = 15995634
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

    def load_data(self, file_path, load_fraction):
        print(" Data loading started ... ")
        lines_to_read = round(15995634 * load_fraction)
        lines = np.empty([lines_to_read, self.number_of_columns + 1], dtype=object)
        with open(file_path) as file:
            for x in range(lines_to_read):
                if x % 100000 == 0:
                    print(x)
                lines[x] = file.readline().rstrip("\n").split(",")
        print(" Data loading finished . Loaded rows")
        #print(lines)
        return lines

    def load_data_by_chunks(self, file_path, chunk_fraction):
        chunksize = round(self.file_length * chunk_fraction) + 1
        return pd.read_csv(file_path, chunksize=chunksize, delimiter="\t", dtype=str)

    def numpy_to_csv(self, matrix,file_name):
        np.savetxt(file_name, matrix, delimiter=",", fmt='%s')

    def csv_to_numpy(self, path):
        data = np.load(path)
        return data

    def get_X_and_Y(self, matrix):
        print(matrix)
        matrix = np.delete(np.delete(matrix, 0, 0), 0, 1)
        print(matrix)
        X = matrix[:, list(range(3, 23))]
        # time_stamp_column = X[:, 0]
        # time_stamp_column = [datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S') for date_time_str in
        #                      time_stamp_column]
        # time_stamp_column = [date_time_str.timestamp() for date_time_str in time_stamp_column]
        # X[:, 0] = time_stamp_column
        print(matrix)
        Y = matrix[:, 0]
        return X, Y

    def preprocess_data(self, data):
        print(" Data preprocessing started ... ")
        print(" Data preprocessing finished . Loaded dataset of {0} rows and {1} columns ".format(data.shape[0],
                                                                                                  data.shape[1]))
        data.tofile("test", ' ', "'%s'")

        return data

    def matrix_features(self,features,matrix):
        array_of_index=[1]
        for feature in features:
            if feature in columns:
                value_index = columns.index(feature)
                array_of_index.append(value_index+1) #because of headers  
        matrix_features=matrix[:,array_of_index]  
        self.numpy_to_csv(matrix_features,data_controller.path_data_dir+"/criteo/matrix_features.csv")
        return matrix_features

    def choose_feature(self,list_of_metods,X,Y,columns,iteration):
        features=[]
        
        for metod in list_of_metods:
            print(metod)
            switcher={
                "sfm_lr": lambda : features.append(Features.select_features_select_from_model_LR(X, Y, columns, iteration).tolist()),
                "sfm_linearsvc":lambda:features.append(Features.select_features_select_from_model_linearsvc(X, Y, columns, iteration).tolist()),
                "sfm_rfc":lambda:features.append(Features.select_features_select_from_model_RandomForest(X, Y, columns, iteration).tolist()),
                "sfm_lasso":lambda:features.append(Features.select_features_select_from_model_lasso(X, Y, columns, iteration).tolist()),#last sfm

                "rle_lr":lambda:features.append(Features.select_features_RFE_LR(X, Y, columns, iteration).tolist()),
                "rle_linearsvc":lambda:features.append(Features.select_features_RFE_linearsvc(X, Y, columns, iteration).tolist()),
                "rle_rfc":lambda:features.append(Features.select_features_RFE_RandomForest(X, Y, columns, iteration).tolist()),
                "rle_lasso":lambda:features.append(Features.select_features_RFE_lasso(X, Y, columns, iteration).tolist()),#last rle

                "permutation_lr":lambda:features.append(Features.select_features_permutation_LR(X, Y, columns, iteration).tolist()),
                "permutation_linearsvc":lambda:features.append(Features.select_features_permutation_linearsvc(X, Y, columns, iteration).tolist()),
                "permutation_rfc":lambda:features.append(Features.select_features_permutation_RandomForest(X, Y, columns, iteration).tolist()),
                "permutation_lasso":lambda:features.append(Features.select_features_permutation_lasso(X, Y, columns, iteration).tolist())

             }.get(metod,lambda: None)()
        flatten = [val for sublist in features for val in sublist]#flatten list
        

        features=list(dict.fromkeys(flatten))# delete duplicates
        return features       
            
        

features = Features()
test = PreProcessing()
analysis = Analysis()
##labelEncoders = pickle.load(open("E:\inz\criteo\criteo\lablencoder.pickle","rb"))

# test.analyze_data(test.load_data("E:\inz\criteo\criteo\csv\criteoCategorized_as_category.csv",1))
matrix=test.load_data(data_controller.path_categorized_criteo,0.001)
X, Y =test.get_X_and_Y(matrix)
#features_selected=Features.select_features_select_from_model_LR(X, Y, columns[3:], 10000)
#print(test.matrix_features(features_selected,matrix))

print(test.choose_feature(["sfm_lr","sfm_linearsvc","permutation_rfc"],X,Y,columns[3:],1000))

# test.numpy_to_csv(test.load_data("E:\inz\criteo\criteo\csv\criteoCategorized_as_category.csv", 0.01))
