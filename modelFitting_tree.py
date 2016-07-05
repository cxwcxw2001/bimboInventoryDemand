import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

print ("Read in raw data...\n")
raw_train=pd.read_csv(r"/Users/xiaoweichen/Kaggle/bimboInventoryDemand/Data/train.csv")
raw_test=pd.read_csv(r"/Users/xiaoweichen/Kaggle/bimboInventoryDemand/Data/test.csv")

# Replace values of categorical variables with the median of the demand by category
# borrowed from Owen Zhang: http://www.slideshare.net/OwenZhang2/tips-for-data-science-competitions
def medianDemandByCategory(df,feature_name):
    grouped=df[[feature_name,"Demanda_uni_equil"]].groupby(feature_name).median()
    grouped.columns=[feature_name]
    return grouped

features=["Agencia_ID","Canal_ID","Ruta_SAK","Cliente_ID","Producto_ID"]    
grouped={feature:medianDemandByCategory(raw_train,feature) for feature in features}

def encodeDataFrame(raw_df,grouped_dict):
    encoded_dict={}
    for feature_name,grouped_df in grouped_dict.items():
        print ("Encoding {}...\n".format(feature_name))
        encoded_feature=grouped_df.loc[raw_df[feature_name]]
        encoded_feature.index=range(len(encoded_feature))
        encoded_dict[feature_name]=encoded_feature
    encoded_df=pd.concat(encoded_dict,axis=1,keys=None)
    return encoded_df

print ("Encoding training set...\n")
encoded_train=encodeDataFrame(raw_train,grouped)
print ("Encoding testing set...\n")
nan_replacement=np.median(raw_train["Demanda_uni_equil"])
encoded_test=encodeDataFrame(raw_test,grouped)
# Because there are some depot ID, channel ID, route ID,client ID and product ID that
# ONLY exist in testing data set, we have some NaNs in the encoded version of test set
nan_replacement=np.median(raw_train["Demanda_uni_equil"])
encoded_test.fillna(nan_replacement,inplace=True)

print ("Fitting model...\n")
tree=DecisionTreeRegressor(max_features="auto",min_samples_leaf=10)
tree.fit(encoded_train,raw_train["Demanda_uni_equil"])
# print ("Feature importances are...")
# print (tree.feature_importances_)

print ("Making prediction...\n")
pred=tree.predict(encoded_test)
result=pd.DataFrame({"Demanda_uni_equil":pred})
result.to_csv(r"/Users/xiaoweichen/Kaggle/bimboInventoryDemand/result_tree.csv",index_label="id")
print ("All Done!")