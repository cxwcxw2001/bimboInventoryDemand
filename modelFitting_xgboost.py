import pandas as pd
import numpy as np
import xgboost as xgb

print ("Read in raw data...\n")
# nrows=700000
raw_train=pd.read_csv(r"/Users/xiaoweichen/Kaggle/bimboInventoryDemand/Data/train.csv",nrows=70000)
# ,nrows=70000
raw_test=pd.read_csv(r"/Users/xiaoweichen/Kaggle/bimboInventoryDemand/Data/test.csv",nrows=70000)

# Featurization: remove categorical variables with the median, 25% percentile, 75% percentile, 5% 
# percentile and 95% percentile of the demand by category
# inspired by Owen Zhang: http://www.slideshare.net/OwenZhang2/tips-for-data-science-competitions
def calcSummaryStats(raw_df,features):
    summaryStats_dict={}
    for feature_name in features:
        summaryStats_dict[feature_name]={}

        medians=raw_df[[feature_name,"Demanda_uni_equil"]].groupby(feature_name).median()
        percentiles_5=raw_df[[feature_name,"Demanda_uni_equil"]].groupby(feature_name).quantile(q=0.05)
        percentiles_25=raw_df[[feature_name,"Demanda_uni_equil"]].groupby(feature_name).quantile(q=0.25)
        percentiles_75=raw_df[[feature_name,"Demanda_uni_equil"]].groupby(feature_name).quantile(q=0.75)
        percentiles_95=raw_df[[feature_name,"Demanda_uni_equil"]].groupby(feature_name).quantile(q=0.95)

        # ranges=raw_df[[feature_name,"Demanda_uni_equil"]].groupby(feature_name).max() - \
        #     raw_df[[feature_name,"Demanda_uni_equil"]].groupby(feature_name).min()

        summaryStats_dict[feature_name]["medians"]=medians
        summaryStats_dict[feature_name]["percentiles_5"]=percentiles_5
        summaryStats_dict[feature_name]["percentiles_25"]=percentiles_25
        summaryStats_dict[feature_name]["percentiles_75"]=percentiles_75
        summaryStats_dict[feature_name]["percentiles_95"]=percentiles_95
        # summaryStats_dict[feature_name]["ranges"]=ranges

    return summaryStats_dict

def encodeDataFrame(raw_df,summaryStats_dict):
    encoded_dict={}
    for feature_name in summaryStats_dict.keys():
        print ("Encoding {}...\n".format(feature_name))
        medians=summaryStats_dict[feature_name]["medians"]
        # The use of .lookup would be faster compared to .loc
        try:
            encoded_median=medians.lookup(raw_df[feature_name],["Demanda_uni_equil"] * len(raw_df))
            encoded_dict[feature_name+"_median"]=encoded_median
        except KeyError:
            encoded_median=medians.loc[raw_df[feature_name]]
            encoded_median.index=range(len(encoded_median))
            encoded_dict[feature_name+"_median"]=encoded_median.ix[:,0]

        percentiles_5=summaryStats_dict[feature_name]["percentiles_5"]
        try:
            encoded_5pct=percentiles_5.lookup(raw_df[feature_name],["Demanda_uni_equil"] * len(raw_df))
            encoded_dict[feature_name+"_5pct"]=encoded_5pct
        except KeyError:
            encoded_5pct=percentiles_5.loc[raw_df[feature_name]]
            encoded_5pct.index=range(len(encoded_5pct))
            encoded_dict[feature_name+"_5pct"]=encoded_5pct.ix[:,0]

        percentiles_25=summaryStats_dict[feature_name]["percentiles_25"]
        try:
            encoded_25pct=percentiles_25.lookup(raw_df[feature_name],["Demanda_uni_equil"] * len(raw_df))
            encoded_dict[feature_name+"_25pct"]=encoded_25pct
        except KeyError:
            encoded_25pct=percentiles_25.loc[raw_df[feature_name]]
            encoded_25pct.index=range(len(encoded_25pct))
            encoded_dict[feature_name+"_25pct"]=encoded_25pct.ix[:,0]

        percentiles_75=summaryStats_dict[feature_name]["percentiles_75"]
        try:
            encoded_75pct=percentiles_75.lookup(raw_df[feature_name],["Demanda_uni_equil"] * len(raw_df))
            encoded_dict[feature_name+"_75pct"]=encoded_75pct
        except KeyError:
            encoded_75pct=percentiles_75.loc[raw_df[feature_name]]
            encoded_75pct.index=range(len(encoded_75pct))
            encoded_dict[feature_name+"_75pct"]=encoded_75pct.ix[:,0]

        percentiles_95=summaryStats_dict[feature_name]["percentiles_95"]
        try:
            encoded_95pct=percentiles_95.lookup(raw_df[feature_name],["Demanda_uni_equil"] * len(raw_df))
            encoded_dict[feature_name+"_95pct"]=encoded_95pct
        except KeyError:
            encoded_95pct=percentiles_95.loc[raw_df[feature_name]]
            encoded_95pct.index=range(len(encoded_95pct))
            encoded_dict[feature_name+"_95pct"]=encoded_95pct.ix[:,0]

        # ranges=summaryStats_dict[feature_name]["ranges"]
        # try:
        #     encoded_ranges=ranges.lookup(raw_df[feature_name],["Demanda_uni_equil"] * len(raw_df))
        #     encoded_dict[feature_name+"_range"]=encoded_ranges
        # except KeyError:    
        #     encoded_ranges=ranges.loc[raw_df[feature_name]]
        #     encoded_ranges.index=range(len(encoded_ranges))
        #     encoded_dict[feature_name+"_range"]=encoded_ranges.ix[:,0]
        
    # Combine all summary statistics across features        
    try:
        encoded_df=pd.DataFrame(encoded_dict)
    except ValueError:
        encoded_df=pd.concat(encoded_dict,axis=1)

    return encoded_df
    

print ("Encoding for training set...\n")
features=["Agencia_ID","Canal_ID","Ruta_SAK","Cliente_ID","Producto_ID"]    
summaryStats_dict=calcSummaryStats(raw_train,features)

encoded_train=encodeDataFrame(raw_train,summaryStats_dict)
print (encoded_train.head(10))
print ("\nEncoding testing set...\n")
encoded_test=encodeDataFrame(raw_test,summaryStats_dict)

#Because there are some depot ID, channel ID, route ID,client ID and product ID that
#ONLY exist in testing data set, we have some NaNs in the encoded version of test set
print ("Fill missing value for testing set...\n")
nan_replacement={}
for feature_name in encoded_test.columns:
    if "median" in feature_name:
        nan_replacement[feature_name]=np.median(raw_train["Demanda_uni_equil"])
    elif "5pct" in feature_name:
        nan_replacement[feature_name]=np.percentile(raw_train["Demanda_uni_equil"],q=5)
    elif "25pct" in feature_name:
        nan_replacement[feature_name]=np.percentile(raw_train["Demanda_uni_equil"],q=25)
    elif "75pct" in feature_name:
        nan_replacement[feature_name]=np.percentile(raw_train["Demanda_uni_equil"],q=75)
    else:
        nan_replacement[feature_name]=np.percentile(raw_train["Demanda_uni_equil"],q=95)

encoded_test.fillna(nan_replacement,inplace=True)
print (encoded_test.head(10))

# print ("Fitting model...\n")
# tree=DecisionTreeRegressor(max_features="auto",min_samples_leaf=10)
# tree.fit(encoded_train,raw_train["Demanda_uni_equil"])
# # print ("Feature importances are...")
# # print (tree.feature_importances_)

# print ("Making prediction...\n")
# pred=tree.predict(encoded_test)
# result=pd.DataFrame({"Demanda_uni_equil":pred})
# result.to_csv(r"/Users/xiaoweichen/Kaggle/bimboInventoryDemand/result_tree2.csv",index_label="id")
# print ("All Done!")