setwd("C:/Users/xchen34/Downloads/train.csv")
library(data.table)

# Sales and returns VS week
raw_train=fread("train.csv",sep=",",header=TRUE)
sales_by_week=raw_train[,list(median_sales=as.double(median(Venta_uni_hoy))),
                        by=Semana]
barplot(sales_by_week$median_sales,names.arg=3:9) # pretty stable pattern, which makes sense
return_by_week=raw_train[,list(median_return=as.double(median(Dev_uni_proxima))),
                         by=Semana]
barplot(return_by_week$median_return,names.arg=3:9) # one straight line...
# conclusion: week is not a useful feature for the prediction of either returns or sales

#  to be analyzed

# sales and returns VS sales channel
sales_by_channel=raw_train[,list(median_sales=as.double(median(Venta_uni_hoy))),
                           by=Canal_ID]
barplot(sales_by_channel$median_sales,names.arg=sales_by_channel$Canal_ID)
return_by_channel=raw_train[,list(median_return=as.double(median(Dev_uni_proxima))),
                         by=Canal_ID]
barplot(return_by_channel$median_return,names.arg=return_by_channel$Canal_ID)
# Pattern of sales and return VS sales channel looks similar. How about demand?
demand_by_channel=raw_train[,list(median_demand=as.double(median(Demanda_uni_equil))),
                           by=Canal_ID]
barplot(demand_by_channel$median_demand,names.arg=demand_by_channel$Canal_ID)
# Conclusion: Sales channel is definitely an informative feature

#There are 552 sales deopt ID (Agencia ID), 3603 Route IDs, 880604 client IDs, 
#and 1799 product IDs  