library(modelconf)
library(rugarch)
library(readr)
library(dplyr)
set.seed(0)
# Suppose there is a combined csv file of all the forecast results for "large" datasets

large_l2_loss <- read_csv("result/large_forecast_err_l2.csv",col_names = FALSE) %>% t()
colnames(large_l2_loss)<-large_l2_loss[1,]
large_l2_loss<-large_l2_loss[-1,]
large_l2_loss<-apply(large_l2_loss,2,as.numeric)

large_l2_SSM <- estMCS(large_l2_loss, test = "t.max", B = 25000, l = 12)

large_l2_SSM
