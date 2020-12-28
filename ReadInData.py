import pandas as pd
import pickle

#with open("pickle_file/Data.pkl", "rb") as file:
    #data = pickle.load(file)

data.loc[:, "開單日期"]=pd.to_datetime(data["開單日期"].astype(str)).dt.date
data=data[(data["類別"]!="感應卡") & (data["類別"]!="其他") & (data["類別"]!="會員")]
data=data[data["開單日期"] > pd.to_datetime("2018")]

