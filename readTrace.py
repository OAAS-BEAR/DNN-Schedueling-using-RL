import pandas as pd
import numpy as np
from math import isnan
import random
np.random.seed(10)


'''
read trace data from sqlite, the trace is downloaded from https://github.com/Azure/AzurePublicDataset/blob/master/AzureTracesForPacking2020.md
'''
import sqlite3
with sqlite3.connect('./packing_trace_zone_a_v1.sqlite') as con:
    #df = pd.read_sql_query("SELECT * FROM (SELECT * FROM vm where starttime >= 0 and starttime < 1 order by random() limit 2182) order by starttime", con=con)
    df = pd.read_sql_query("SELECT * FROM vm where starttime >= 0 and starttime < 1 order by random()", con=con)
    
    print(df.shape)
    
    i=0
    num=0
    timeup=0
    tmp=0
    # maxnum = np.zeros(2182)
    # for indexs in df.index:
    #     if round(df.loc[indexs].values[-2]*86400)>i:
    #         tmp =round(df.loc[indexs].values[-2]*86400)
    #         timeup = 1
    #     num = num +1
    #     if timeup == 1:
    #         for j in range(indexs):
    #             if np.isnan(df.loc[j].values[-1]*86400):
    #                 continue
    #             if round(df.loc[j].values[-1]*86400) > i and round(df.loc[j].values[-1]*86400) <=tmp:
    #                 num = num -1
    #     maxnum[indexs] = num
    #     timeup = 0
    #     i =tmp
    # print(np.max(maxnum))
    
    for i in range(100):
        dftmp = df.loc[i*4363:(i+1)*4363-1]
        if i <80:
            for indexs in dftmp.index:
                dftmp.loc[indexs,'starttime'] = int(round(dftmp.loc[indexs].values[-2]*86400,0))
                if isnan(dftmp.loc[indexs].values[-1]):
                    pass
                else:
                    dftmp.loc[indexs,'endtime'] = int(round(dftmp.loc[indexs].values[-1]*86400,0))
            dftmp.to_csv("./data-4363/test"+str(i)+".csv")
        else:
            for indexs in dftmp.index:
                dftmp.loc[indexs,'starttime'] = int(round(dftmp.loc[indexs].values[-2]*86400,0))
                if isnan(dftmp.loc[indexs].values[-1]):
                    pass
                else:
                    dftmp.loc[indexs,'endtime'] = int(round(dftmp.loc[indexs].values[-1]*86400,0))
            dftmp.to_csv("./data-4363/val"+str(i-80)+".csv")
