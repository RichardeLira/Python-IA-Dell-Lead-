import pandas as pd 
import numpy as np 


pd = pd.DataFrame()


lista =  [10,20,30,50]

di = {
    ['knn'] : 20
}

pd = pd.assign(di)
print(pd)