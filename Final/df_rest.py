from typing_extensions import Self
import pandas as pd 
import numpy as np 


# ------------ Code -------------- # 

class Saving_DF:

    def __init__(self):
        self.df = pd.DataFrame()

    def Add_Colun(self,Colun,Data): # Add colun e your data 
        Self.df[Colun] = Data
    
    def Add_line(Data):
        Self.df = Self.df.append(Data)
    
    def Save_Df():
        Self.df.to_csv("Scores.csv")
