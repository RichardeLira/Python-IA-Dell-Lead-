from fastapi import FastAPI
from Models_API import *
# import scrip
import Models_API

import importlib
importlib.reload(Models_API)

app = FastAPI()
@app.get("/")
def home(text:str):
    pred = Predict(text)
    return {'TIPO :': pred}