import string
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def predict_Model(text:str):
    return {"message": text}