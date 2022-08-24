from pre_processing import*
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib


def clean_text_(text):
    prepros = Pre_Processing()
    pre_text = prepros.clean_text(text)
    pre_text = prepros.remove_extra_spaces(pre_text)
    pre_text = prepros.lower_case(pre_text)
    pre_text = prepros.stemming(pre_text)
    return pre_text

def Predict(sample):
    text = clean_text_(sample)
    clf = joblib.load(r'C:\Users\Rícharde\Desktop\Python IA\Python-IA-Dell-Lead-\Final\Best_model\pre_processed_stemming__Bayes__TFIDF.pkl')
    text_predicted = clf.predict([text])
    new_predicted = text_predicted[0]
    if new_predicted == 0:
        type_clf = 'Non-Toxic'
    else:
        type_clf = 'Toxic'

    return type_clf
    