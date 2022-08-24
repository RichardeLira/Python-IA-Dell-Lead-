from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes  import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from enum import Enum
import random
import numpy as np
# ----------- Models imports ---------------

class Models(Enum):
    KNN = 'KNN'
    BAYES = 'Bayes'
    DECITRE = 'Decison Tree'
    RANDOM_FOREST = 'Random_Forest'

class Vectorizers(Enum):
    TFIDF = 'TFIDF'
    BOW = 'BOW'
    

VECTORIZERS = {
    Vectorizers.BOW.value: CountVectorizer(),
     Vectorizers.TFIDF.value: TfidfVectorizer()
}


CLF_MODELS = {
    Models.KNN.value: KNeighborsClassifier(n_jobs=-1),
    Models.BAYES.value: GaussianNB(),
    Models.RANDOM_FOREST.value: RandomForestClassifier(),
    Models.DECITRE.value: DecisionTreeClassifier()
}


CLF_PARAMS = {
    Models.KNN.value: {
        'n_neighbors': [5, 7, 9, 11, 15]
    },
    Models.DECITRE.value: {
        'max_depth': [10,20,50,100]
    },
    Models.BAYES.value:{
        'var_smoothing': np.logspace(0,-9, num=20)
    },
    Models.RANDOM_FOREST.value:  {
        'n_estimators': list(np.arange(25, 45,10)),
    }
}