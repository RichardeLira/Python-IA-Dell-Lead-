from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes  import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from enum import Enum
import random
import numpy as np
# ----------- Models imports ---------------

class Models(Enum):
    KNN = 'KNN'
    SVM_LINEAR = 'SVM_Linear'
    BAYES = 'Bayes'
    MLP_1LAYER = 'MLP_1Layer'
    MLP_2LAYERS = 'MLP_2Layers'
    RANDOM_FOREST = 'Random_Forest'



class Vectorizers(Enum):
    TFIDF = 'TFIDF'
    BOW = 'BOW'
    

VECTORIZERS = {
    Vectorizers.BOW.value: CountVectorizer(),
     Vectorizers.TFIDF.value: TfidfVectorizer()
}


CLF_MODELS = {
    Models.KNN.value: KNeighborsClassifier(),
    Models.BAYES.value: GaussianNB(),
    Models.SVM_LINEAR.value: SVC(kernel='linear', max_iter=1000),
    Models.MLP_1LAYER.value: MLPClassifier(early_stopping=True),
    Models.MLP_2LAYERS.value: MLPClassifier(early_stopping=True),
    Models.RANDOM_FOREST.value: RandomForestClassifier()
}

C = [2 ** i for i in range(-5, 16, 2)]

CLF_PARAMS = {
    Models.KNN.value: {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 17]
    },
    Models.SVM_LINEAR.value: {
        'C': C
    },
    Models.BAYES.value:{
        'var_smoothing': np.logspace(0,-9, num=20)
    },
    Models.MLP_1LAYER.value: {
        'hidden_layer_sizes': [random.randrange(2,500,25) for i in range(10)]
    },
    Models.MLP_2LAYERS.value: {
        'hidden_layer_sizes': [(random.randrange(2, 500, 25), random.randrange(2, 500, 25)) for i in range(10)]
    },
    Models.RANDOM_FOREST.value:  {
        'n_estimators': list(np.arange(25, 2001,40)),
    }
}