import pandas as pd 
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes  import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from pre_processing import *
import joblib

# --------- Import data --------
train_df = pd.read_csv(r'C:\Users\Rícharde\Desktop\Python IA\Python-IA-Dell-Lead-\Final\train_binary_small.csv')
test_df = pd.read_csv(r'C:\Users\Rícharde\Desktop\Python IA\Python-IA-Dell-Lead-\Final\test_binary_small.csv')
# -------- Label enconder Trasnform ----------
label_enconder = preprocessing.LabelEncoder()
train_df['Toxic'] = label_enconder.fit_transform(train_df['Toxic'])
test_df['Toxic'] = label_enconder.fit_transform(test_df['Toxic'])
# ------- Applying preprocessing of the best model ----------
# ------- pre_processed_stemming__Bayes__TFIDF ---------

def pre_processing(df,local_data):
    # Basic Clear Data 
    pre_proces = Pre_Processing()
    df[local_data] = df[local_data].apply(lambda local_data: pre_proces.expand_contractions(local_data))
    df[local_data] = df[local_data].apply(lambda local_data: pre_proces.lower_case(local_data))
    df[local_data] = df[local_data].apply(lambda local_data: pre_proces.clean_text(local_data))
    df[local_data] = df[local_data].apply(lambda local_data: pre_proces.remove_extra_spaces(local_data))
    return df 

train_df = pre_processing(train_df,'comment_text')
test_df  = pre_processing(test_df,'comment_text')

# ------- Applying preprocessig stemmming -------
stemming_pre = Pre_Processing()
train_df['comment_text'] = train_df['comment_text'].apply(lambda x: stemming_pre.stemming(x))
test_df['comment_text'] = test_df['comment_text'].apply(lambda x: stemming_pre.stemming(x))

# ------- Start  Nested Cross-Validation ----------
gridcvs = {}
inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)

# ------ Applying TFID ---------
vectorize_ = TfidfVectorizer()
bayes = MultinomialNB()

param = {}
param['classifier__alpha'] = [[1,0.1,00.1,000.1]]
param['classifier'] = [bayes]

param2 = {}
param2['vect__max_df'] = [0.2, 0.3, 0.4, 0.5, 0.6]
param2['vect'] = [vectorize_]


parms = [param,param2]

pipeline = Pipeline(
            [
                ("vect", vectorize_),
                ("classifier", bayes)
            ]
        )

rscv = GridSearchCV(pipeline,param,cv=inner_cv,scoring='accuracy',n_jobs=-1)

# --------- Separate data -----------
Coluns_train = train_df.drop(columns=['id'])
Coluns_test  = test_df.drop(columns=['id'])
Data = pd.merge(Coluns_train,Coluns_test, how='outer')
y = Data['Toxic']
X = Data['comment_text']

#-------- Cross Vall -------------

outer_scores = []
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


for train_idx, valid_idx in outer_cv.split(X,y):

    rscv.fit(X[train_idx], y[train_idx]) # run inner loop hyperparam tuning

    # perf on test fold (valid_idx)
    outer_scores.append(rscv.best_estimator_.score(X[valid_idx],y[valid_idx]))


print('Accuracy %.2f%% +/- %.2f' % (np.mean(outer_scores) * 100, np.std(outer_scores) * 100))

#-------- Saving the best model traning -----------
best_model = 'pre_processed_stemming__Bayes__TFIDF'
dst_file = f'C:/Users/Rícharde/Desktop/Python IA/Python-IA-Dell-Lead-/Final/Best_model/{best_model}.pkl'
joblib.dump(rscv,dst_file)
