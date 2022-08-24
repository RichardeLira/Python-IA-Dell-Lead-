# -------- Pre Processing file -------

import contractions
import re 
import string
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import spacy
nlp = spacy.load('en_core_web_sm')

class Pre_Processing():


    def expand_contractions(self,sentence):
        expanded_words = []   
        for word in sentence.split():
            expanded_words.append(contractions.fix(word))  
        return ' '.join(expanded_words)

    def lower_case(self,sentence):
        sentence = sentence.lower()
        return sentence

    # remove links,dots,commas,numbers 
    def clean_text(self,instance):
        instance = instance.lower()
        instance = re.sub('\[.*?\]', ' ', instance)
        instance = re.sub('https?://\S+|www\.\S+', ' ', instance)
        instance = re.sub('<.*?>+', ' ', instance)
        instance = re.sub('[%s]' % re.escape(string.punctuation), ' ', instance)
        instance = re.sub('\n', '', instance)
        instance = re.sub('\w*\d\w*', ' ', instance)
        return instance

    def remove_extra_spaces(self,instance):
        instance = re.sub(' +', ' ', instance)
        return instance

    def remove_stopwords(self,text,stopwords):
        words = [ word for word in word_tokenize(text) if not word in stopwords]
        new_text =  " ".join(words)
        return new_text

    def stemming(self,text):
        stemmer = SnowballStemmer("portuguese")
        words = [stemmer.stem(word) for word in text.split()]
        new_text =  " ".join(words)
        return new_text

    def lemmatization(self,text):
        doc = nlp(text)
        lemmas = [token.lemma_ for token in doc]
        new_text =  " ".join(lemmas)
        return new_text