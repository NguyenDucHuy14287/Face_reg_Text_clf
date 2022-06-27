import pickle
from underthesea import word_tokenize
import re

def read_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)

def is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False

def preprocess_text(text):    
        text = re.sub(r'<[^>]*>', '', text) 
        text = text.lower()
        text = word_tokenize(text, format="text")
        text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_\[\]]',' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = ' '.join(map(lambda x: '<number>' if is_number(x) else x, text.split()))
        return text

class Text_CLF():
    __instance = None

    @staticmethod
    def getInstance(tfidf_vectorizer_path, clf_path):
        if Text_CLF.__instance == None:
            Text_CLF(tfidf_vectorizer_path, clf_path)
        return Text_CLF.__instance

    def __init__(self, tfidf_vectorizer_path, clf_path):
        self.TfidfVectorizer = read_pickle(tfidf_vectorizer_path)
        self.clf = read_pickle(clf_path)
        Text_CLF.__instance = self

    def classify(self, text):
        preprocessed_text = preprocess_text(text)
        feature_vector = self.TfidfVectorizer.transform([preprocessed_text])
        return self.clf.predict(feature_vector)[0]

    def predict_score(self, text):
        preprocessed_text = preprocess_text(text)
        feature_vector = self.TfidfVectorizer.transform([preprocessed_text])
        return self.clf.predict_proba(feature_vector)[:,1][0]

