import pickle
import lightgbm
import numpy as np

def read_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)

class Fusion_CLF():
    __instance = None

    @staticmethod
    def getInstance(clf_path):
        if Fusion_CLF.__instance == None:
            Fusion_CLF(clf_path)
        return Fusion_CLF.__instance

    def __init__(self, clf_path):
        self.clf = read_pickle(clf_path)
        Fusion_CLF.__instance = self

    def predict_score(self, text_prediction_score,image_prediction_score, face_regconition):
        return self.clf.predict_proba(np.array([text_prediction_score, image_prediction_score, face_regconition]).reshape(1, -1))[:,1][0]

