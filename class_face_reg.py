import os
import cv2
import imutils
from tqdm import tqdm
import numpy as np

from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.engine import Model

from sklearn.metrics.pairwise import cosine_similarity

from facenet_pytorch import MTCNN
import torch

from numpy import dot
from numpy.linalg import norm

def get_info_video(filename=''):
    video = cv2.VideoCapture(filename)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = int(frame_count/fps)
    video.release()
    return fps, frame_count, duration


class face_detector_torch(object):
    def __init__(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True)

    def detect(self, frame):
        return_faces = []
        frame = imutils.resize(frame, width=400)
        boxes, _ = self.mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                (startX, startY, endX, endY) = box.astype("int")
                return_faces.append(frame[startY:endY, startX:endX])

        return return_faces


class similarity_face(object):
    def __init__(self, layer="fc7"):
        layer_name = layer 
        vgg_model = VGGFace() 
        out = vgg_model.get_layer(layer_name).output
        self.vgg_model_new = Model(vgg_model.input, out)

    def get_embedding_vector(self, frame):
        try:
            resized = cv2.resize(frame, (224, 224))
            x = image.img_to_array(resized)
            x = np.expand_dims(x, axis=0)
            x = utils.preprocess_input(x, version=2) 
            preds = self.vgg_model_new.predict(x)
            return preds
        except Exception as e:
            print(e)
            return np.zeros((1, 4096))


def calculate_cosine_similarity(matrix, vector): # matrix:(n, 4096), vector: (1, 4096)
    p1 = dot(matrix, vector.reshape(-1,1))
    p2 = norm(matrix, axis=1)*norm(vector)
    return p1/p2.reshape(-1,1)


class face_similarity_seacher(similarity_face):
    def __init__(self):
        similarity_face.__init__(self)
        self.embed_matrix_dict = {}
    
    def load_embed_matrix_dict(self, embeding_matrix_folder='face_embedding'):
        embed_matrix_dir = os.listdir('face_embedding')
        for matrix_path in embed_matrix_dir:
            embedding_matrix = np.load('face_embedding/' + matrix_path)
            self.embed_matrix_dict[matrix_path[:-4]] = embedding_matrix
            
    @staticmethod
    def calculate_cosine_similarity(matrix, vector): # matrix:(n, 4096), vector: (1, 4096)
        p1 = dot(matrix, vector.reshape(-1,1))
        p2 = norm(matrix, axis=1)*norm(vector)
        return p1/p2.reshape(-1,1)
    
    def seach_similarity_face(self, face_frame, thresh_hold = 0.9):
        embed_vector = self.get_embedding_vector(face_frame)
        ret_val = None
        for name, embedding_matrix in self.embed_matrix_dict.items():
            max_similarity_score = calculate_cosine_similarity(embedding_matrix, embed_vector).max()
            if max_similarity_score > thresh_hold:
                ret_val = (name, max_similarity_score)
                break
        return ret_val


class Face_reg():
    __instance = None

    @staticmethod
    def getInstance():
        if Face_reg.__instance == None:
            Face_reg()
        return Face_reg.__instance

    def __init__(self):
        self.face_detector = face_detector_torch()
        self.face_searcher = face_similarity_seacher()
        self.face_searcher.load_embed_matrix_dict()
        Face_reg.__instance = self

    def search_face(self, frame):
        ret_lst = []
        return_faces = self.face_detector.detect(frame)
        for face_frame in return_faces:
            ret_lst.append(self.face_searcher.seach_similarity_face(face_frame, thresh_hold = 0.95))
        return ret_lst

    def get_face_in_vid(self, video_path='VIDEO/1.NGUYENPHUTRONG/1.mp4'):
        video = cv2.VideoCapture(video_path)
    
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = int(frame_count/fps)

        frame_count = 0
        face_detection_lst = []

        with tqdm(total=duration, desc = 'Process video') as pbar:
            while(video.isOpened()):
                ret, frame = video.read()
                if ret == True:
                    if frame_count % fps == 0:
                        # do somethings
                        info = self.search_face(frame)
                        # if len(info) != 0 or info[0] != None:
                        face_detection_lst.append(info)
                        pbar.update(1)
                    frame_count += 1
                else:
                    break 

        pbar.close()
        video.release()

        ret_dict = {
            'file_name': video_path,
            'duration': duration,
            'fps': fps,
            'face_reg': face_detection_lst
            }
        return ret_dict

    def get_face_in_image(self, img_path):
        frame = cv2.imread(img_path)
        info = self.search_face(frame)

        if len(info) == 0:
            return 0
        else:
            for i in info:
                if i is not None:
                    return 1
        return 0

    def predict_video_frames(self, frames_folder_path):
        frame_lst = os.listdir(frames_folder_path)

        return np.max(list(map(lambda image_name: self.get_face_in_image(frames_folder_path + '/' + image_name), frame_lst)))
