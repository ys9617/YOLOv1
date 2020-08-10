import cv2
import numpy as np
import time

from openvino.inference_engine import IENetwork, IECore


class YoloOpenVINO:
    def __init__(self, model_name, device='CPU'):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device

        self.model = IECore().read_network(model=self.model_structure, weights=self.model_weights)
        
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape

        self.labels = []
        self.load_time = 0
        self.total_inference_time = 0


    def load_model(self):
        start_time = time.time()
        self.net = IECore().load_network(self.model, self.device)
        self.load_time = time.time() - start_time


    def predict(self, image):
        output = self.net.infer({self.input_name:image})
        
        return output['Squeeze_62']


    # def preprocess_input(self, image):
    #     p_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    #     p_img = cv2.resize(p_img, (self.input_shape[3], self.input_shape[2]))
    #     p_img = p_img.transpose((2,0,1))
    #     p_img = p_img.reshape(1, *p_img.shape)

    #     return p_img

    # def get_model_load_time(self):
    #     return self.load_time

    # def get_total_inference_time(self):
    #     return self.total_inference_time






