import torch
import torch.nn as nn
#import torch.onnx
#import onnx
#import onnxruntime
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# YOLO v1 pytorch 
class Yolo(nn.Module):
    def __init__(self):
        super(Yolo, self).__init__()

        self.conv_0 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv_1 = nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_2 = nn.Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_4 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_5 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_7 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_8 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_9 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_10 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_11 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_12 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_13 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_14 = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_15 = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_16 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_17 = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_18 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_19 = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_20 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_21 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv_22 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_23 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conn_0 = nn.Linear(7*7*1024, 4096)
        self.conn_1 = nn.Linear(4096, 7*7*30)

        self.dropout = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool2d(2,2)
        self.leaky = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x):
        x = self.maxpool(self.leaky(self.conv_0(x)))
        x = self.maxpool(self.leaky(self.conv_1(x)))
        x = self.leaky(self.conv_2(x))
        x = self.leaky(self.conv_3(x))
        x = self.leaky(self.conv_4(x))
        x = self.maxpool(self.leaky(self.conv_5(x)))
        x = self.leaky(self.conv_6(x))
        x = self.leaky(self.conv_7(x))
        x = self.leaky(self.conv_8(x))
        x = self.leaky(self.conv_9(x))
        x = self.leaky(self.conv_10(x))
        x = self.leaky(self.conv_11(x))
        x = self.leaky(self.conv_12(x))
        x = self.leaky(self.conv_13(x))
        x = self.leaky(self.conv_14(x))
        x = self.maxpool(self.leaky(self.conv_15(x)))
        x = self.leaky(self.conv_16(x))
        x = self.leaky(self.conv_17(x))
        x = self.leaky(self.conv_18(x))
        x = self.leaky(self.conv_19(x))
        x = self.leaky(self.conv_20(x))
        x = self.leaky(self.conv_21(x))
        x = self.leaky(self.conv_22(x))
        x = self.leaky(self.conv_23(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.leaky(self.conn_0(x)))
        x = self.leaky(self.conn_1(x))
        x = torch.squeeze(x)

        return x




def load_weights(file, net):
    fp = open(file, 'rb')
    #The first 5 values are header information 
    # 1. Major version number
    # 2. Minor Version Number
    # 3. Subversion number 
    # 4. Images seen by the network (during training)
    header = np.fromfile(fp, dtype = np.int32, count = 4)
    weights = np.fromfile(fp, dtype = np.float32)

    conv_list = [net.conv_0, net.conv_1, net.conv_2, net.conv_3, net.conv_4, 
                 net.conv_5, net.conv_6, net.conv_7, net.conv_8, net.conv_9, 
                 net.conv_10, net.conv_11, net.conv_12, net.conv_13, net.conv_14, 
                 net.conv_15, net.conv_16, net.conv_17, net.conv_18, net.conv_19, 
                 net.conv_20, net.conv_21, net.conv_22, net.conv_23]

    conn_list = [net.conn_0, net.conn_1]

    ptr = 0
    for layer in conv_list:
        num_biases = layer.bias.numel()
        conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
        ptr = ptr + num_biases
        conv_biases = conv_biases.view_as(layer.bias.data)
        layer.bias.data.copy_(conv_biases)

        num_weights = layer.weight.numel()
        conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
        ptr = ptr + num_weights
        conv_weights = conv_weights.view_as(layer.weight.data)
        layer.weight.data.copy_(conv_weights)

    for layer in conn_list:
        num_biases = layer.bias.numel()
        conn_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
        ptr = ptr + num_biases
        conn_biases = conn_biases.view_as(layer.bias.data)
        layer.bias.data.copy_(conn_biases)

        num_weights = layer.weight.numel()
        conn_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
        ptr = ptr + num_weights
        conn_weights = conn_weights.view_as(layer.weight.data.T)
        conn_weights = conn_weights.T
        layer.weight.data.copy_(conn_weights)
        

def preprocess_input(self, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = cv2.resize(img, (448, 448))
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)
    img = img.type(torch.FloatTensor).unsqueeze(0)

    return img


# convert pytorch to onnx
# def to_onnx(net, output, batch_size, n_channel, height, width):
#     x = torch.randn(batch_size, n_channel, height, width, requires_grad=True)

#     torch_out = net(x)

#     torch.onnx.export(net, x, output,
#                     export_params=True,
#                     opset_version=11,
#                     do_constant_folding=True,
#                     input_names = ['input'],
#                     output_names = ['output'],
#                     dynamic_axes={'input' : {0 : 'batch_size'},
#                                   'output' : {0 : 'batch_size'}})

#     onnx_model = onnx.load(output)
#     onnx.checker.check_model(onnx_model)

#     ort_session = onnxruntime.InferenceSession(output)

#     def to_numpy(tensor):
#         return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

#     ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
#     ort_outs = ort_session.run(None, ort_inputs)

#     np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

#     print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    


        























