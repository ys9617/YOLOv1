[image1]: ./bin/network.png "YOLO v1 Network"


# YOLO v1

YOLO v1 implementation and performance comparison between Pytorch, ONNX and OpenVINO.

## Overview

1. YOLO v1 Network Design
2. PyTorch YOLO v1 Implemention and Loading Pre-trained Weights
3. Converting PyTorch to ONNX and OpenVINO
4. Performance Comparison Between PyTorch, ONNX and OpenVINO
5. Reference

## YOLO v1 Network Design

![YOLO v1 Network][image1]

YOLO v1 network has 24 convolutional layers followed by 2 fully connected layers and has total 41 GFLOPs.


## PyTorch YOLO v1 Implemention and Loading Pre-trained Weights

Model weights file can download from [here](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU)

```python
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
```


## Reference

* [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) by Joseph Redmon et al

* [Darkflow, YOLO v1 & v2 Tensorflow implementation](https://github.com/thtrieu/darkflow)