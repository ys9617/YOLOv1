import argparse
import cv2
import numpy as np
import os
import onnxruntime
import time
import torch

from yolo_openvino import YoloOpenVINO
from yolo import Yolo, load_weights


def draw_boxes(outputs, img):
    h, w, c = img.shape

    for box in outputs:
        x0 = int(box[0] * w / 448)
        y0 = int(box[1] * h / 448)
        x1 = int(box[2] * w / 448)
        y1 = int(box[3] * h / 448)
        label = box[4]

        cv2.rectangle(img, (x0, y0),  (x1, y1), (0, 0, 255), 1)
        cv2.putText(img, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        
    cv2.imshow('img', img)

def output_handle(output, img, threshold):
    input_width = 448
    input_height = 448

    num_boxes = 2
    boxes = []

    slide = 7

    grid_w = int(input_width / slide)
    grid_h = int(input_height / slide)

    labels = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
              'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa',
              'train','tvmonitor']

    for ih in range(slide):
        for iw in range(slide):
            for ib in range(num_boxes):
                idx = ih * slide + iw
                p_idx = slide*slide*len(labels) + num_boxes * idx + ib
                b_idx = slide*slide*(len(labels)+num_boxes) + (num_boxes * idx + ib) * 4

                objectness = output[p_idx]
                x = int((iw + output[b_idx + 0]) * grid_w)
                y = int((ih + output[b_idx + 1]) * grid_h)
                w = pow(output[b_idx + 2], 2) * input_width
                h = pow(output[b_idx + 3], 2) * input_height

                for i in range(len(labels)):
                    class_idx = idx * len(labels)
                    prob = objectness * output[class_idx+i]

                    if prob > threshold:
                        x0 = x - w/2
                        y0 = y - h/2
                        x1 = x + w/2
                        y1 = y + h/2

                        boxes.append([x0,y0,x1,y1, labels[i]])

    return boxes            
    
def predict(net, img, framework):
    # preprocess
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = cv2.resize(img, (448, 448))
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis,:,:,:]
    img = img.astype('float32')

    start_time = time.time()

    if framework == 'pytorch':
        img = torch.from_numpy(img)
        output = net(img)
    elif framework =='onnx':
        ort_inputs = {net.get_inputs()[0].name: img}
        output = net.run(None, ort_inputs)[0]
    elif framework == 'openvino':
        output = net.predict(img)

    inference_time = time.time() - start_time

    return output, inference_time


def main(args):
    # args load
    device = args.device
    input_type = args.input_type
    input_file = args.input_file
    model_type = args.model_type
    model_path = args.model_path
    threshold = float(args.threshold)
    debug_mode = args.debug_mode
    output_path = args.output_path

    # load model
    start_time = time.time()

    if model_type == 'pytorch':
        print('Load Pytorch model')
        net = Yolo()
        net.eval()   
        net.load_state_dict(torch.load(model_path))
    elif model_type == 'onnx':
        print('Load ONNX model')
        net = onnxruntime.InferenceSession(model_path)
    elif model_type == 'openvino':
        print('Load OpenVINO model')
        net = YoloOpenVINO(model_path, device)
        net.load_model()
    
    model_load_time = time.time() - start_time

    total_count = 0
    total_inference_time = 0
    
    if input_type == 'image':
        img = cv2.imread(input_file)
        
        output, inference_time = predict(net, img, model_type)

        total_count += 1

        boxes = output_handle(output, img, threshold)
        draw_boxes(boxes, img)

        cv2.waitKey(0)
    elif input_type == 'video' or input_type == 'cam':
        if input_type == 'cam':
            input_file = 0
            
        cap = cv2.VideoCapture(input_file)

        w = int(cap.get(3))
        h = int(cap.get(4))

        while cap.isOpened():
            flag, frame = cap.read()

            if not flag:
                break

            output, inference_time = predict(net, frame, model_type)

            total_count += 1

            if debug_mode == 'on':
                total_inference_time += inference_time
                fps = total_count / total_inference_time

                cv2.putText(frame, "FPS : {:.3f}s".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,), 2)

            boxes = output_handle(output, frame, threshold)
            draw_boxes(boxes, frame)

            key_pressed = cv2.waitKey(60)

            if key_pressed == 27:
                break

    
    if debug_mode == 'on':
        print('Model Load Time : ', model_load_time)
        print('FPS : ', total_count / total_inference_time)
        print('Average Inference Time : ', total_inference_time / total_count)

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(model_load_time)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='CPU')
    parser.add_argument('--input_type', default="image")
    parser.add_argument('--input_file', default='bin/eagle.jpg')
    parser.add_argument('--model_type', default='pytorch')
    parser.add_argument('--model_path', default='yolo_v1')
    parser.add_argument('--threshold', default='0.3')
    parser.add_argument('--debug_mode', default='off')
    parser.add_argument('--output_path', default='./stats')


    args = parser.parse_args()

    main(args)