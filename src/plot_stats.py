#%%
import matplotlib.pyplot as plt


#%%
device_list = ['pytorch', 'onnx', 'openvino']

inference_time = []
fps = []
model_load_time = []

for device in device_list:
    with open('./stats/' + device + '/stats.txt', 'r') as f:
        inference_time.append(float(f.readline().split("\n")[0]))
        fps.append(float(f.readline().split("\n")[0]))
        model_load_time.append(float(f.readline().split("\n")[0]))



#%% plot 

# total inference time
plt.bar(device_list, inference_time)
plt.xlabel("Device Used")
plt.ylabel("Total Inference Time in Seconds")
plt.show()

# frame per second
plt.bar(device_list, fps)
plt.xlabel("Device Used")
plt.ylabel("Frames per Second")
plt.show()

# model load time
plt.bar(device_list, model_load_time)
plt.xlabel("Device Used")
plt.ylabel("Model Loading Time in Seconds")
plt.show()





# %%
