from Utils.live_predict import live_predict
from Utils.image_predict import image_predict
from keras.models import load_model
import random

import os
import glob


# scegliere un modello da Model\\Trained_Models\\
#model = load_model("Trained_Models\\ResNet\\second_resnet_imagenet\\second_resnet.16-0.01.h5")



model_path = "Models\\ResNet\\V1\\"
model_list = os.listdir(model_path)
for i, element in enumerate(model_list):
    print(f"[{i}] {element}")
scelta = int(input("Scegli un modello: "))
if scelta >= 0 and scelta <= len(model_list):
    model = load_model(model_path+model_list[i])
    classi = ['Davide', 'Francesco', 'Gabriele', 'Stefano']
    live_predict(model=model, class_names=classi, max_classes=1)

    while True:

        index_class = random.randint(0,3)
        chosen_class = classi[index_class]
        images_list = os.listdir(f"Dataset\\Original\\{chosen_class}\\")

        image_predict(f"Dataset\\Original\\{chosen_class}\\{images_list[random.randint(0,len(images_list)-1)]}", model, classi)


