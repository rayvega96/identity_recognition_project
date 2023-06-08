import tensorflow as tf
import h5py
import logging
import cv2
import numpy as np
from ultralytics import YOLO
from keras.models import load_model


loaded_model = load_model('model/second_custom_model_conti.h5')

class_names = ['Davide', 'Francesco', 'Gabriele', 'Stefano', 'Unknown']

# initialize the object camera (0 indicate main pc camera)
camera = cv2.VideoCapture(0)

# Disabilita i messaggi di log generati da YOLO
logging.disable(logging.CRITICAL)

# Load a pretrained YOLO model (recommended for training)
yolo_model = YOLO('YoloFaceDetector/yolov8n-face.pt')
yolo_model.verbose = False


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
thickness = 2

def calculate_text(name, x, y, w, h):

    text = name

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    text_x = x + int((w - text_size[0]) / 2)
    text_y = y + h + int(text_size[1]) + 10

    return text_x, text_y


while True:

    # reading frames. ret is a boolean indicating if the frame got captured correctly
    ret, frame = camera.read()

    results = yolo_model(frame)
    boxes = results[0].boxes


    for box in boxes:

        top_left_x = int(box.xyxy.tolist()[0][0])
        top_left_y = int(box.xyxy.tolist()[0][1])
        bottom_right_x = int(box.xyxy.tolist()[0][2])
        bottom_right_y = int(box.xyxy.tolist()[0][3])

        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (50,200,129), 2)
        resized_face = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]


        width = bottom_right_x - top_left_x
        height = bottom_right_y - top_left_y

        # se la ROI è più piccola di 64x64 applico un resize con interpolazione bicubica, migliore nella costruzione di pixel aggiuntivi
        if width < 64 or height < 64:
            resized_face = cv2.resize(resized_face,(64,64), interpolation=cv2.INTER_CUBIC)
        # se la ROI è più grande di 64x64 aplico un resize con interpolazione ad area, migliore nella riduzione delle dimensioni con perdita minima di qualità
        elif width > 64 or height > 64:
            resized_face = cv2.resize(resized_face,(64,64), interpolation=cv2.INTER_AREA) 

        normalized_face = resized_face / 255.0

        input_face = tf.expand_dims(normalized_face, axis=0)
        

        prediction = loaded_model.predict(input_face)

        
        predicted_class_index = np.argmax(prediction)


        text_x, text_y = calculate_text(class_names[predicted_class_index], top_left_x, top_left_y, width, height)

        cv2.putText(frame, class_names[predicted_class_index], (text_x, text_y), font, font_scale, (0, 255, 0), thickness)


        cv2.imshow(f"video recognition", frame)
        key = cv2.waitKey(1)

        #press 'q' to quit the loop
        if key == ord('q'):
            break