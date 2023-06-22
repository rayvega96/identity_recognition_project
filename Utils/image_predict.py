import tensorflow as tf
import logging
import cv2
import numpy as np
from ultralytics import YOLO
from keras.models import load_model
from keras.layers import Resizing

def calculate_text(name, x, y, w, h):

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    text = name

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    text_x = x + int((w - text_size[0]) / 2)
    text_y = y + h + int(text_size[1]) + 10

    return text_x, text_y, font, font_scale, thickness

def image_predict(image_path, model, class_names, dim=64):

    loaded_model = model


    resizer = Resizing(dim, dim, interpolation='bilinear', crop_to_aspect_ratio=True)


    # initialize the object camera (0 indicate main pc camera)
    image = cv2.imread(image_path)

    # Disabilita i messaggi di log generati da YOLO
    logging.disable(logging.CRITICAL)

    # Load a pretrained YOLO model (recommended for training)
    yolo_model = YOLO('Utils\\YoloFaceV8\\yolov8n-face.pt')
    yolo_model.verbose = False

    predictions = yolo_model.predict(source=image, device=0, max_det=4, conf=0.70, verbose=False)

    for prediction in predictions:

        ROIs = prediction.boxes
        ROIs = ROIs.cpu()   # E' necessario copiare il tensore CUDA sulla CPU prima di poterlo convertire a numpy
        ROIs = ROIs.numpy()

        for roi in ROIs:

            x1 = int(roi.xyxy.tolist()[0][0])
            y1 = int(roi.xyxy.tolist()[0][1])
            x2 = int(roi.xyxy.tolist()[0][2])
            y2 = int(roi.xyxy.tolist()[0][3])

            width = x2 - x1
            height = y2 - y1

            face = image[y1:y2, x1:x2]

            # se la ROI è più piccola di 64x64 applico un resize con interpolazione bicubica, migliore nella costruzione di pixel aggiuntivi
            #if width < dim or height < dim:
            #    face = cv2.resize(face,(dim,dim), interpolation=cv2.INTER_CUBIC)
            # se la ROI è più grande di 64x64 aplico un resize con interpolazione ad area, migliore nella riduzione delle dimensioni con perdita minima di qualità
            #elif width > dim or height > dim:
            #    face = cv2.resize(face,(dim,dim), interpolation=cv2.INTER_AREA) 

            face = resizer(face)

            normalized_face = face #/ 255.0

            input_face = tf.expand_dims(normalized_face, axis=0)

            prediction = loaded_model.predict(input_face)

            predicted_class_index = np.argmax(prediction)


            text_x, text_y, font, font_scale, thickness= calculate_text(class_names[predicted_class_index], x1, y1, width, height)

            cv2.putText(image, class_names[predicted_class_index], (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
            cv2.rectangle(image, (x1, y1, width, height), (255, 0, 0), 1)


    cv2.imshow(f"video recognition", image)
    key = cv2.waitKey(0)
