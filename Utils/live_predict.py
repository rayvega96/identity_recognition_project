import tensorflow as tf
import logging
import cv2
import numpy as np
from ultralytics import YOLO
from keras.models import load_model


def live_predict(model, class_names, max_classes=4):

    loaded_model = model


    # initialize the object camera (0 indicate main pc camera)
    camera = cv2.VideoCapture('Dataset\\video_test_finale.mp4')

    # Disabilita i messaggi di log generati da YOLO
    logging.disable(logging.CRITICAL)

    # Load a pretrained YOLO model (recommended for training)
    yolo_model = YOLO('Utils\\YoloFaceV8\\yolov8n-face.pt')
    yolo_model.verbose = False


    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    def calculate_text(name, x, y, w, h):

        text = name

        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

        text_x = x + int((w - text_size[0]) / 2)
        text_y = y + h + int(text_size[1]) + 10

        return text_x, text_y


    while True:

        # reading frames. ret is a boolean indicating if the frame got captured correctly
        ret, frame = camera.read()

        predictions = yolo_model.predict(source=frame, device=0, max_det=4, conf=0.70, verbose=False)

        faces = []

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

                face = frame[y1:y2, x1:x2]

                # se la ROI è più piccola di 64x64 applico un resize con interpolazione bicubica, migliore nella costruzione di pixel aggiuntivi
                if width < 64 or height < 64:
                    face = cv2.resize(face,(256,256), interpolation=cv2.INTER_CUBIC)
                # se la ROI è più grande di 64x64 aplico un resize con interpolazione ad area, migliore nella riduzione delle dimensioni con perdita minima di qualità
                elif width > 64 or height > 64:
                    face = cv2.resize(face,(256,256), interpolation=cv2.INTER_AREA) 

                normalized_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                input_face = tf.expand_dims(normalized_face, axis=0)

                prediction = loaded_model.predict(input_face)

                
                predicted_class_indices = np.argsort(prediction)[0][::-1]
                predicted_class_probs = prediction[0][predicted_class_indices]

                text_x, text_y = calculate_text(class_names[predicted_class_indices[0]], x1, y1, width, height)

                for i, class_index in enumerate(predicted_class_indices):

                    if i >= max_classes:
                        break

                    class_name = class_names[class_index]
                    class_prob = predicted_class_probs[i]

                    class_prob_text = f"{class_name}: {class_prob:.2f}"
                    prob_text_y = text_y + (i * 20)  # Spazio tra le probabilità di classe

                    cv2.putText(frame, class_prob_text, (text_x, prob_text_y), font, font_scale, (0, 255, 0), thickness)

                cv2.rectangle(frame, (x1, y1, width, height), (255, 0, 0), 1)

        cv2.imshow(f"video recognition", frame)
        key = cv2.waitKey(1)

        # press 'q' to quit the loop
        if key == ord('q'):
            break

'''
def live_predict(model, class_names):

    loaded_model = model


    # initialize the object camera (0 indicate main pc camera)
    camera = cv2.VideoCapture('Utils\\video.mp4')

    # Disabilita i messaggi di log generati da YOLO
    logging.disable(logging.CRITICAL)

    # Load a pretrained YOLO model (recommended for training)
    yolo_model = YOLO('Utils\\YoloFaceV8\\yolov8n-face.pt')
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

        predictions = yolo_model.predict(source=frame, device=0, max_det=4, conf=0.70, verbose=False)

        faces = []

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

                face = frame[y1:y2, x1:x2]

                # se la ROI è più piccola di 64x64 applico un resize con interpolazione bicubica, migliore nella costruzione di pixel aggiuntivi
                if width < 64 or height < 64:
                    face = cv2.resize(face,(64,64), interpolation=cv2.INTER_CUBIC)
                # se la ROI è più grande di 64x64 aplico un resize con interpolazione ad area, migliore nella riduzione delle dimensioni con perdita minima di qualità
                elif width > 64 or height > 64:
                    face = cv2.resize(face,(64,64), interpolation=cv2.INTER_AREA) 

                normalized_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                input_face = tf.expand_dims(normalized_face, axis=0)

                prediction = loaded_model.predict(input_face)

                predicted_class_index = np.argmax(prediction)
                


                text_x, text_y = calculate_text(class_names[predicted_class_index], x1, y1, width, height)

                cv2.putText(frame, class_names[predicted_class_index], (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
                cv2.rectangle(frame, (x1, y1, width, height), (255, 0, 0), 1)


        cv2.imshow(f"video recognition", frame)
        key = cv2.waitKey(1)

        #press 'q' to quit the loop
        if key == ord('q'):
            break
'''