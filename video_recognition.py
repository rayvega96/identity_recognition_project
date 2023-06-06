# Import the YOLO DL Model
from ultralytics import YOLO
import logging
import cv2
import sys

def yolo_video_recognition(video_path=None, save_path=None, show_preview=True):

    if video_path is not None:
        camera = cv2.VideoCapture(video_path)
    else:
        # initialize the object camera (0 indicate main pc camera)
        camera = cv2.VideoCapture(0)

    # Disabilita i messaggi di log generati da YOLO
    logging.disable(logging.CRITICAL)

    # Load a pretrained YOLO model (recommended for training)
    yolo_model = YOLO('YoloFaceDetector/yolov8n-face.pt')
    yolo_model.verbose = False

    counter = 0

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
                face = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                if save_path is not None: 
                    cv2.imwrite(f"{save_path}/face_{counter}.jpg", face)
                    counter+=1

            if show_preview: 
                cv2.imshow(f"video recognition", frame)
                key = cv2.waitKey(1)

                #press 'q' to quit the loop
                if key == ord('q'):
                    break

    #release the camera and close all windows
    camera.release()
    cv2.destroyAllWindows()

def yolo_face_recognition(face_path):

    image = cv2.imread(face_path)

    # Disable YOLO logging on terminal
    logging.disable(logging.CRITICAL)

    # Load a pretrained YOLO model (recommended for training)
    yolo_model = YOLO('YoloFaceDetector/yolov8n-face.pt')
    yolo_model.verbose = False


    results = yolo_model(image)
    boxes = results[0].boxes

    faces_found = []

    for box in boxes:

        # Check if the probability associated to the predicted face is more than 70
        score = box.conf[0]
        score = score.numpy()

        # If so, save the bounding box as a new image
        if score > 0.80:
 
            top_left_x = int(box.xyxy.tolist()[0][0])
            top_left_y = int(box.xyxy.tolist()[0][1])
            bottom_right_x = int(box.xyxy.tolist()[0][2])
            bottom_right_y = int(box.xyxy.tolist()[0][3])

            face = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            faces_found.append(face)

    return faces_found


if __name__ == '__main__':
    args = sys.argv

    preview = False
    video_path = None
    save_path = None

    if '--preview' in args:
        preview = True
    
    if '--video_path' in args:
        index = args.index("--video_path")
        video_path = args[index+1]
    
    if '--save_path' in args:
        index = args.index("--save_path")
        save_path = args[index+1]


    if args[1] == '--help':
        print("python video_recognition.py (video_path) (save_path) (preview)\n\
              [video_path] = None | path to load video\n\
              [save_path] = None | path to save images\n\
              [preview] = True | False")
    else:
        yolo_video_recognition(video_path=video_path, save_path=save_path, show_preview=preview)
