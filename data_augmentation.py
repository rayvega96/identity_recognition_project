import cv2
from keras.preprocessing.image import ImageDataGenerator
import os
from video_recognition import yolo_face_recognition
import numpy as np
import time

def get_filespath(folder_path):
    files_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith("jpg"):
                file_path = os.path.join(root, file)
                files_list.append((file_path, file))
    files_list.sort()
    return files_list

# Face recognition
def get_faces(images_list, save_path):


    n_files = len(images_list)

    counter = 1

    for i in range(n_files):

        single_time = time.time()

        faces_found = yolo_face_recognition(images_list[i][0])

        for face in faces_found:

            if counter < 10:
                cv2.imwrite(f"{save_path}0{counter}.jpg", face)
            else:
                cv2.imwrite(f"{save_path}0{counter}.jpg", face)
            counter += 1
        
        total_single_time = time.time() - single_time
        eta = total_single_time * (n_files - i+1)
        os.system('clear')
        print("Started Face Recognition Process...")
        print(f"Elaborating Image {i}/{n_files}")
        print(f"ETA: {str(int(eta // 60))}:{str(int(eta % 60))} seconds.")
    
    print("Face Recognition Process finished.")

# Face cropping to 64x64
def crop_work(file_path, save_path=None):

    n_files = len(file_path)

    for i in range(n_files):
        os.system("clear")
        print("Started Cropping Process...")
        print(f"Cropping Image {i+1}/{n_files}")
        image = cv2.imread(file_path[i][0])

        height, width = image.shape[0], image.shape[1]

        # se la ROI è più piccola di 64x64 applico un resize con interpolazione bicubica, migliore nella costruzione di pixel aggiuntivi
        if width < 64 or height < 64:
            image = cv2.resize(image,(64,64), interpolation=cv2.INTER_CUBIC)
        # se la ROI è più grande di 64x64 aplico un resize con interpolazione ad area, migliore nella riduzione delle dimensioni con perdita minima di qualità
        elif width > 64 or height > 64:
            image = cv2.resize(image,(64,64), interpolation=cv2.INTER_AREA) 

        #savepath must finish with /
        cv2.imwrite(f"{save_path}{file_path[i][1]}", image)
    
    print("Cropping Process finished.")


def data_augment(file_path, save_path):

    n_files = len(file_path)

    # Crea un oggetto ImageDataGenerator per la data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,              # Rotazione casuale tra -20 e +20 gradi
        width_shift_range=0.1,          # Spostamento orizzontale casuale tra -10% e +10%
        height_shift_range=0.1,         # Spostamento verticale casuale tra -10% e +10%
        brightness_range=(0.5, 0.7),    # Variazione casuale della luminosità
        channel_shift_range=75.0,
        shear_range=0.2,                # Taglio casuale tra -20 e +20 gradi
        zoom_range=0.2,                 # Zoom casuale tra 80% e 120%
        horizontal_flip=True,           # Ribaltamento orizzontale casuale
        fill_mode='nearest'             # Modalità di riempimento dei pixel per le trasformazioni
    )

    for i in range(n_files):
        os.system('clear')
        print("Augmentation Process Started...")
        print(f"Augmenting Image {i+1}/{n_files}")
        image = cv2.imread(file_path[i][0])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_name = file_path[i][1].rstrip(".jpg")

        # Add a new dimension to preserve the ImageDataGenerator format
        image = np.expand_dims(image, axis=0)

        # Generate a new image with the datagen format specified above
        augmented_images = datagen.flow(image, batch_size=5,save_prefix=image_name+'-aug', save_to_dir=save_path, save_format="jpg")
        # Generate 5 images
        for j in range(5):
            next(augmented_images)





lista_immagini = get_filespath("../../datasets/unknown_14k/")
save_path = "augmented_training_set/"
get_faces(lista_immagini, save_path)

faces_path = get_filespath(save_path)
crop_work(faces_path, save_path)

cropped_path = get_filespath("augmented_training_set/")

print(cropped_path)

data_augment(cropped_path, save_path)

