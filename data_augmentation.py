import cv2
import keras.preprocessing
import os

def get_filespath(folder_path):
    files_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith("jpg"):
                file_path = os.path.join(root, file)
                files_list.append((file_path, file))
    files_list.sort()
    return files_list

# Image cropping

lista_immagini = get_filespath("../../datasets/unknown_6k/")
save_path = "augmented_training_set/"

def crop_work(file_path, save_path=None):

    n_files = len(file_path)
    print("Started Cropping Process...")

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

crop_work(lista_immagini, save_path)