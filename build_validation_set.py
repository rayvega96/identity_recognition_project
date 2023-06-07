import os
import random
import shutil

#FC 35999, DS 32963, SC 27034, GM 30715

# Imposta il percorso della cartella contenente il training set
training_set_dir = 'training_set'

# Imposta il percorso delle cartelle del validation set
validation_set_dir = 'validation_set'
os.makedirs(validation_set_dir, exist_ok=True)

# Imposta la frazione delle immagini da assegnare al validation set
validation_ratio = 0.2

# Elenca le cartelle delle classi nel training set
class_folders = ['Davide_Sgroi', 'Stefano_Corrao', 'Francesco_Conti', 'Gabriele_Musso', 'Unknown']

# Trova il numero minimo di immagini tra tutte le classi
min_images_per_class = float('inf')
for class_folder in class_folders:
    class_folder_path = os.path.join(training_set_dir, class_folder)
    num_images = len(os.listdir(class_folder_path))
    min_images_per_class = min(min_images_per_class, num_images)

# Itera attraverso le cartelle delle classi
for class_folder in class_folders:

    print(f"Processing '{class_folder}' folder.")

    # Imposta il percorso della cartella della classe nel training set
    class_folder_path = os.path.join(training_set_dir, class_folder)
    
    # Ottieni la lista di immagini nella cartella della classe
    images = os.listdir(class_folder_path)
    
    # Calcola il numero di immagini da assegnare al validation set
    num_validation_images = int(len(images) * validation_ratio)
    
    # Seleziona casualmente le immagini da spostare nel validation set
    validation_images = random.sample(images, num_validation_images)
    
    # Itera attraverso le immagini di validazione
    for image in validation_images:
        # Imposta il percorso dell'immagine nel training set
        image_path = os.path.join(class_folder_path, image)
        
        # Sposta l'immagine nel validation set
        destination_path = os.path.join(validation_set_dir, class_folder, image)
        shutil.move(image_path, destination_path)