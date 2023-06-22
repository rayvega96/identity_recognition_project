import os
import random
import shutil

# Cartella principale contenente le classi
directory = "Resized\\"

# Cartelle di destinazione per i set di addestramento, validazione e convalida
train_dir = "Dataset\\Training\\"
validation_dir = "Dataset\\Validation\\"
test_dir = "Dataset\\Test\\"

# Percentuale di immagini per i set
train_percentage = 0.6
validation_percentage = 0.2
test_percentage = 0.2

# Elenco delle classi
classi = os.listdir(directory)

# Itera su ogni classe e suddividi le immagini
for classe in classi:
    print(f"Elaboro la classe [{classe}]")

    # Path completo alla classe
    classe_path = os.path.join(directory, classe)
    
    # Elenco delle immagini per la classe corrente
    immagini = os.listdir(classe_path)
    
    # Shuffle delle immagini
    random.shuffle(immagini)
    
    # Calcola le dimensioni dei set
    num_immagini = len(immagini)
    num_train = int(train_percentage * num_immagini)
    num_validation = int(validation_percentage * num_immagini)
    num_test = num_immagini - num_train - num_validation
    
    # Dividi le immagini in set
    train_images = immagini[:num_train]
    validation_images = immagini[num_train:num_train+num_validation]
    test_images = immagini[num_train+num_validation:]
    
    # Crea le cartelle di destinazione se non esistono
    os.makedirs(os.path.join(train_dir, classe), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, classe), exist_ok=True)
    os.makedirs(os.path.join(test_dir, classe), exist_ok=True)
    
    # Sposta le immagini nei rispettivi set
    for img in train_images:
        src_path = os.path.join(classe_path, img)
        dest_path = os.path.join(train_dir, classe, img)
        shutil.move(src_path, dest_path)
    
    for img in validation_images:
        src_path = os.path.join(classe_path, img)
        dest_path = os.path.join(validation_dir, classe, img)
        shutil.move(src_path, dest_path)
    
    for img in test_images:
        src_path = os.path.join(classe_path, img)
        dest_path = os.path.join(test_dir, classe, img)
        shutil.move(src_path, dest_path)
