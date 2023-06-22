from keras.preprocessing.image import ImageDataGenerator

def initialize_dataset(train_dir, val_dir, test_dir):


    # Impostazioni per il generatore di immagini per la data augmentation
    augmented_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        brightness_range=[0.8,1.5],
        horizontal_flip=True
    )

    # Impostazioni per il generatore di immagini
    datagen = ImageDataGenerator(rescale=1./255)

    # Generazione del set di addestramento
    train_generator = augmented_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(256, 256),
        batch_size=5,
        class_mode='categorical',
        shuffle=True
    )

    # Generazione del set di convalida
    validation_generator = datagen.flow_from_directory(
        directory=val_dir,
        target_size=(256, 256),
        batch_size=5,
        class_mode='categorical',
        shuffle=True
    )

    # Generazione del set di test
    test_generator = datagen.flow_from_directory(
        directory=test_dir,
        target_size=(256, 256),
        batch_size=5,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator