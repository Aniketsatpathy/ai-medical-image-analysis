import tensorflow as tf

def load_data(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):

    classes = ['normal', 'pneumonia']

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=classes,
        shuffle=True
    )

    val_data = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=classes,
        shuffle=True
    )

    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=classes,
        shuffle=False
    )

    return train_data, val_data, test_data