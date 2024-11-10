import tensorflow as tf
import os
from tf.keras.models import Model
from tf.keras.layers import Dense, Flatten, Dropout
from tf.tensorflow.keras.preprocessing.image import ImageDataGenerator

class training:
    def __init__(self, model = 'models/model.h5', train_path = "artifacts\training", valid_path = 'artifacts\training', batch_size = 32, epochs = 10):
        self.model = model
        self.train_path = train_path
        self.valid_path = valid_path
        self.batch_size = batch_size
        self.epochs = epochs
        train_gen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

# Image data generators for training and validation
        train_gen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        validation_gen = ImageDataGenerator(rescale=1./255)

        # Ensure the correct paths for training and validation directories
        train_generator = train_gen.flow_from_directory(
            self.train_path,  # Make sure this path is correct
            target_size=(160, 160), 
            batch_size=batch_size,
            class_mode='categorical',  # Use 'sparse' for sparse categorical crossentropy
            color_mode='rgb',
            shuffle=True
        )

        val_generator = validation_gen.flow_from_directory(
            self.valid_path,  # Update to the correct validation path
            target_size=(160, 160),
            batch_size=batch_size,
            class_mode='categorical',  # Use 'sparse' for sparse categorical crossentropy
            color_mode='rgb',
            shuffle=False,
        )
        
        new_model = Model(inputs=model.input, outputs=model.layers[-4].output)

        # Freeze all layers
        for layer in new_model.layers:
            layer.trainable = False
            
        x = Dense(220, activation='relu')(new_model.output)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(4, activation='softmax')(x)
        
        final_model = Model(inputs=new_model.input, outputs=output)

        # Compile the model
        final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Fit the model
        final_model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=10,
            validation_data=val_generator,
            validation_steps=val_generator.samples // batch_size
        )
        
        final_model.save("artifacts/models/model.h5")