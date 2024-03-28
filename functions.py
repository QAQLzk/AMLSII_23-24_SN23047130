import os
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#ignore warning
import warnings
warnings.filterwarnings("ignore")



def data_preprocessing_and_split(work_dir, input_shape):

    # import train lable and image
    labels = pd.read_csv(os.path.join(work_dir, "train.csv"))
    print(Counter(labels['label']))

    # seperate train(90%)  and test(10%) dataset
    train_data, test_data = train_test_split(labels, test_size=0.1, random_state=66)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))


    #visualisation of datasets
    for i in ['top', 'right', 'left']:
        ax.spines[i].set_visible(False)
    ax.spines['bottom'].set_color('black')

    # Ensure the palette matches the number of unique labels
    n_colors = train_data['label'].nunique()
    palette = list(reversed(sns.color_palette("viridis", n_colors)))

    sns.countplot(x='label', data=labels, edgecolor='black', palette=palette)
    plt.xlabel('Classes', fontfamily='serif', size=15)
    plt.ylabel('Count', fontfamily='serif', size=15)
    plt.xticks(fontfamily='serif', size=12)
    plt.yticks(fontfamily='serif', size=12)
    ax.grid(axis='y', linestyle='--', alpha=0.9)
    plt.title("The Visualisation of Dataset")
    plt.savefig("results/dataset_view.png")


    BATCH_SIZE = 8
    TARGET_SIZE = input_shape

    #match label and image
    train_data.label = train_data.label.astype('str')
    # data augmentation
    train_datagen = ImageDataGenerator(validation_split = 0.2,
                                        preprocessing_function = None,
                                        rotation_range = 45,
                                        zoom_range = 0.2,
                                        horizontal_flip = True,
                                        vertical_flip = True,
                                        fill_mode = 'nearest',
                                        height_shift_range = 0.1,
                                        width_shift_range = 0.1)

    train_generator = train_datagen.flow_from_dataframe(train_data,
                            directory = os.path.join(work_dir, "train_images"),
                            subset = "training",
                            x_col = "image_id",
                            y_col = "label",
                            target_size = (TARGET_SIZE, TARGET_SIZE),
                            batch_size = BATCH_SIZE,
                            class_mode = "categorical")


    validation_datagen = ImageDataGenerator(validation_split = 0.2)

    validation_generator = validation_datagen.flow_from_dataframe(train_data,
                            directory = os.path.join(work_dir, "train_images"),
                            subset = "validation",
                            x_col = "image_id",
                            y_col = "label",
                            target_size = (TARGET_SIZE, TARGET_SIZE),
                            batch_size = BATCH_SIZE,
                            class_mode = "categorical")

    test_data.label = test_data.label.astype('str')

    test_datagen = ImageDataGenerator()

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_data,
        directory=os.path.join(work_dir, "train_images"),
        x_col="image_id",
        y_col="label",
        target_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # visualisation of data augmentation
    generator = train_datagen.flow_from_dataframe(train_data.iloc[10:11],
                        directory = os.path.join(work_dir, "train_images"),
                        x_col = "image_id",
                        y_col = "label",
                        target_size = (TARGET_SIZE, TARGET_SIZE),
                        batch_size = BATCH_SIZE,
                        class_mode = 'categorical')

    aug_images = [generator[0][0][0]/255 for i in range(4)]
    fig, axes = plt.subplots(2, 2, figsize = (5,5))
    axes = axes.flatten()
    for img, ax in zip(aug_images, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("results/image_after_augmentation.png")
    print("The graph of overview of model has generated")

    return train_generator, validation_generator, test_generator


def build_model(input_shape):

    def create_model():
    
        model = Sequential()
        # initialize the model with input shape
        model.add(EfficientNetB3(input_shape = (input_shape, input_shape, 3), include_top = False,
                                weights = 'imagenet',
                                drop_connect_rate=0.6))
        model.add(GlobalAveragePooling2D())
        model.add(Flatten())
        model.add(Dense(256, activation = 'relu', bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
        model.add(Dropout(0.5))
        model.add(Dense(5, activation = 'softmax'))
        
        return model

    leaf_model = create_model()
    leaf_model.summary()

    filename = f'results/model_diagram_{input_shape}.png'
    keras.utils.plot_model(leaf_model, to_file=filename, show_shapes=True, show_layer_names=True)
    print("model graph has generated")

    return leaf_model


# model compiling and training
def model_training_plot(model, train_generator, validation_generator,EPOCHS, STEPS_PER_EPOCH, BATCH_SIZE,VALIDATION_STEPS,input_shape):
    
    leaf_model = model

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False,
                                                    label_smoothing=0.0001,
                                                    name='categorical_crossentropy' )

    leaf_model.compile(optimizer = Adam(learning_rate = 1e-3),
                        loss = loss, #'categorical_crossentropy'
                        metrics = ['categorical_accuracy']) #'acc'

    # Stop training when the val_loss has stopped decreasing for 3 epochs.
    es = EarlyStopping(monitor='val_loss', mode='min', patience=3,
                        restore_best_weights=True, verbose=1)

    # Save the model with the minimum validation loss
    checkpoint_cb = ModelCheckpoint("results/model_checkpoint",
                                    save_best_only=True,
                                    monitor = 'val_loss',
                                    mode='min')

    # reduce learning rate
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                    factor = 0.2,
                                    patience = 2,
                                    min_lr = 1e-6,
                                    mode = 'min',
                                    verbose = 1)

    history = leaf_model.fit(train_generator,
                                validation_data = validation_generator,
                                epochs= EPOCHS,
                                batch_size = BATCH_SIZE,
                                #class_weight = d_class_weights,
                                steps_per_epoch = STEPS_PER_EPOCH,
                                validation_steps = VALIDATION_STEPS,
                                callbacks=[es, checkpoint_cb, reduce_lr])
    
    filename = f'results/trained_model_{input_shape}.h5'

    leaf_model.save(filename)  
    print('model has saved')

    # print the training process
    def Train_Val_Plot(acc,val_acc,loss,val_loss):
        
        fig, (ax1, ax2) = plt.subplots(1,2, figsize= (15,10))
        fig.suptitle(" MODEL'S METRICS VISUALIZATION TS_512 ", fontsize=20)

        ax1.plot(range(1, len(acc) + 1), acc)
        ax1.plot(range(1, len(val_acc) + 1), val_acc)
        ax1.set_title('History of Accuracy', fontsize=15)
        ax1.set_xlabel('Epochs', fontsize=15)
        ax1.set_ylabel('Accuracy', fontsize=15)
        ax1.legend(['training', 'validation'])


        ax2.plot(range(1, len(loss) + 1), loss)
        ax2.plot(range(1, len(val_loss) + 1), val_loss)
        ax2.set_title('History of Loss', fontsize=15)
        ax2.set_xlabel('Epochs', fontsize=15)
        ax2.set_ylabel('Loss', fontsize=15)
        ax2.legend(['training', 'validation'])

        plt.savefig(f"results/train_val_plot_{input_shape}.png")
        print("training and validation graph has saved")
        

    Train_Val_Plot(history.history['categorical_accuracy'],history.history['val_categorical_accuracy'],
                history.history['loss'],history.history['val_loss'])
    
    best_train_acc=  max(history.history['categorical_accuracy'])
    best_val_acc = max(history.history['val_categorical_accuracy'])
    print('Train_Cat-Acc: ', max(history.history['categorical_accuracy']))
    print('Val_Cat-Acc: ', max(history.history['val_categorical_accuracy']))

    return leaf_model, best_train_acc, best_val_acc



# test and evaluate model
def test_evaluation(leaf_model,test_generator):

    test_loss, test_accuracy = leaf_model.evaluate(test_generator)
    print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

    # Predict test set
    test_steps_per_epoch = np.math.ceil(test_generator.samples / test_generator.batch_size)
    predictions = leaf_model.predict(test_generator, steps=test_steps_per_epoch)

    predicted_classes = np.argmax(predictions, axis=1)

    # get label
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())   

    # confusion matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes)

    class_accuracies = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    class_accuracies = np.round(class_accuracies, 2)  


    annot = np.empty_like(conf_matrix, dtype=object)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            annot[i, j] = str(conf_matrix[i, j])
        annot[i, i] += "\n(" + str(class_accuracies[i] * 100) + "%)"

    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=annot, fmt="", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, cbar=False)

    plt.title("Confusion Matrix with Accuracy")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.savefig("results/confusion_matrix.png")
    print("confusion matrix has generated")

    return test_accuracy
















