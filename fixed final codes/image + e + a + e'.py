import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from keras import Sequential, Input
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, Concatenate
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
import tensorflow as tf


def create_model(input_shape_img, input_shape_x, input_shape_y, input_shape_z):
    # Define input layers for each type of data
    input_img = Input(shape=input_shape_img, name='input_img')
    input_x = Input(shape=input_shape_x, name='input_x')
    input_y = Input(shape=input_shape_y, name='input_y')
    input_z = Input(shape=input_shape_z, name= 'input_z')

    # 3D convolutional layer for images
    conv3d_img = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(input_img)
    print("Shape of conv3d_img:", conv3d_img.shape)
    maxpool_img = MaxPooling3D(pool_size=(2, 2, 2))(conv3d_img)

    # Flatten the convolutional output
    flattened_img = Flatten()(maxpool_img)

    # Fully connected layers for input_x
    dense_x = Dense(32, activation='relu')(input_x)

    # Fully connected layers for input_y
    dense_y = Dense(32, activation='relu')(input_y)

    dense_z =Dense(32, activation='relu')(input_z)

    # Flatten the dense layers for input_x and input_y
    flattened_x = Flatten()(dense_x)
    flattened_y = Flatten()(dense_y)
    flattened_z = Flatten()(dense_z)
    # Flatten the convolutional output
    flattened_img = Flatten()(maxpool_img)

    # Concatenate the flattened convolutional output with flattened_x and flattened_y
    concatenated = Concatenate()([flattened_img, flattened_x, flattened_y,flattened_z])

    dense = Dense(128, activation='relu')(concatenated)
    dropout = Dropout(0.5)(dense)

    # Output layer
    output = Dense(1, activation='sigmoid')(dropout)  # Change the number of units for your specific problem

    # Create a model that takes all three inputs and produces the output
    model = tf.keras.Model(inputs=[input_img, input_x, input_y,input_z], outputs=output)

    # Plot the model architecture and save it as an image
    plot_model(model, to_file='fullmodel.png', show_shapes=True, show_layer_names=True)
    

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_and_evaluate_model(images, x_values, y_values, z_values, labels):
    # Initialize k-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    all_fprs = []
    all_tprs = []
    all_confusion_matrices = []
    all_training_loss = []
    all_validation_loss = []

    for train_index, test_index in kfold.split(images):
        x_train_img, x_test_img = images[train_index], images[test_index]
        x_train_x, x_test_x = x_values[train_index], x_values[test_index]
        x_train_y, x_test_y = y_values[train_index], y_values[test_index]
        x_train_z, x_test_z = z_values[train_index], z_values[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Extract input shapes from the training data
        input_shape_img = x_train_img.shape[1:]
        input_shape_x = x_train_x.shape[1:]
        input_shape_y = x_train_y.shape[1:]
        input_shape_z = x_train_z.shape[1:]

        model = create_model(input_shape_img, input_shape_x, input_shape_y,input_shape_z)
        model.summary()
        history = model.fit(
            [x_train_img, x_train_x, x_train_y,x_train_z],
            y_train,
            epochs=20,
            batch_size=32,
            validation_data=([x_test_img, x_test_x, x_test_y,x_test_z], y_test)
        )
        all_training_loss.append(history.history['loss'])
        all_validation_loss.append(history.history['val_loss'])
        # Evaluate the model
        _, accuracy = model.evaluate([x_test_img, x_test_x, x_test_y,x_test_z], y_test)
        accuracies.append(accuracy)

        y_pred_prob = model.predict([x_test_img, x_test_x, x_test_y,x_test_z])

        # Calculate ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        all_fprs.append(fpr)
        all_tprs.append(tpr)

        # Calculate confusion matrix
        y_pred = (y_pred_prob > 0.5).astype(int)
        confusion_matrix_result = confusion_matrix(y_test, y_pred)
        all_confusion_matrices.append(confusion_matrix_result)

    # Calculate average accuracy
    average_accuracy = np.mean(accuracies)

    # Combine all TPRs and FPRs from different folds
    combined_fpr = np.concatenate(all_fprs)
    combined_tpr = np.concatenate(all_tprs)
    plot_loss(all_training_loss, all_validation_loss)
    return average_accuracy, accuracies, (combined_fpr, combined_tpr), all_confusion_matrices

def plot_loss(all_training_loss, all_validation_loss):
    plt.figure(figsize=(10, 6))

    for fold, (training_loss, validation_loss) in enumerate(zip(all_training_loss, all_validation_loss), start=1):
        plt.plot(range(1, len(training_loss) + 1), training_loss, label=f'Fold {fold} Training Loss', marker='o')
        plt.plot(range(1, len(validation_loss) + 1), validation_loss, label=f'Fold {fold} Validation Loss', marker='o')

    plt.title('Training and Validation Loss Over Epochs (All Folds)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
def load_data(gif_folder, verhoogdedruk, verlaagdedruk):
    # Read Excel files
    df = pd.read_excel(verhoogdedruk, sheet_name='Blad1', usecols='A,X,Y,Z', skiprows=lambda x: x > 106)
    df2 = pd.read_excel(verlaagdedruk, sheet_name='Blad1', usecols='A,X,Y,Z', skiprows=lambda x: x > 103)

    # Extracting numbers from gif filenames and matching them in df and df2
    data = []
    for filename in os.listdir(gif_folder):
        if filename.endswith('.gif'):
            patient_number = int(filename.split('_')[3].split('.')[0])
            if patient_number in df['studienummer'].values:
                label = 1
                x_value = df[df['studienummer'] == patient_number]['E'].values[0]
                y_value = df[df['studienummer'] == patient_number]['A'].values[0]
                z_value = df[df['studienummer'] == patient_number]['e septaal'].values[0]
            elif patient_number in df2['studienummer'].values:
                label = 0
                x_value = df2[df2['studienummer'] == patient_number]['E'].values[0]
                y_value = df2[df2['studienummer'] == patient_number]['A'].values[0]
                z_value = df2[df2['studienummer'] == patient_number]['e septaal'].values[0]
            else:
                label = -1  # Use -1 as a placeholder for gifs with unknown labels
                x_value = np.nan
                y_value = np.nan
                z_value = np.nan
            if pd.notna(x_value) and pd.notna(y_value) and pd.notna(z_value):
                img_path = os.path.join(gif_folder, filename)
                frames = load_gif_frames(img_path, target_size=(224, 224))  # Load all frames
                data.append((frames, x_value, y_value,z_value, label))

    # Convert data to numpy arrays
    frame_arrays = np.array([item[0] for item in data])
    frame_arrays = np.expand_dims(frame_arrays, axis=-1)
    x_values = np.array([item[1] for item in data])
    y_values = np.array([item[2] for item in data])
    z_values = np.array([item[3] for item in data])
    labels = np.array([item[4] for item in data])

    # Reshape x_values and y_values to (186, 1)
    x_values = x_values.reshape(-1, 1)
    y_values = y_values.reshape(-1, 1)
    z_values = z_values.reshape(-1,1)
    return frame_arrays, x_values, y_values, z_values, labels


def load_gif_frames(gif_path, num_frames=20, target_size=(224, 224)):
    frames = []
    with Image.open(gif_path) as img:
        for i in range(min(num_frames, img.n_frames)):
            img.seek(i)
            frame = img.copy()

            # Resize the frame to the target size
            frame = frame.resize(target_size, Image.LANCZOS)

            # Convert to grayscale
            frame = frame.convert('L')

            frame_array = np.array(frame)
            frames.append(frame_array)
    return frames


def plot_confusion_matrix(confusion_matrix, labels, ds_type):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.4)
    ax = sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    ax.set_title(f'Confusion matrix of action recognition for {ds_type}')
    ax.set_xlabel('Predicted Action')
    ax.set_ylabel('Actual Action')
    plt.xticks(np.arange(len(labels)) + 0.5, labels)
    plt.yticks(np.arange(len(labels)) + 0.5, labels)
    plt.show()


def main():
    # File paths
    gif_folder = r'C:\Users\oskar\Downloads\OneDrive_2023-07-19\Def 4kamer 209 inclusies (zonder 5k)'
    elevated_pressure = r'C:\Users\oskar\Downloads\Def Inclusie VERHOOGDE DRUK EchoAI_versie 7.23 (cave zonder 5k).xlsx'
    normal_pressure = r'C:\Users\oskar\Downloads\Def Inclusie NORMALE DRUK EchoAI_versie 7.23 (cave zonder 5k).xlsx'

    images, x_values, y_values, z_values, labels = load_data(gif_folder, elevated_pressure, normal_pressure)
    assert images.shape[0] == len(x_values) == len(y_values) == len(labels), "Data shapes do not match."
    assert x_values.shape == (len(x_values), 1), "x_values have an unexpected shape."
    assert y_values.shape == (len(y_values), 1), "y_values have an unexpected shape."
    assert images.shape[0] == len(labels), "Number of images and labels do not match."

    print("Loaded Data:")
    print("Number of Images:", len(images))
    print("Number of X Values:", len(x_values))
    print("Number of Y Values:", len(y_values))
    print("Number of Labels:", len(labels))
    print()
    print("Shape of x_values:", x_values.shape)
    print("Shape of y_values:", y_values.shape)
    print("Shape of images: ", images.shape)

    average_accuracy, accuracies, roc_data, all_confusion_matrices = train_and_evaluate_model(
        images, x_values, y_values,z_values, labels)

    plt.figure()

    # Combine all TPRs and FPRs from different folds
    combined_fpr, combined_tpr = roc_data

    # Sort the combined data based on FPR
    sorted_indices = np.argsort(combined_fpr)
    combined_fpr = combined_fpr[sorted_indices]
    combined_tpr = combined_tpr[sorted_indices]

    roc_auc = auc(combined_fpr, combined_tpr)
    plt.plot(combined_fpr, combined_tpr, lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()
    confusion_matrices = []  # Initialize an empty list to store the confusion matrices

    for fold_number in range(1, 6):  # Assuming you have 5 folds
        # Compute the confusion matrix for the current fold
        # Replace the following line with your actual computation
        current_fold_confusion_matrix = np.array([[10, 6], [5, 17]])  # Replace with actual data

        labels = ["Class 0", "Class 1"]  # Replace with your actual class labels
        ds_type = f"Fold {fold_number}"  # Update with the appropriate dataset type

        # Append the current fold's confusion matrix to the list
        confusion_matrices.append(current_fold_confusion_matrix)

        # Plot the confusion matrix for the current fold
        plot_confusion_matrix(current_fold_confusion_matrix, labels, ds_type)

    # Print accuracies and average accuracy
    print("Accuracies:", accuracies)
    print("Average Accuracy:", average_accuracy)
    print("Fpr:", combined_fpr)
    print("Fpr:", combined_tpr)
    for i, confusion_matrix_result in enumerate(all_confusion_matrices):
        print(f"Confusion Matrix - Fold {i + 1}:")
        print(confusion_matrix_result)

if __name__ == "__main__":
    main()
