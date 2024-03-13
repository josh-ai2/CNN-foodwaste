

"""##CNN model implementaton;
We don't have 'before' photos so we train resnet on an open dataset of Chinese food. This is ground truth

**Helpful links:**
https://medium.com/@onur_andros_ozbek/food-101-classifier-using-resnet50-on-colab-f0cc6ac6487 (training resnet on food classifier)
https://data.mendeley.com/datasets/fspyss5zbb/1 (dataset)
https://www.aboutdatablog.com/post/how-to-successfully-add-large-data-sets-to-google-drive-and-use-them-in-google-colab (uploading dataset into google drive, and accessing)

**Input:** The model takes an input image with dimensions 600x600 pixels.

**ResNet-50 Feature Extraction:** ResNet-50 is used as a feature extractor, processing the image through its convolutional and residual layers.

**Global Average Pooling:** The output feature maps from ResNet-50 are condensed into a smaller set of values using global average pooling.

**Dense Layer:** A fully connected dense layer with 1024 neurons follows, using ReLU (Rectified Linear Unit) activation function to introduce non-linearity.

**Output Layer:** The final dense layer consists of 240 neurons (if there are 240 classes to classify) with a softmax activation function, which outputs the probabilities for each class.

**Training:**The model is trained on a custom dataset, adjusting its weights to minimize the loss function, which measures the difference between the predicted and true labels.

**Prediction:** After training, the model can take a new image, process it through these layers, and predict the class label, completing the flow from raw data to an actionable result.
"""

# "preprocessing:"
# import os
# import cv2
# from tqdm import tqdm
#
# # Paths to the original and new directories for train and validation sets
# src_train_directory = '/Users/joshmanto/Downloads/CNN-foodwaste/CNFOOD-241/train600x600'  # Replace with your path
# dst_train_directory = '/Users/joshmanto/Downloads/CNN-foodwaste/train-end2'  # Replace with your path
# src_val_directory = '/Users/joshmanto/Downloads/CNN-foodwaste/CNFOOD-241/val600x600'  # Replace with your path
# dst_val_directory = '/Users/joshmanto/Downloads/CNN-foodwaste/val-end'  # Replace with your path
#
# def resize_and_save_images(src_directory, dst_directory, target_size=(224, 224), total_size_gb=None):
#     """
#     Resizes images from the source directory and saves them to the destination directory.
#     """
#     if total_size_gb:
#         size_limit_bytes = total_size_gb * 1024 ** 3  # Convert GB to bytes
#     else:
#         size_limit_bytes = float('inf')  # No size limit if total_size_gb is None
#
#     # Ensure the target directory exists
#     if not os.path.exists(dst_directory):
#         os.makedirs(dst_directory)
#
#     total_size_bytes = 0  # Keep track of the size of processed images
#     image_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(src_directory) for f in filenames if
#                    f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#
#     # Create a progress bar instance with tqdm
#     with tqdm(total=min(size_limit_bytes, sum(os.path.getsize(f) for f in image_files)), unit='B', unit_scale=True,
#               desc="Resizing images") as pbar:
#         for file_path in image_files:
#             if total_size_bytes >= size_limit_bytes:
#                 break
#
#             image = cv2.imread(file_path)
#             if image is None:
#                 continue  # Skip files that aren't valid images
#
#             # Resize and save image
#             resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
#             resized_file_path = os.path.join(dst_directory, os.path.relpath(file_path, src_directory))
#
#             # Create subdirectory if it doesn't exist
#             os.makedirs(os.path.dirname(resized_file_path), exist_ok=True)
#
#             cv2.imwrite(resized_file_path, resized_image)
#             file_size = os.path.getsize(resized_file_path)
#             total_size_bytes += file_size
#
#             # Update the progress bar
#             pbar.update(file_size)
#
#     # Return the total size of processed images in GB
#     return total_size_bytes / 1024 ** 3
#
# total_train_size_gb = resize_and_save_images(src_train_directory, dst_train_directory, total_size_gb=3)
# total_val_size_gb = resize_and_save_images(src_val_directory, dst_val_directory, total_size_gb=1.18)
#
# print(f"Total training set size: {total_train_size_gb} GB")
# print(f"Total validation set size: {total_val_size_gb} GB")


# /usr/bin/python3 /Users/joshmanto/Downloads/CNN-foodwaste/reset_food_waste_image_recognition.py
# Resizing images: 3.22GB [09:28, 5.67MB/s]
# Resizing images:  41%|████▏     | 487M/1.18G [01:24<02:00, 5.73MB/s]
# Total training set size: 3.000010601244867 GB
# Total validation set size: 0.4531932659447193 GB
#
# Process finished with exit code 0

#second attempt to preprocessing:
# import os
# import cv2
# from tqdm import tqdm
#
# def resize_and_save_images(src_directory, dst_directory, target_size=(224, 224)):
#     """
#     Resizes all images from the source directory and saves them to the destination directory.
#     """
#     # Ensure the target directory exists
#     if not os.path.exists(dst_directory):
#         os.makedirs(dst_directory)
#
#     image_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(src_directory) for f in filenames if
#                    f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#
#     # Create a progress bar instance with tqdm
#     with tqdm(total=len(image_files), desc="Resizing images", unit='files') as pbar:
#         for file_path in image_files:
#             image = cv2.imread(file_path)
#             if image is None:
#                 pbar.update(1)
#                 continue  # Skip files that aren't valid images
#
#             # Resize and save image
#             resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
#             resized_file_path = os.path.join(dst_directory, os.path.relpath(file_path, src_directory))
#
#             # Create subdirectory if it doesn't exist
#             os.makedirs(os.path.dirname(resized_file_path), exist_ok=True)
#
#             cv2.imwrite(resized_file_path, resized_image)
#
#             # Update the progress bar
#             pbar.update(1)
#
#
# # Example usage:
# src_train_directory = '/Users/joshmanto/Downloads/CNN-foodwaste/CNFOOD-241/train600x600'  # Replace with the path to your unzipped train images
# dst_train_directory = '/Users/joshmanto/Downloads/CNN-foodwaste/train-end2'  # Replace with the path where you want to save resized images
#
# resize_and_save_images(src_train_directory, dst_train_directory)


#Model start:
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import legacy

# Instantiate the legacy Adam optimizer with your desired parameters
adam_optimizer = legacy.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam'
)

# Data generators
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Setup train and validation generators
train_generator = train_datagen.flow_from_directory(
    '/Users/joshmanto/Downloads/CNN-foodwaste/train-end2',  # Update to your training directory path
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical') #label is based on the name of the folder

val_generator = val_datagen.flow_from_directory(
    '/Users/joshmanto/Downloads/CNN-foodwaste/val-end',  # Validation directory path
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical') #label is based on the name of the folder

# Load and customize ResNet-50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(241, activation='softmax')(x)  # adjusted to 240 classes
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,  # Number of epochs to train for
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size)



