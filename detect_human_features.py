import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import tensorflow as tf
from scipy.ndimage import zoom

# Defining paths
ROOT = os.getcwd()
HUMANS_DIR = os.path.join(ROOT, "humans", "img_align_celeba")
HUMAN_PICS = os.listdir(HUMANS_DIR)
OUTPUT_DIR = os.path.join(ROOT, "output")

# Image Dimentions, trained with     v --> grayscale
HEIGHT, WIDTH, CHANNELS = (218, 178, 1)

DATAFILE = "list_landmarks_align_celeba.txt"



# Returns feature point hashmap
def get_labels(end,start=2):
    hmap = dict()
    with open(DATAFILE, "r") as f:
        for line in f.readlines()[start:end]:
            line = line.strip('\n')
            data = line.split(" ")
            hmap[data[0]] = [int(datapoint) for datapoint in data[1:] if datapoint !='']
    return hmap

# Return grayscale images from filename
def gray_image(filename):
    img_path = os.path.join(HUMANS_DIR, filename)
    img_gray = mpimg.imread(img_path)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
    img_gray = np.asarray(img_gray, dtype=np.float32)

    return img_gray

# Returns color images from filename
def color_image(filename):
    img_path = os.path.join(HUMANS_DIR, filename)
    image = mpimg.imread(img_path)
    return image

# Saves image to the humains directory or output director
def save_image(image, image_name, training=1):
    # Select where to save
    file = os.path.join(HUMANS_DIR, image_name)
    if not training:
        file = os.path.joing(OUTPUT_DIR, image_name)

    im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(file, im_rgb)
    print("SAVED")

# Zooms into the center of the image by a zoom_factor
def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

# Performs some translation on the x, y labels and the image
def perform_translation(image, x_vals, y_vals, width_offset, height_offset):
    # Translation on the image
    T = np.float32([[1, 0, width_offset],
                    [0, 1, height_offset]])
    image = cv2.warpAffine(image, T, (WIDTH, HEIGHT))

    # Translation on the x_vals
    for i in range(len(x_vals)):
        x_vals[i] += width_offset

    # Translation on the y_vals
    for i in range(len(y_vals)):
        y_vals[i] += height_offset

    return image, x_vals, y_vals


# Performs some zoom on the x, y labels and the image
def perform_zoom(image, x_vals, y_vals, zoom_factor):
    # Zoom on the image
    image = clipped_zoom(image, zoom_factor)

    # Zoom on the x_vals
    for i,x in enumerate(x_vals):
        x_vals[i] = WIDTH/2 - (WIDTH/2 - x) * zoom_factor

    # Zoom on the y_vals
    for i,y in enumerate(y_vals):
        y_vals[i] = HEIGHT/2 - (HEIGHT/2 - y) * zoom_factor

    return image, x_vals, y_vals    

# Performs some rotation on the x, y labels and the image
def perform_rotation(image, x_vals, y_vals):
    # ignore for now
    return image, x_vals, y_vals

# Turn the points from [x1, y1, x2, y2] -> [x1, x2, ...] , [y1, y2, ...]
def define_labels(labels):
    x_vals = []
    y_vals = []
    # Define labels
    for i, val in enumerate(labels):
        if i%2 == 0:
            x_vals.append(val)
        else:
            y_vals.append(val)

    assert(len(x_vals) == len(y_vals))
    return x_vals, y_vals

def normalize(image, x_vals, y_vals):
    image = np.asarray(image, dtype=np.float32)
    x_vals = np.asarray(x_vals, dtype=np.float32)
    y_vals = np.asarray(y_vals, dtype=np.float32)

    # Normalize
    image /= 255
    x_vals = (x_vals - WIDTH/2)/(WIDTH/2)
    y_vals = (y_vals - HEIGHT/2)/(HEIGHT/2)

    return image, x_vals, y_vals

def unnormalize(image, x_vals, y_vals):
    image = np.asarray(image, dtype=np.float32)
    x_vals = np.asarray(x_vals, dtype=np.float32)
    y_vals = np.asarray(y_vals, dtype=np.float32)

    # Normalize
    image *= 255
    x_vals = (x_vals)*(WIDTH/2) + WIDTH/2
    y_vals = (y_vals)*(HEIGHT/2) + HEIGHT/2

    return image, x_vals, y_vals


# Returns the training and validation data
def get_data(hmap, color=0):
    X = []
    y = []

    for filename in hmap.keys():
        # random transaltional offset
        WIDTH_OFFSET = np.random.randint(-WIDTH/6, WIDTH/6, size=1) [0]
        HEIGHT_OFFSET = np.random.randint(-HEIGHT/6, HEIGHT/6, size=1)[0]
        # random zoom offset
        ZOOM_FACTOR = round(np.random.uniform(1.2, 1.7), 1)


        x_vals, y_vals = define_labels(hmap[filename])
        image = gray_image(filename)
        if color:
            image = color_image(filename)

        # # Add rotation
        image, x_vals, y_vals = perform_rotation(image, x_vals, y_vals)
        # # Add translation
        image, x_vals, y_vals = perform_translation(image, x_vals, y_vals, WIDTH_OFFSET, HEIGHT_OFFSET)
        # # Add magnification
        image, x_vals, y_vals = perform_zoom(image, x_vals, y_vals, ZOOM_FACTOR)

        # Normalize
        image, x_vals, y_vals = normalize(image, x_vals, y_vals)

        # color is not for training, no need to reshape
        if not color:
            image = image.reshape(218,178,1)
        features = []
        for i in range(len(x_vals)):
            features.append(x_vals[i])
            features.append(y_vals[i])

        X.append(image)
        y.append(features)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return X, y

# display the image
def show_image(img):
    plt.imshow(img)
    plt.draw()
    plt.waitforbuttonpress()
    plt.close()

# Draw circles at those x and y points
def draw_labels(image, x_vals, y_vals, color=(255, 255, 255)):
    for x, y in zip(x_vals, y_vals):
        cv2.circle(image, (int(x), int(y)), 2, color, -1)
    return image

# view the training data
def test_input_data(data_idx=np.random.randint(0, 98, size=1)[0]):
    hmap = get_labels(100)
    filename = list(hmap.keys())[data_idx]
    hmap = {filename : hmap[filename]}
    
    X_train, y_train = get_data(hmap)
    x_vals, y_vals = define_labels(y_train[0])
    image = X_train[0].reshape(-1, 218, 178)[0]
    image, x_vals, y_vals = unnormalize(image, x_vals, y_vals)

    # Circle the labels
    image = draw_labels(image, x_vals, y_vals)
    show_image(image)

def create_model(X_train):
    model = tf.keras.models.Sequential()

    # # conv layer with 16, 3 by 3 filters for the inputs
    model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=X_train.shape[1:]))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

    # conv layer with 32, 3 by 3 filters from previous layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

    # conv layer with 64, 3 by 3 filters from previous layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

    # conv layer with 128, 3 by 3 filters from previous layer
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

    # conv layer with 256, 3 by 3 filters from previous layer
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

    # Convert all the values to 1D array
    model.add(tf.keras.layers.Flatten())

    # Add 512 size neural network
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))

    # Output layer
    model.add(tf.keras.layers.Dense(10))

    return model

def train_model():
    model_name = 'model1.h5'
    checkpoint_name = "checkpoint1.hdf5"
    data_size = 100

    for i in range(1, 16):
        if(model_name in os.listdir('.')):
            print("LOADING CURRENT MODEL")
            hmap = get_labels(i*100+100, i*100)
            X_train, y_train = get_data(hmap)
            model = tf.keras.models.load_model(model_name)
        else:
            print("CREATING NEW MODEL")
            hmap = get_labels(100)
            X_train, y_train = get_data(hmap)
            model = create_model(X_train)


        print("Training datapoint shape: X_train.shape:{}".format(X_train.shape))
        print("Training labels shape: y_train.shape:{}".format(y_train.shape))

        epochs = 40
        batch_size = 15

        hist = tf.keras.callbacks.History()

        logdir = "logs/scalars/" + datetime.now().strftime("%Yj%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_name, verbose=1, save_best_only=True)

        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        model.fit = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint, hist, tensorboard_callback], verbose=1)
        model.save(model_name)


if __name__ == "__main__":
    # Viewing the test data
    # test_input_data()
    
    # Running the model
    train_model()
    