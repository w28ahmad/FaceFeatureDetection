import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from detect_human_features import get_labels, gray_image, show_image, get_data, define_labels, unnormalize, draw_labels

model = tf.keras.models.load_model("model1.h5")
IMAGE = np.random.randint(1000, 200000, size=1)[0]

def selectImage():
    print("IMAGE NUMBER: ", IMAGE)
    hmap = get_labels(end=IMAGE, start=IMAGE-1)
    return hmap

def detect_points(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_gray = img_gray.reshape(-1, 218, 178, 1)
    result = model.predict(img_gray)[0]
    return result

def run_test():
    hmap = selectImage()
    filename = list(hmap.keys())[0]
    hmap = {filename : hmap[filename]}
    
    # Make sure the picture is colored and not grayscale
    X_train, y_train = get_data(hmap, color=1, addRotation=True, addTranslation=True, addZoom=True)
    # predict the image
    predict_labels = detect_points(X_train[0])
    
    # Correct Labels
    x_vals, y_vals = define_labels(y_train[0])
    # predict Lables
    x_predict, y_predict = define_labels(predict_labels)

    image = X_train[0]

    # Correct Labels
    image, x_vals, y_vals = unnormalize(image, x_vals, y_vals)
    # predict Labels
    _, x_predict, y_predict = unnormalize(0, x_predict, y_predict)

    # Circle the labels
    # image = draw_labels(image, x_vals, y_vals)
    image = draw_labels(image, x_predict, y_predict, color=(0, 100, 100))
    show_image(image.astype(np.uint8))

if __name__ == "__main__":
    run_test()


