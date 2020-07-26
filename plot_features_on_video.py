import cv2
import tensorflow as tf
from detect_human_features import ROOT, define_labels, unnormalize, normalize
import os

VID = os.path.join(ROOT, "videoData", "videoplayback.mp4")
HEIGHT, WIDTH, CHANNELS = (218, 178, 1)
haarcascade = os.path.join(ROOT, "haarcascade_frontalface_default.xml")
model = tf.keras.models.load_model("model1.h5")

def detect_features(file=VID):
    cap = cv2.VideoCapture(VID)

    ret = True
    while(ret):
        # Read through the frames
        ret, frame = cap.read()

        if(frame is not None):
            # Make the frame gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Use CascadeClassifier to detect faces
            face_cascade = cv2.CascadeClassifier(haarcascade)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                model_input = gray[y:y+HEIGHT, x:x+WIDTH]
                # Reshape & normalize
                model_input, _, _ = normalize(model_input, None, None)
                try:
                    model_input = model_input.reshape(-1, 218, 178, 1)
                    assert model_input.shape == (-1, 218, 178, 1)
                except:
                    pass
                predict_labels = model.predict(model_input)[0]
                x_predict, y_predict = define_labels(predict_labels)
                _, x_predict, y_predict = unnormalize(0, x_predict, y_predict)

                # Draw Labels
                for xp, yp in zip(x_predict, y_predict):
                    cv2.circle(frame, (int(x+xp), int(y+yp)), 4, (255, 255, 255), -1)
                cv2.rectangle(frame,(x,y),(x+WIDTH,y+HEIGHT),(255,0,0),2)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_features()