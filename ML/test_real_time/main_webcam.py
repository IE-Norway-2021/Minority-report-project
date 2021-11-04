import cv2
from tensorflow import keras
import numpy as np
from scipy import stats

model = keras.models.load_model('rgb_only_new_weights.h5')
actions = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def tester():
    # 1. New detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    cap = cv2.VideoCapture(0)

    try:
        results = []
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()

            image = cv2.flip(frame, 1)
            image = np.asarray(cv2.resize(image, (160, 120)))
            res = model.predict(image.reshape((1, image.shape[0], image.shape[1], image.shape[2])))

            results.append(np.argmax(res[0]))

            if len(results) == 10:
                print(results)
                print(np.unique(results[0]))
                results = []

            # Show to screen
            cv2.imshow('OpenCV Feed', frame)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    tester()
