from TreeClassifier import TreeClassifier

import numpy as np
import cv2
import pyautogui

class TreeDetector():
    def __init__(self):
        self.prediction_min = 2.0
        self.window_size = 300
        self.classifier = TreeClassifier()
        self.classifier.load_model()

    def get_found_locations(self, img):
        locations = []

        for x in range(0, img.shape[0] - self.window_size, img.shape[0] // 25):
            for y in range(0, img.shape[1] - self.window_size, img.shape[1] // 25):
                window = img[x: min(x + self.window_size, img.shape[0]), y: min(y + self.window_size, img.shape[1])]
                window = cv2.resize(window, (self.window_size // 2, self.window_size // 2))
                predictions = self.classifier.predict_from_image(window)
                prediction_idx = np.argmax(predictions[0])
                if predictions[0][prediction_idx] >= self.prediction_min:
                    print(f'Added ({x}, {y}): {predictions[0][prediction_idx]}')
                    locations.append((x, y))

        i = 0
        for location in locations:
            cv2.imwrite(f'test_{i}.png', img[location[0]: location[0] + self.window_size, location[1]: location[1] + self.window_size])
            i += 1

        return locations

test = TreeDetector()
img = np.float32(pyautogui.screenshot())
cv2.imwrite('test.png', img)
# img = cv2.resize(img, (img.shape[0] // 4, img.shape[1] // 4))
print(test.get_found_locations(np.float32(img)))