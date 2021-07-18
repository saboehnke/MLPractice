import numpy as np
import cv2
import pyautogui
import os

class DatasetCreator():
    def __init__(self, output_directory):
        self.clip_size = (300, 300)
        self.index = 0
        self.output_directory = output_directory

    def take_screenshot(self):
        pos = pyautogui.position()
        image = pyautogui.screenshot()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = image[pos[1] - self.clip_size[1]: pos[1], 
                    pos[0] - self.clip_size[0] // 2: pos[0] + self.clip_size[0] // 2]
        image = cv2.resize(image, (self.clip_size[0] // 2, self.clip_size[1] // 2))
        cv2.imwrite(f'{os.getcwd()}\\{self.output_directory}\\{self.index}.png', image)
        self.index += 1

    def run(self):
        while True:
            self.take_screenshot()

datasetCreator = DatasetCreator('trees\\oak')
datasetCreator.run()
    