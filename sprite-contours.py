from helper import GENERATIONS, DIR_PROCESSED, DIR_PROCESSED_COUNTOURS
from os import listdir, path, makedirs
from PIL import Image
import numpy as np
import cv2 as cv
 


for generation in GENERATIONS.items():
    dir_base = f'{DIR_PROCESSED}\\{generation[1]}'
    dir_processed_countours = f"{DIR_PROCESSED_COUNTOURS}\\{generation[1]}"
    if(not path.exists(dir_processed_countours)):
        makedirs(dir_processed_countours)
        print(f"Criando diretório de destino: {dir_processed_countours}")
    for f in listdir(dir_base):
        print(f"Processando {dir_base}\\{f}")
        im = cv.imread(f'{dir_base}\\{f}', 0)
        out = np.zeros_like(im)
        im = cv.GaussianBlur(im, (9, 9), 0)
        im = cv.adaptiveThreshold(im, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 2)
        contours, hierarchy = cv.findContours(im, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(out, contours, -1, 255, thickness=cv.FILLED)
        cv.imwrite(f'{dir_processed_countours}\\{f}', out)