from helper import GENERATIONS, DIR_PROCESSED, DIR_PROCESSED_SHADOWS
from os import listdir, path, makedirs
import numpy as np
import cv2 as cv
 
for generation in GENERATIONS.items():
    dir_base = f'{DIR_PROCESSED}\\{generation[1]}'
    dir_processed_shadows = f"{DIR_PROCESSED_SHADOWS}\\{generation[1]}"
    if(not path.exists(dir_processed_shadows)):
        makedirs(dir_processed_shadows)
        print(f"Criando diretório de destino: {dir_processed_shadows}")
    for f in listdir(dir_base):
        print(f"Processando {dir_base}\\{f}")
        im = cv.imread(f'{dir_base}\\{f}', 0)

        # Cria threshold:
        _, mask = cv.threshold(im, 244, 255, cv.THRESH_BINARY)
        mask_inv = cv.bitwise_not(mask)

        # Acha contornos:
        contours, _ = cv.findContours(mask_inv, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        largest_contour = sorted(contours, key=cv.contourArea, reverse=True)[0]    

        # Cria canvas e desenha:
        out = np.zeros_like(im)
        cv.drawContours(out, [largest_contour], -1, 255, thickness=cv.FILLED)

        # Inverte e salva:
        out = cv.bitwise_not(out)
        cv.imwrite(f'{dir_processed_shadows}\\{f}', out)