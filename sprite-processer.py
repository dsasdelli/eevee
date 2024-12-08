from helper import GENERATIONS, DIR_BASE, DIR_PROCESSED
from os import listdir, path, makedirs
from PIL import Image, ImageOps

def alphaToWhite(im):
    nim = Image.new("RGBA", im.size, "WHITE")
    nim.paste(im, (0, 0), im)  
    return nim.convert('RGB')

def crop(im):
    return im.crop(ImageOps.invert(im).getbbox())

def resize(im, size=256):
    scale = size / max(im.size)
    nim = Image.new(im.mode, (size, size), "WHITE")
    rim = im.resize((int(im.width * scale), int(im.height * scale)), resample=Image.BOX)
    nim.paste(rim, ((size-rim.width)//2, (size-rim.height)//2))
    return nim

for generation in GENERATIONS.items():
    dir_base = f'{DIR_BASE}\\{generation[1]}'
    dir_processed = f"{DIR_PROCESSED}\\{generation[1]}"
    if(not path.exists(dir_processed)):
        makedirs(dir_processed)
        print(f"Criando diretório de destino: {dir_processed}")
    for f in listdir(dir_base):
        im = Image.open(f'{dir_base}\\{f}').convert("RGBA")
        print(f"Processando {dir_base}\\{f}")
        im_processed = resize(crop(alphaToWhite(im)))
        im_processed.save(f'{dir_processed}\\{f}')