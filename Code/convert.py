import cv2
import os
from mtcnn.mtcnn import MTCNN
from glob import glob
from PIL import Image
import numpy as np
import skimage.io
from random import randrange
detector = MTCNN()


#helper funcation for inflation method 
def inflate(path):
  img = cv2.imread(path)
  results = detector.detect_faces(img)
  if len(results) != 0:
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    img = img[y1:y2, x1:x2]
  img = cv2.resize(img,(224,224))
  return img


#inflation method
def massInflate(all_images):
  for x in all_images:
    img = inflate(x)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im1 = Image.fromarray(img)
    im1 = im1.save(x)
    print(x)


#rotation method
def massRotate(all_images):
  angles = [-55, 55, -145, 145]
  for x in all_images:
    img = cv2.imread(x)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rot = skimage.transform.rotate(img, angles[randrange(len(angles) - 1)], preserve_range=True).astype(np.uint8)


#all images in dataset
all_images=glob('train-rotated-faces/' + '*/*/*.jpg')


#final function to rename all family folders by adding an x before the name
fam = glob('train-rotated-faces/*')
for x in fam:
  cut = x[20:]
  print(cut)
  os.rename(x, 'train-rotated-faces/x' + cut)



