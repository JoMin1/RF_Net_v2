import cv2
import os

import sys

if len(sys.argv) < 2:
	print('Please input data category')

print('Argument List: {}'.format(str(sys.argv)))

ctg = sys.argv[1]


print('You asked {} to augment data'.format(ctg))

FOLDER=r"./data/{}/train/0.normal".format(ctg)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)

    return images

def rotate(img):
	return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def flip(img, f: 0):
	return cv2.flip(img, 0)

def resize(img, h, w):
	return cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_AREA)

images = []


print("agumenting dataset from {}".format(FOLDER))
for img in load_images_from_folder(FOLDER):
	img90 = rotate(img)
	img180 = rotate(img90)
	img270 = rotate(img180)
	
	images.append(img90)
	images.append(img180)
	images.append(img270)
	images.append(flip(img, f=0))
	images.append(flip(img, f=1))
	images.append(flip(img, f=-1))
	# images.append(resize(img,512,512))



print('saving agumented data...')
num = 0
for img in images:
	# cv2.imwrite('{}\\aug_{}.png'.format(FOLDER, num), img)
	cv2.imwrite(r'./data/{}/train/0.normal/ag-{:03d}.png'.format(ctg,num), img)
	num = num + 1






