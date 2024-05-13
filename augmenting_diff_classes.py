from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# path = 'test_patch_SYD copy/diff'
path = '/Users/jessy/Documents/MATLAB/third study/Australian Patch/test_patch_SYD/diff'
n = 35
for i in range(n):
    image_path = os.path.join(path, f'diff{i+1}.png')
    image = Image.open(image_path)
    img = np.array(image)
    plt.imshow(img,cmap=plt.cm.gray)
    plt.show()

    # contrast
    # enhancer = ImageEnhance.Contrast(image)
    # im_ = enhancer.enhance(factor=0.05)
    # im_.save(f"{i}c.png")

    # brightness
    # enhance_b = ImageEnhance.Brightness(image)
    # im_b = enhance_b.enhance(factor=1.3)
    # im_b.save(f"{i}b.png")

    # sharpness
    enhancer_s = ImageEnhance.Sharpness(image)
    im_s = enhancer_s.enhance(2)
    im_s.save(f"{i}s.png")


    # rotation
    im_45 = image.rotate(45)
    im_45.save(f"{i}r45.png")


    im_90 = image.rotate(90)
    im_90.save(f"{i}r90.png")

    im_145 = image.rotate(145)
    im_145.save(f"{i}r145.png")

    im_180 = image.rotate(180)
    im_180.save(f"{i}r180.png")

    # flip image
    im_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
    im_flip.save(f"{i}flip.png")

    im_flip_v = image.transpose(Image.FLIP_TOP_BOTTOM)
    im_flip_v.save(f"{i}flip.png")




