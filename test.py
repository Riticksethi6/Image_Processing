import cv2
import numpy as np
import math

# Task 1: Affine Transformation
image = cv2.imread('/Users/riticksethi/Desktop/HKA3rdsem/Efficient_video_coding/Affine_Original_Image.jpg')
array = np.zeros(image.shape, image.dtype)

height, width, channels = image.shape

image1 = image.copy()
image2 = image.copy()
image3 = array.copy()
image4 = image.copy()

pts1 = ([50, 50], [width-21, 20], [25, height-19])
pts2 = ([157, 188], [width//2, 10], [(width//4)*3, (height//10)*9])

for img in [image2, array, image3]:
    for i, (x, y) in enumerate(pts1):
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        if i != 1:
            p, q = x + 10, y + 10
        else:
            p, q = x - 160, y + 25
        cv2.putText(img, '({}, {})'.format(x, y), (p, q), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

for i, (x, y) in enumerate(pts2):
    cv2.circle(image3, (x, y), 4, (0, 255, 0), -1)
    if i != 1:
        p, q = x + 10, y + 10
    else:
        p, q = x + 10, y + 30
    cv2.putText(image3, '({}, {})'.format(x, y), (p, q), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

for i, j in zip(pts1, pts2):
    cv2.arrowedLine(image3, tuple(i), tuple(j), (255, 0, 0), 2, cv2.LINE_AA)

M = cv2.getAffineTransform(np.float32(pts1), np.float32(pts2))
image4 = cv2.warpAffine(image4, M, (width, height))

# Task 2: Perspective Transformation
image5 = cv2.imread('/Users/riticksethi/Desktop/HKA3rdsem/Efficient_video_coding/Perspective_Original_Image.jpg')
array2 = np.zeros(image5.shape, image5.dtype)

height2, width2, channels2 = image5.shape

image6 = image5.copy()
image7 = array2.copy()
image8 = image5.copy()

pts3 = ([0, 0], [width2-1, 0], [0, height2-1], [width2-1, height2-1])
pts4 = ([157, 188], [width2//2, 10], [(width2//4)*3, (height2//10)*9], [700, height2-51])

for img in [image6, array2, image7]:
    for i, (x, y) in enumerate(pts3):
        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
        if i == 0:
            p, q = x + 10, y + 25
        elif i == 1:
            p, q = x - 160, y + 25
        elif i == 2:
            p, q = x + 10, y - 10
        else:
            p, q = x - 180, y - 15
        cv2.putText(img, '({}, {})'.format(x, y), (p, q), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

for i, (x, y) in enumerate(pts4):
    cv2.circle(image7, (x, y), 4, (0, 255, 0), -1)
    if i == 0:
        p, q = x + 10, y + 10
    elif i == 1:
        p, q = x - 160, y + 20
    elif i == 2:
        p, q = x - 200, y - 10
    else:
        p, q = x -150, y - 20
    cv2.putText(image7, '({}, {})'.format(x, y), (p, q), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

for i, j in zip(pts3, pts4):
    cv2.arrowedLine(image7, tuple(i), tuple(j), (255, 0, 0), 2, cv2.LINE_AA)

M2 = cv2.getPerspectiveTransform(np.float32(pts3), np.float32(pts4))
image8 = cv2.warpPerspective(image8, M2, (width2, height2))

# Task 3: Image Resizing
image9 = cv2.imread('/Users/riticksethi/Desktop/HKA3rdsem/Efficient_video_coding/Resizing_Original_Image_increase.jpg')
image10 = cv2.imread('/Users/riticksethi/Desktop/HKA3rdsem/Efficient_video_coding/Resizing_Original_Image_increase.jpg')

interpolation_type2 = ['cv2.INTER_NEAREST',
                       'cv2.INTER_LINEAR',
                       'cv2.INTER_CUBIC',
                       'cv2.INTER_AREA',
                       'cv2.INTER_LANCZOS4',
                       'cv2.INTER_LINEAR_EXACT']

# for increasing size
interpolation_big = [cv2.resize(image9, (800, 800), interpolation=eval(item)) for item in interpolation_type2]

# for decreasing size
interpolation_small = [cv2.resize(image10, (500, 500), interpolation=eval(item)) for item in interpolation_type2]

# Task 4: Image Flipping
image11 = cv2.imread('/Users/riticksethi/Desktop/HKA3rdsem/Efficient_video_coding/Flipping_Original_Image.jpg')

# Method 1
rotate_plus90 = cv2.rotate(image11, cv2.ROTATE_90_CLOCKWISE)
rotate_minus90 = cv2.rotate(image11, cv2.ROTATE_90_COUNTERCLOCKWISE)
rotate_180 = cv2.rotate(image11, cv2.ROTATE_180)

# Method 2
mat_default = np.array(([0, 0], [width - 1, 0], [0, height - 1]), dtype=np.float32)
mat_1 = np.array(([width - 1, 0], [0, 0], [width - 1, height - 1]), dtype=np.float32)
mat_0 = np.array(([0, height - 1], [width - 1, height - 1], [0, 0]), dtype=np.float32)
mat_minus1 = np.array(([width - 1, height - 1], [0, height - 1], [width - 1, 0]), dtype=np.float32)

Mat_1 = cv2.getAffineTransform(mat_default, mat_1)
Mat_0 = cv2.getAffineTransform(mat_default, mat_0)
Mat_minus1 = cv2.getAffineTransform(mat_default, mat_minus1)

# Method 3
mat_default2 = np.array(([0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]), dtype=np.float32)
mat_2 = np.array(([width - 1, 0], [0, 0], [width - 1, height - 1], [0, height - 1]), dtype=np.float32)
mat_minus2 = np.array(([width - 1, height - 1], [0, height - 1], [width - 1, 0], [0, 0]), dtype=np.float32)

Mat_2 = cv2.getPerspectiveTransform(mat_default2, mat_2)
Mat_minus2 = cv2.getPerspectiveTransform(mat_default2, mat_minus2)

rotate_90 = cv2.warpAffine(image11, Mat_1, (width, height))
rotate_180_2 = cv2.warpAffine(image11, Mat_0, (width, height))
rotate_minus90_2 = cv2.warpAffine(image11, Mat_minus1, (width, height))
rotate_90_3 = cv2.warpPerspective(image11, Mat_2, (width, height))
rotate_minus90_3 = cv2.warpPerspective(image11, Mat_minus2, (width, height))

# Task 5: Blending and Transition
image12 = cv2.imread('/Users/riticksethi/Desktop/HKA3rdsem/Efficient_video_coding/Original_Image.jpg')
image13 = cv2.imread('/Users/riticksethi/Desktop/HKA3rdsem/Efficient_video_coding/Pyramid_Original_Image_Up.jpg')

# Resize image13 to match the dimensions of image12
image13_resized = cv2.resize(image13, (image12.shape[1], image12.shape[0]))

# Blend the resized image13 with image12
blended = cv2.addWeighted(image12, 0.7, image13_resized, 0.3, 0)

# Task 6: Arithmetic Operations
image14 = cv2.imread('/Users/riticksethi/Desktop/HKA3rdsem/Efficient_video_coding/Warping_Original_Image.jpg')

# Convert the image to float32
image14_float = image14.astype(np.float32)

# Perform the multiplication operation
multiplication = cv2.multiply(image14_float, np.ones(image14.shape, dtype=np.float32) * 1.5)

# Convert the result back to uint8
multiplication_result = np.clip(multiplication, 0, 255).astype(np.uint8)

# Perform the addition operation
addition = cv2.add(image14, np.ones(image14.shape, dtype=np.uint8) * 50)

# Perform the subtraction operation
subtraction = cv2.subtract(image14, np.ones(image14.shape, dtype=np.uint8) * 50)

# Perform the division operation
division = cv2.divide(image14_float, np.ones(image14.shape, dtype=np.float32) * 1.5)

# Task 7: Bitwise Operations
image15 = cv2.imread('/Users/riticksethi/Desktop/HKA3rdsem/Efficient_video_coding/Perspective_Original_Image.jpg')
image16 = cv2.imread('/Users/riticksethi/Desktop/HKA3rdsem/Efficient_video_coding/Pyramid_Original_Image_Down.jpg')

# Resize image16 to match the dimensions of image15
image16_resized = cv2.resize(image16, (image15.shape[1], image15.shape[0]))

# Perform bitwise operation after resizing
bit_and = cv2.bitwise_and(image15, image16_resized)
bit_or = cv2.bitwise_or(image15, image16_resized)
bit_xor = cv2.bitwise_xor(image15, image16_resized)
bit_not1 = cv2.bitwise_not(image15)
bit_not2 = cv2.bitwise_not(image16_resized)

# Task 8: Thresholding
image17 = cv2.imread('/Users/riticksethi/Desktop/HKA3rdsem/Efficient_video_coding/Rotation_Original_Image.jpg')
image18 = cv2.cvtColor(image17, cv2.COLOR_BGR2GRAY)

_, th1 = cv2.threshold(image18, 127, 255, cv2.THRESH_BINARY)
_, th2 = cv2.threshold(image18, 127, 255, cv2.THRESH_BINARY_INV)
_, th3 = cv2.threshold(image18, 127, 255, cv2.THRESH_TRUNC)
_, th4 = cv2.threshold(image18, 127, 255, cv2.THRESH_TOZERO)
_, th5 = cv2.threshold(image18, 127, 255, cv2.THRESH_TOZERO_INV)

# Task 9: Edge Detection
image19 = cv2.imread('/Users/riticksethi/Desktop/HKA3rdsem/Efficient_video_coding/Translation_Original_Image.jpg')
image20 = cv2.cvtColor(image19, cv2.COLOR_BGR2GRAY)

edge1 = cv2.Canny(image20, 100, 200)
edge2 = cv2.Canny(image20, 30, 100)

# Displaying the images
cv2.imshow('Affine Transformation 1', image1)
cv2.imshow('Affine Transformation 2', image2)
cv2.imshow('Affine Transformation 3', image3)
cv2.imshow('Affine Transformation 4', image4)

cv2.imshow('Perspective Transformation 1', image5)
cv2.imshow('Perspective Transformation 2', image6)
cv2.imshow('Perspective Transformation 3', image7)
cv2.imshow('Perspective Transformation 4', image8)

for i, (big, small) in enumerate(zip(interpolation_big, interpolation_small)):
    cv2.imshow('Image Resizing (Increasing) - ' + interpolation_type2[i], big)
    cv2.imshow('Image Resizing (Decreasing) - ' + interpolation_type2[i], small)

cv2.imshow('Image Flipping 1', rotate_plus90)
cv2.imshow('Image Flipping 2', rotate_minus90)
cv2.imshow('Image Flipping 3', rotate_180)
cv2.imshow('Image Flipping 4', rotate_90)
cv2.imshow('Image Flipping 5', rotate_180_2)
cv2.imshow('Image Flipping 6', rotate_minus90_2)
cv2.imshow('Image Flipping 7', rotate_90_3)
cv2.imshow('Image Flipping 8', rotate_minus90_3)

cv2.imshow('Blending and Transition', blended)

cv2.imshow('Arithmetic Operations - Multiplication', multiplication_result)
cv2.imshow('Arithmetic Operations - Addition', addition)
cv2.imshow('Arithmetic Operations - Subtraction', subtraction)
cv2.imshow('Arithmetic Operations - Division', division.astype(np.uint8))

cv2.imshow('Bitwise Operations 1', bit_and)
cv2.imshow('Bitwise Operations 2', bit_or)
cv2.imshow('Bitwise Operations 3', bit_xor)
cv2.imshow('Bitwise Operations 4', bit_not1)
cv2.imshow('Bitwise Operations 5', bit_not2)

cv2.imshow('Thresholding 1', th1)
cv2.imshow('Thresholding 2', th2)
cv2.imshow('Thresholding 3', th3)
cv2.imshow('Thresholding 4', th4)
cv2.imshow('Thresholding 5', th5)

cv2.imshow('Edge Detection 1', edge1)
cv2.imshow('Edge Detection 2', edge2)

cv2.waitKey(0)
cv2.destroyAllWindows()
