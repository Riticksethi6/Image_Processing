import cv2
import numpy as np

# Load the image
image = cv2.imread('/Users/riticksethi/Desktop/HKA3rdsem/Comm/Imgs/Screenshot 2024-04-20 at 16.27.50.png')

# Define the image dimensions
height, width = image.shape[:2]

# Define the transformation parameters
angle = 30        # Rotation angle in degrees
tx, ty = 50, 100  # Translation parameters
scale_x, scale_y = 1.5, 0.5  # Scaling factors
shear_x, shear_y = 0.2, 0.3   # Shearing factors

# Define the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)

# Apply rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# Define the translation matrix
translation_matrix = np.float32([[1, 0, tx],
                                 [0, 1, ty]])

# Apply translation
translated_image = cv2.warpAffine(image, translation_matrix, (width, height))

# Apply scaling
scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y)

# Define the shearing matrix
shearing_matrix = np.float32([[1, shear_x, 0],
                              [shear_y, 1, 0]])

# Apply shearing
sheared_image = cv2.warpAffine(image, shearing_matrix, (width, height))

# Apply reflection (horizontal flip)
reflected_image = cv2.flip(image, 1)

# Display the original and transformed images
cv2.imshow('Original Image', image)
cv2.imshow('Rotated Image', rotated_image)
cv2.imshow('Translated Image', translated_image)
cv2.imshow('Scaled Image', scaled_image)
cv2.imshow('Sheared Image', sheared_image)
cv2.imshow('Reflected Image', reflected_image)

# Wait for key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
