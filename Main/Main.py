# By Nitro
# MIT License

import numpy as np
import cv2
import math

# Haarcascade
faceCascade = cv2.CascadeClassifier('Frontal_Face.xml')
handCascade = cv2.CascadeClassifier('Hand.xml')

# Video Capture
cap = cv2.VideoCapture(0)

# Main filter and detection loop
while True:
    ret, img = cap.read()

    # Create Binary Mask
    HSV_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    YCbCr_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    binary_mask_image = HSV_image

    # Skin tone color range, you might need to change this value based on race, or environment (light exposure).
    lower_HSV_values = np.array([0, 40, 0], dtype="uint8")
    upper_HSV_values = np.array([25, 255, 255], dtype="uint8")
    lower_YCbCr_values = np.array((0, 138, 67), dtype="uint8")
    upper_YCbCr_values = np.array((255, 173, 133), dtype="uint8")

    # A binary mask
    mask_YCbCr = cv2.inRange(YCbCr_image, lower_YCbCr_values, upper_YCbCr_values)
    mask_HSV = cv2.inRange(HSV_image, lower_HSV_values, upper_HSV_values)

    binary_mask_image = cv2.add(mask_HSV, mask_YCbCr)

    image_foreground = cv2.erode(binary_mask_image, None, iterations=2)  # Remove noise
    dilated_binary_image = cv2.dilate(binary_mask_image, None, iterations=2)
    ret, image_background = cv2.threshold(dilated_binary_image, 1, 128, cv2.THRESH_BINARY)  # Set all background regions to 128

    image_marker = cv2.add(image_foreground, image_background)  # Generate markers.
    image_marker32 = np.int32(image_marker)  # Convert to 32SC1 format

    cv2.watershed(img, image_marker32)
    m = cv2.convertScaleAbs(image_marker32)  # Convert back to uint8

    # Bitwise the mask with the input image
    ret, image_mask = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    output = cv2.bitwise_and(img, img, mask=image_mask)


    CamW = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    CamH = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    FaceSize = CamW / 9
    Gap = 20
    #cv2.rectangle(output, (int(round(CamW / 2 - FaceSize + Gap)), int(round(CamH / 2 - FaceSize + Gap))), (int(round(CamW / 2 + FaceSize - Gap)), int(round(CamH / 2 + FaceSize - Gap))), (15, 196, 241), 4)
    #cv2.rectangle(output, (int(round(CamW / 2 - FaceSize - Gap)), int(round(CamH / 2 - FaceSize - Gap))), (int(round(CamW / 2 + FaceSize + Gap)), int(round(CamH / 2 + FaceSize + Gap))), (34, 126, 230), 4)

    # Detect
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(output, (x, y), (x + w, y + h), (176, 144, 39), 4)

    LHands = handCascade.detectMultiScale(gray, 1.1, 6)
    for (x, y, w, h) in LHands:
        cv2.rectangle(output, (x, y), (x + w, y + h), (101, 186, 148), 4)

    #RHands = RHand_cascade.detectMultiScale(gray, 1.3, 5)
    #for (x, y, w, h) in RHands:
    #    cv2.rectangle(output, (x, y), (x + w, y + h), (101, 186, 148), 4)

    cv2.imshow('Main', output)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()