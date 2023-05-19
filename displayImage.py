import cv2

!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg  # link rename
im = cv2.imread("./input.jpg")
cv2_imshow(im)
