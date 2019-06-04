import cv2
import imutils
from skimage.filters import threshold_local
import numpy as np
import argparse

##TAKEN THIS FUNCTION FROM PYIMAGE##
def four_point_transform(image, rect):
    # obtain a consistent order of the points and unpack them
    # individually
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


#taking input
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path of image")
args = vars(ap.parse_args())
img = cv2.imread(args["image"])
img = cv2.resize(img,(500,500))
copy_img = img.copy()
cp_img = img.copy()
orig = img.copy()
ratio = img.shape[0] / 500.0
#grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#blurred=cv2.GaussianBlur(gray_image,(5,5),0)
edge_image = cv2.Canny(gray_image,30,50)
#edge_image = cv2.bitwise_not(edge_image)
#cv2.imshow("canny", edge_image)
#cv2.waitKey(0)
ret,thresh = cv2.threshold(edge_image,127,255,cv2.THRESH_BINARY_INV)
contour, heirarchy = cv2.findContours(edge_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contour = sorted(contour,key = cv2.contourArea,reverse=True)[:50]

for c in contour:
    acc = 0.01 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, acc, True)
    if len(approx) == 4:
        cv2.drawContours(copy_img, [approx], 0, (0,255,0),2)
        break

#cv2.imshow("out", copy_img)

#cv2.waitKey(0)
cv2.imshow("app", copy_img)
#cv2.waitKey(0)
approx = np.float32(approx)
warped = four_point_transform(orig, approx.reshape(4,2) * ratio)

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

# show the original and scanned images
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)

