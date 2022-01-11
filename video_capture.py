import cv2

# Opens the inbuilt camera of laptop to capture video.
cap = cv2.VideoCapture("./videos/L.mp4")
i = 0

while (cap.isOpened()):
    ret, frame = cap.read()

    # This condition prevents from infinite looping
    # incase video ends.
    if ret == False:
        break
    print("save image:", i);
    # Save Frame by Frame into disk using imwrite method
    img = cv2.bilateralFilter(frame, 9, 75, 75)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img, (224, 224))
    ret1, image = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    cv2.imwrite("./data/" + 'L_' + str(i) + '.jpg', image)
    i += 1

cap.release()
cv2.destroyAllWindows()