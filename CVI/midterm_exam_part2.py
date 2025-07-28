import cv2
import numpy as np

def apply_invisible_cloak(frame, background):
    # convert to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define green color ranges and create masks
    lower_green = (30, 100, 50)
    upper_green = (90, 255, 255)
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)


    # combine masks and refine if needed
    kernel = np.ones((5, 5), np.uint8)
    #cv2.morphologyEx() helps removes noise from the mask
    #MORPH_OPEN removes small objects from the mask, 
    #while MORPH_DILATE expands the white regions to cover the cloak area
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # create inverse mask and isolate cloak area
    inverse_mask = cv2.bitwise_not(mask)
    # cv2.bitwise_and() helps to isolate the cloak area from the current frame
    cloak_area = cv2.bitwise_and(frame, frame, mask=inverse_mask)
    #isolate non-cloak area from the background
    nonCloak_area = cv2.bitwise_and(background, background, mask=mask)

    # combine background with current frame
    final_output = cv2.add(cloak_area, nonCloak_area)

    return final_output


cap = cv2.VideoCapture(0)
#create a named window to display the output
cv2.namedWindow("Cloak Effect")
# capture background (press 'b' to save it)
for i in range(60):  # capture 30 frames for background
    ret, background = cap.read()
    if not ret:
        print("Failed to capture background. Exiting.")
        cap.release()
        cv2.destroyAllWindows()
        exit()
    background = cv2.flip(background, 1)  # flip the background for a mirror effect

while True:
    ret, background = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('b'):
        break
index = 0
while True:
    ret, frame = cap.read()
    output = apply_invisible_cloak(frame, background)
    cv2.imshow("Cloak Effect", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('s'):
        img_name = 'output' + str(index) + '.jpg'
        index += 1
        print("Saving output image...")
        cv2.imwrite(img_name, output)

cap.release()
cv2.destroyAllWindows()
