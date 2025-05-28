import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
#initialize array to store edited images and action histories
history_img = []
history_actions = []

def display_comparisons(o_img, e_img): 
    fig = plt.figure(figsize=(10,5)) #create new figure 
    plots = fig.subplots(1,2) #create layout for the images. 1 row 2 column

    #pair each image with it's title and display them in a loop
    for plot, img, title in zip(plots, (o_img, e_img), ("Original", "Edit")):
        plot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plot.set_title(title)
        plot.axis('off')

    plt.show()

def update_state(img, action):
    #pushes the edited image and action to the arrays so user can see the list of made edits
    history_img.append(copy.deepcopy(img))
    history_actions.append(action)

def adjust_brightness(img):
    #prompt user to enter beta value between 100 to 100
    #if input is invalid, user will be prompted again until user enter appropriate values
    while True:
        try:
            print("Enter brightness offset (-100 ~ 100): ")
            offset_val = float(input())
        except ValueError:
            print("Invalid number, please enter value between -100 ~ 100: ")
            continue
        if -100 <= offset_val <= 100:
            break
        print("Please enter number within the range of -100 ~ 100:")


    #adjust the brightness using convertScaleAbs function
    edit_img = cv2.convertScaleAbs(img, alpha = 1.0, beta = offset_val)
    
    #log actions and images then display the pre-edit photo and new eddited photo
    update_state(edit_img, f"brightness {offset_val:+}")
    display_comparisons(history_img[-2], edit_img)

    return edit_img

def adjust_contrast(img):
    #prompt user to enter alpha value with range 1.0 to 3.0
    #if input ios invalid, user will be prompted again until user enter appropriate values
    while True:
        try:
            print("Enter contrast value (1.0 = no change; 1.0 ~ 3.0): ")
            alpha = float(input())
        except ValueError:
            print("Invalid value type.")
            continue
        if 0.0 <= alpha <= 3.0:
            break
        print("Please enter number between range 1.0 ~ 3.0")

    edit_img = cv2.convertScaleAbs(img, alpha = alpha, beta = 0)

    #log actions and images then display the pre-edit photo and new eddited photo
    update_state(edit_img, f"contrast x {alpha}")
    display_comparisons(history_img[-2], edit_img)
    
    return edit_img

def convert_to_grayscale(img):
    #converts BGR to single-channel grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #convert img back to BGR so future edits expecting 3 channels will work
    edit_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    #log actions and images then display the pre-edit photo and new eddited photo
    update_state(edit_img, "converted to grayscale")
    display_comparisons(history_img[-2], edit_img)

    return edit_img

def add_padding(img):
    #get original img height and width
    h, w = img.shape[:2]
    #prompt user to enter border type
    while True:
        try:
            print("Choose border type: ")
            print(""" 
                    === Borders ===
                    1. Constant
                    2. Reflect
                    3. Replicate
                    4. Edge        
                    5. Wrap
                """)
            print("Type (enter number): ")
            border_input = int(input())
        except ValueError:
            print("Invalid input, please enter a number between 1 - 5")
            continue
        if 1 <= border_input <= 5:
            break
        print("Please enter an option from 1 ~ 5:")

    #match user input to corresponding border type values
    match border_input:
        case 1:
            border = cv2.BORDER_CONSTANT
        case 2:
            border = cv2.BORDER_REFLECT
        case 3:
            border = cv2.BORDER_REPLICATE
        case 4:
            border = cv2.BORDER_DEFAULT
        case 5:
            border = cv2.BORDER_WRAP
    
    if(border_input == 1):
        while True:
            try:
                #prompt user to choose colour for border
                print("""
                    === Color ===
                        1. Red
                        2. Blue
                        3. Green
                        4. White
                        5. Black   
                    """)
                print("Colour (enter number): ")
                colour_input = int(input())
            except ValueError:
                print("Invalid input.")
                continue
            if 1 <= colour_input <= 5:
                break
            print("Please enter option between 1 ~5.")

        #match user input to corresponding colour
        match colour_input:
            case 1:
                colour = (0,0,255)
            case 2:
                colour = (255,0,0)
            case 3:
                colour = (0,255,0)
            case 4:
                colour = (0,0,0)
            case 5: 
                colour = (255,255,255)
    else:
        colour = (0,0,0)
     

    while True:
        try:
            #prompt user to chose border ratio
            print("""
                === Proportion ===
                    1. Square
                    2. Rectangle
                    3. Custom   
                """)
            print("Type: ")
            ratio_input = int(input())
        except ValueError:
            print("Invalid input.")
            continue
        if 1 <= ratio_input <= 3:
            break
        print("Please enter option within range 1 ~ 3.")

    #assign ratio based on user input
    if(ratio_input == 1):
        ratio = 1.0
    elif (ratio_input == 2):
        ratio = w / h
    else:
        #this is for custom ratio
        #prompt user to enter their target ratio
        print("Enter w:h = ")
        #split the string to store it in an array, ex 1:2 = [1,2]
        num = input().split(":")
        #calculate ratio
        ratio = float(num[0]/ num[1])


    while True:
        try:
            #prompt user to enter padding
            print("Total padding (pixels): ")
            padding_pixels = int(input())
        except ValueError:
            print("Invalid input.")
            continue
        if padding_pixels > 0 :
            break
        print("Please enter a positive value.")

    #calculate new dimensions based on entered desired padding
    new_w = w + padding_pixels * 2 
    new_h = h + padding_pixels * 2
    current_ratio = new_w / new_h #calculate new ratio based on desired padding

    #See if original image need to adjust to user's chosen aspect ratio needs
    #ratio = width / height
    if current_ratio < ratio:
        #current image is too tall
        #calculate the extra width then add it to the left and right axis dimensions
        extra = int((ratio * new_h - new_w) / 2)
        x1 = padding_pixels + extra
        x2 = padding_pixels + extra
        y1 = y2 = padding_pixels
    else:
        #current image is too wide
        #calculate the extra height then add it to the top and bottom axis dimensions
        extra = int((new_w / ratio - new_h) / 2)
        y1 = padding_pixels + extra
        y2 = padding_pixels + extra
        x1 = x2 = padding_pixels

    #make the border with all the values calcualted/input from above
    edit_img = cv2.copyMakeBorder(img, top=y1, bottom= y2, left = x1, right = x2, borderType= border, value= colour)

    #log actions and images then display the pre-edit photo and new eddited photo
    update_state(edit_img, f"padded {padding_pixels}px, ratio={ratio:.2f}, type={border}, colour={colour}")
    display_comparisons(history_img[-2], edit_img)

    return edit_img

def threshold(img):
    #convert img to grayscale for threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #prompt user to choose threshold type
    print("""
           === Threshold ===
            1. BINARY
            2. BINARY_INV  
        """)
    while True:
        try:
            print("Threshold (enter num): ")
            threshold_input = int(input())
        except ValueError:
            print("Invalid input.")
            continue
        if 1 <= threshold_input <= 2:
            break
        print("Please enter either 1 or 2.")

    threshold_type = cv2.THRESH_BINARY if threshold_input == 1 else cv2.THRESH_BINARY_INV

    while True:
        try:
            #prompt user to enter threshold value
            print("Enter threshold value (0-255): ")
            threshold_value = int(input())
        except ValueError:
            print("Invalid Input.")
            continue
        if threshold_input >= 0:
            break
        print("Please enter a number between 0 and 255.")

    #apply threshold function to image, first return value is the retval which is disregarded and not needed
    _, thresh = cv2.threshold(gray, threshold_value, 255, threshold_type)

    #convert image back to 3channel for consistency and future operations
    edit_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    #log actions and images then display the pre-edit photo and new eddited photo
    update_state(edit_img, f"threshold {'BINARY' if threshold_input == 1 else 'INV'}, threshold value: {threshold_value}")
    display_comparisons(history_img[-2], edit_img)

    return edit_img

def blend_image(img):
    #prompt user to enter name or path to second image, then read and load the image
    print("Enter name/path to second image: ")
    img2_path = input()
    img2 = cv2.imread(img2_path)

    #check if image exits, if not then continue to prompt user to reenter the path/name until it's correct
    while img2 is None:
        print("Fail to load image, try again")
        img2_path = input()
        img2 = cv2.imread(img2_path)

    while True:
        try:
            #prompt user to enter the alpha factor
            print("Enter blending factor num (alpha) 0.0 ~ 1.0: ")
            alpha = float(input())
        except ValueError:
            print("Invalid input.")
        if 0.0 <= alpha <= 1.0:
            break
        print("Please enter a number between 0.0 ~ 1.0.")

    #ensure that both images are of same sizes, adjust second image to the same size as the original image
    img2 = cv2.resize(img2, (img.shape[0], img.shape[1]))

    #makes new copies of the image data where each pixel is 32-bit so there won't be overflow when multiplying the 2 images
    img1_f = img.astype(np.float32)
    img2_f = img2.astype(np.float32)

    #use the blend formula provided in the doc then round the values ensuring that all the values will be in the rang of 0 - 255
    blend = (1.0 - alpha) * img1_f + alpha * img2_f
    blend = np.clip(blend, 0, 255).astype(np.uint8)
    
    #log actions and images then display the pre-edit photo and new eddited photo
    update_state(blend, f"blend with {img2_path} with {alpha} factor")
    display_comparisons(history_img[-2], blend)

    return blend

def undo():
    #check if user edited the photo yet, if no history then convey to user there are no actions to undo
    #if there are previous operations, remove the last item in history_action and history_img and return next last edited image
    if len(history_img) <= 1:
        print("No actions to undo")
        return history_img[-1]
    else:
        print(f"Undo last operation {history_actions[-1]}")
        history_img.pop()
        history_actions.pop()
        return history_img[-1]

def display_action_history():
    print("History of operations: ")
    #display all the previous operations
    for actions in history_actions:
        print(f"* {actions}")

def save_exit ():
    #prompt user to enter path or name to save the final edited image to
    print("Enter filename to save final edited image to: ")
    file = input()
    #save the image
    cv2.imwrite(file, history_img[-1])
    print(f"Final edit saved to: {file}")
    #display the history of operations
    display_action_history()
    print("Bye! See you next time :)")
    sys.exit() #quit the application

def main():
    #prompt user to enter path to image and continue to do so until image can be loaded
    while True:
        print("Enter image's path/filename: ")
        original_img_path = input()
        original_img = cv2.imread(original_img_path)
        if original_img is None:
            print("Could not load image, try again.")
        else:
            print(f"Successfully loaded {original_img_path}")
            break
    
    #initialize arrays for image and history
    history_actions.clear()
    history_img.clear()
    history_img.append(original_img) #add original image to history_img array

    while True:
        #prompt users to select an operation
        print("""
           === Mini Photo Editor ===
           1. Adjust Brightness
           2. Adjust Contrast
           3. Convert to Grayscale
           4. Add Padding (choose border type)
           5. Apply Thresholding (binary or inverse)
           6. Blend with another image
           7. Undo last operation
           8. View History of Operations
           9. Save and Exit
        """)
        while True:
            try:
                print("Select and operation: ")
                choice = int(input())
            except ValueError:
                print("Invalid input, please select from 1 ~ 9.")
            if 1 <= choice <= 9:
                break
            print("Please select options 1 to 9.")
            
        current = history_img[-1] #retrieve the lastest image

        #match user input to call for the corresponding operations
        match choice:
            case 1:
                current = adjust_brightness(current)
            case 2:
                current = adjust_contrast(current)
            case 3:
                current = convert_to_grayscale(current)
            case 4:
                current = add_padding(current)
            case 5: 
                current = threshold(current)
            case 6:
                current = blend_image(current)
            case 7:
                current = undo()
            case 8:
                display_action_history()
            case 9:
                save_exit()
            case _:
                print("Invalid operation, choose again.")

if __name__ == "__main__":
    main()

    





