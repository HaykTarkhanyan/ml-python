import cv2

# checks if direcorty is valid
# is being called from predict.py file

def check_input(path):
    try:
        im = cv2.imread(path)
        im.shape
    except:
        print("Please check your input directory")
        return False
    return True
