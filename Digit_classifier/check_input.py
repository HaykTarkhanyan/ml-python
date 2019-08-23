import cv2

# Thats a helping function
# Checks if given directory is empty

def check_input(path):
    try:
        im = cv2.imread(path)
        im.shape
    except:
        print("Please check your input directory")
        return False
    return True
