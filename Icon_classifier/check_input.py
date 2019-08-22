import cv2


def check_input(path):
    try:
        im = cv2.imread(path)
        im.shape
    except:
        print("Please check your input directory")
        return False

    # sizes = im.shape
    # if sizes[0] > 60 or sizes[1] > 60:
    #     print("Try smaller image")
    #     return False
    # elif sizes[0] < 12 or sizes[1] < 12:
    #     print("Try bigger image")
    #     return False
    # elif not sizes[2] == 3:
    #     print("Model is suited to rgb images")
    #     return False
    return True
