import cv2
from captionbot import CaptionBot

c = CaptionBot()
cap = cv2.VideoCapture(0)
while True:
    r, f = cap.read()
    cv2.imshow("Stream", f)
    key = cv2.waitKey(1)
    if key == ord('a'):
        print("Generating Caption...")
        cv2.imwrite('image.jpg',f)
        caption = c.file_caption('/home/aditya/Hack-a-bit2019/' + 'image.jpg')
        print(caption)
    elif key == 27:
        break
cv2.destroyAllWindows()
cap.release()
