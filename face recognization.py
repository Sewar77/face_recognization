import threading
import cv2 as cv 
from deepface import DeepFace
from glob import glob

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 648)

cap.set(cv.CAP_PROP_FRAME_HEIGHT , 480)

counter = 0
 
face_match = False

#images = glob("C:/Users/User/Desktop/graduation_project/deepfake/0.png")


reference_img = cv.imread('C:/Users/User/Desktop/archive/test_set/test_set/cats/cat.4016.jpg')

def chech_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
           face_match = True
        else:
           face_match = False
    except ValueError:
       face_match = False



while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=chech_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        if face_match:
            cv.putText(frame, "Match", (20,450), cv.FONT_HERSHEY_SIMPLEX, 2 , (0,255,0), 3)
        else:
            cv.putText(frame, "NO Match", (20,450), cv.FONT_HERSHEY_SIMPLEX, 2 , (0,0,255), 3)

        cv.imshow("Vedio", frame)



    key = cv.waitKey(1)
    if key == ord('q'):
       break

cv.destroyAllWindows()





