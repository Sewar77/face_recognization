import threading
import cv2 as cv
from deepface import DeepFace
from glob import glob

# Initialize video capture from the default camera (0) using DirectShow
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 648)  # Set the width of the video frame
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)  # Set the height of the video frame

counter = 0  # Frame counter
face_match = False  # Flag to indicate if a face match is found

# Load the reference image for face verification
reference_img = cv.imread('cat.4016.jpg')

# Function to check if the face in the frame matches the reference image
def check_face(frame):
    global face_match
    try:
        # Verify if the face in the frame matches the reference image
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False

# Main loop to process video frames
while True:
    ret, frame = cap.read()  # Read a frame from the video capture

    if ret:
        # Every 30 frames, start a new thread to check the face match
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1  # Increment the frame counter

        # Display the match result on the video frame
        if face_match:
            cv.putText(frame, "Match", (20, 450), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv.putText(frame, "NO Match", (20, 450), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv.imshow("Video", frame)  # Show the video frame

    # Break the loop if 'q' key is pressed
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# Release resources and close all OpenCV windows
cv.destroyAllWindows()
