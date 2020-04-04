import cv2
import numpy as np
from Predictor import Predictor

def predict(image, predictor,LABELS):
    """
    Makes a prediction on image by calling the predict method from the Predictor
    class.
    
    Parameters
    ----------
    image: image
    
    predictor: objects of the class Predictor
    
    LABELS: list of possibly labels, where each label is a string
    
    Returns
    -------
    A list of probabilities for labels
    """
    logits = predictor.predict(image)
    print(logits)
    print('the predicted label is: {}'.format(LABELS[np.argmax(logits)]))

    return LABELS[np.argmax(logits)]

def live_stream(predictor,LABELS,CASCADE_PATH):
    """
    Opens up the the desktop camera, and makes predictions in real time.
    Draws the rectangle around the face and prints the prediction with highest 
    probability on the top of the rectangle

    Parameters
    ----------
    predictor: predictor object from Predictor class
    
    LABELS: list of possibly labels, where each label is a string
    
    CASCADE_PATH: path to the haar_cascade_classifier

    Returns
    -------
    None
    """

    faceCascade = cv2.CascadeClassifier(CASCADE_PATH)
    video_capture = cv2.VideoCapture(0)
    w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter('demo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, (w, h))
    saved = False

    # video_capture.set(cv2.CV_CAP_PROP_FPS, 10)
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minSize=(120, 120),
        )

        # Draw a rectangle around the faces
        if len(faces) != 0:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            faceimg = gray[y:y + w, x:x + h]
            lastimg = cv2.resize(faceimg, (predictor.resolution, predictor.resolution))
            predicted = predict(lastimg, predictor,LABELS)
            cv2.putText(frame, predicted, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=5)

        writer.write(frame)
        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

