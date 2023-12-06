import cv2
from threading import Thread
from queue import Queue

cap = cv2.VideoCapture("./data/asd.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

buffer = Queue(maxsize=100)  # Buffer to store frames, adjust the size as needed

def capimg():
    while True:
        ret, frame = cap.read()
        if frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        buffer.put(frame)  # Put the frame into the buffer

        key = cv2.waitKey(int(300 / fps)) & 0xFF

        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def playback():
    while True:
        frame = buffer.get()  # Get a frame from the buffer
        if frame is not None:
            cv2.imshow('img', frame)

        # Calculate delay based on the original frame rate
        delay = int(300 / fps)

        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    t1 = Thread(target=capimg)
    t1.start()

    t2 = Thread(target=playback)
    t2.start()
