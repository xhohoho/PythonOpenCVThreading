import cv2
from threading import Thread
from queue import Queue
import time
from flask import Flask, render_template, Response

import argparse
import numpy as np

from yunet import YuNet

# Check OpenCV version
assert cv2.__version__ >= "4.8.0", \
       "Please install latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA_FP16],
    [cv2.dnn.DNN_BACKEND_TIMVX,  cv2.dnn.DNN_TARGET_NPU],
    [cv2.dnn.DNN_BACKEND_CANN,   cv2.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(description='YuNet: A Fast and Accurate CNN-based Face Detector (https://github.com/ShiqiYu/libfacedetection).')
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set input to a certain image, omit if using camera.')
parser.add_argument('--model', '-m', type=str, default='face_detection_yunet_2023mar.onnx',
                    help="Usage: Set model type, defaults to 'face_detection_yunet_2023mar.onnx'.")
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--conf_threshold', type=float, default=0.9,
                    help='Usage: Set the minimum needed confidence for the model to identify a face, defauts to 0.9. Smaller values may result in faster detection, but will limit accuracy. Filter out faces of confidence < conf_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3,
                    help='Usage: Suppress bounding boxes of iou >= nms_threshold. Default = 0.3.')
parser.add_argument('--top_k', type=int, default=5000,
                    help='Usage: Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', action='store_true',
                    help='Usage: Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')
args = parser.parse_args()

cap = cv2.VideoCapture("./data/asd.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

buffer = Queue(maxsize=100)  # Buffer to store frames, adjust the size as needed
app = Flask(__name__)

def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    output = image.copy()
    landmark_color = [
        (255,   0,   0), # right eye
        (  0,   0, 255), # left eye
        (  0, 255,   0), # nose tip
        (255,   0, 255), # right mouth corner
        (  0, 255, 255)  # left mouth corner
    ]

    if fps is not None:
        cv2.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    for det in results:
        bbox = det[0:4].astype(np.int32)
        cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

        conf = det[-1]
        cv2.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv2.circle(output, landmark, 2, landmark_color[idx], 2)

    return output

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

def detection():
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    # Instantiate YuNet
    model = YuNet(modelPath=args.model,
                  inputSize=[320, 320],
                  confThreshold=args.conf_threshold,
                  nmsThreshold=args.nms_threshold,
                  topK=args.top_k,
                  backendId=backend_id,
                  targetId=target_id)
    
    model.setInputSize([w, h])
    tm = cv2.TickMeter()
    while cv2.waitKey(1) < 0:
        frame = buffer.get()  # Get a frame from the buffer
        if frame is not None:
            tm.start()
            results = model.infer(frame) # results is a tuple
            tm.stop()

            # Draw results on the input image
            frame = visualize(frame, results, fps=tm.getFPS())
            cv2.imshow('img', frame)
            ret, encoded_frame = cv2.imencode('.jpg', frame)
            frame = encoded_frame.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            tm.reset()

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def run_flask():
    app.run(port=81)

if __name__ == '__main__':
    t1 = Thread(target=capimg)
    t1.start()

    t2 = Thread(target=run_flask)
    t2.start()
