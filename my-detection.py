from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput
import jetson.inference
import jetson.utils
import sys
net = detectNet("ssd-mobilenet-v2", threshold=0.5)
img_path="/home/nvidia/jetson-inference/examples/test.jpg"
img=jetson.utils.loadImage(img_path)

if img is None: # capture timeout
    print("cannot find the image")
    sys.exit(1)
detections = net.Detect(img)
print(detections)
jetson.utils.saveImage("output.jpg",img)
