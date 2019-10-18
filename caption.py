import numpy as np
import time
import cv2
import os
from captionbot import CaptionBot

c = CaptionBot()
cap = cv2.VideoCapture(0)

labelsPath = "/home/aditya/Hack-a-bit2019/yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = "/home/aditya/Hack-a-bit2019/yolo-coco/yolov3.weights"
configPath = "/home/aditya/Hack-a-bit2019/yolo-coco/yolov3.cfg"

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)



while True:
    r, f = cap.read()
    cv2.imwrite('image.jpg',f)
    (H, W) = f.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(f, 1 / 255.0, (416, 416),
    	swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
    	for detection in output:
    		scores = detection[5:]
    		classID = np.argmax(scores)
    		confidence = scores[classID]
    		if confidence > 0.5:
    			box = detection[0:4] * np.array([W, H, W, H])
    			(centerX, centerY, width, height) = box.astype("int")
    			x = int(centerX - (width / 2))
    			y = int(centerY - (height / 2))
    			boxes.append([x, y, int(width), int(height)])
    			confidences.append(float(confidence))
    			classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
    	0.3)
    if len(idxs) > 0:
    	for i in idxs.flatten():
    		(x, y) = (boxes[i][0], boxes[i][1])
    		(w, h) = (boxes[i][2], boxes[i][3])
    		color = [int(c) for c in COLORS[classIDs[i]]]
    		cv2.rectangle(f, (x, y), (x + w, y + h), color, 2)
    		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
    		cv2.putText(f, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
    			0.5, color, 2)

    cv2.imshow("Stream", f)
    key = cv2.waitKey(1)
    if key == ord('a'):
        print("Generating Caption...")
        caption = c.file_caption('/home/aditya/Hack-a-bit2019/' + 'image.jpg')
        print(caption)
    elif key == 27:
        break


cv2.destroyAllWindows()
cap.release()
