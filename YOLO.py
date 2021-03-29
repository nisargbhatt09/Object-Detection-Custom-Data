import cv2
import numpy as np
import glob
import random


# Put your weights and cfg files here.
net = cv2.dnn.readNet("/home/nisarg/Downloads/YOLOV3_Tiny_New/yolov3_training_last.weights", "/home/nisarg/Downloads/YOLOV3_Tiny_New/yolov3_training.cfg")

# Name custom object
# Class Name
classes = ["Bucket"]


cap = cv2.VideoCapture(0)
ret, fr = cap.read()
while True:
    if ret:
        ret, fr = cap.read()
        # 0.4, 0.4
        img = cv2.resize(fr, None, fx=1, fy=1)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        outs = net.forward(output_layers)

        class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        print("OUTS: ", out)
        for detection in out:
            # print("detection: ", detection)
            scores = detection[5:]
            # print("Scores: ", scores)
            class_id = np.argmax(scores)
            # print("ArgMax: ", class_id)
            confidence = scores[class_id]
            # print("confidence: ", confidence)
            if confidence > 0.4:
                # Object detected
                # print("Class id: ", class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 20), font, 3, color, 2)

    img1 = cv2.resize(img, (720, 960))
    cv2.imshow("Image", img1)
    key = cv2.waitKey(20)
    if(key == 27):
        break

cv2.destroyAllWindows()