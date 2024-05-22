import cv2 as cv
import numpy as np
import os

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

# Load names of classes
classesFile = 'C:/Users/Mnyang/Desktop/yolov3-Helmet-Detection-master/obj.names'
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = 'C:/Users/Mnyang/Desktop/yolov3-Helmet-Detection-master/yolov3-obj.cfg'
modelWeights = 'C:/Users/Mnyang/Desktop/yolov3-Helmet-Detection-master/yolov3-obj_2400.weights'

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    output_layers = net.getUnconnectedOutLayers().flatten().astype(int)  # Flatten to 1D array
    output_names = [layersNames[i - 1] for i in output_layers]
    return output_names

# Draw the predicted bounding box
def drawPred(frame, classId, conf, left, top, right, bottom):
    print("Drawing box: {classId}, {conf}, {left}, {top}, {right, {bottom}")
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    label_name, label_conf = label.split(':')
    print(f"Label: {label_name}, Confidence: {label_conf}")  # Debugging line
    if label_name.strip().lower() == 'helmet':  # Check if it is a helmet
        cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
        return 1
    return 0

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    count_person = 0
    if len(indices) > 0:
        for i in indices:
            if isinstance(i, (list, tuple, np.ndarray)):
                i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            count_person += drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        print(f"count_person: {count_person}")
    else:
        print("No indices found")

# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

# Try different camera indices
for cam_idx in range(10):
    cap = cv.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Camera index {cam_idx} not opened, trying next...")
        cap.release()
        continue
    
    print(f"Camera index {cam_idx} opened successfully")
    
    while cv.waitKey(1) < 0:
        # Get frame from the video capture
        hasFrame, frame = cap.read()
        if not hasFrame:
            print("No frame captured from camera, exiting...")
            break

        # Create a 4D blob from a frame
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
        
        # Sets the input to the network
        net.setInput(blob)
        
        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))
        
        # Remove the bounding boxes with low confidence
        postprocess(frame, outs)
        
        # Put efficiency information
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # Display the frame
        cv.imshow(winName, frame)

    cap.release()

cv.destroyAllWindows()
