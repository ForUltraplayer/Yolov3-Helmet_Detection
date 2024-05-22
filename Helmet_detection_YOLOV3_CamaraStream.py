#pip install numpy opencv-python opencv-python-headless
#python 설치 필수

import cv2 as cv
import numpy as np
import os

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

# Load names of classes
helmetClassesFile = 'C:/Users/Mnyang/Desktop/yolov3-Helmet-Detection-master/obj.names' #본인의 경로로 변경해야함
personClassesFile = 'C:/Users/Mnyang/Desktop/yolov3-Helmet-Detection-master/coco.names' #본인의 경로로 변경해야함

with open(helmetClassesFile, 'rt') as f:
    helmetClasses = f.read().rstrip('\n').split('\n')

with open(personClassesFile, 'rt') as f:
    personClasses = f.read().rstrip('\n').split('\n')

# Load the helmet detection model
helmetModelConfiguration = 'C:/Users/Mnyang/Desktop/yolov3-Helmet-Detection-master/yolov3-obj.cfg' #본인의 경로로 변경해야함
helmetModelWeights = 'C:/Users/Mnyang/Desktop/yolov3-Helmet-Detection-master/yolov3-obj_2400.weights'#본인의 경로로 변경해야함
helmetNet = cv.dnn.readNetFromDarknet(helmetModelConfiguration, helmetModelWeights)

# Load the person detection model
personModelConfiguration = 'C:/Users/Mnyang/Desktop/yolov3-Helmet-Detection-master/yolov3.cfg' #본인의 경로로 변경해야함
personModelWeights = 'C:/Users/Mnyang/Desktop/yolov3-Helmet-Detection-master/yolov3.weights' #본인의 경로로 변경해야함
personNet = cv.dnn.readNetFromDarknet(personModelConfiguration, personModelWeights)

# Set backend and target to CUDA if available
if cv.cuda.getCudaEnabledDeviceCount() > 0:
    print("CUDA enabled, using GPU.")
    helmetNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    helmetNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    personNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    personNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
else:
    print("CUDA not available, using CPU.")
    helmetNet.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    helmetNet.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    personNet.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    personNet.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(frame, classId, conf, left, top, right, bottom, classes):
    print(f"Drawing box: {classId}, {conf}, {left}, {top}, {right}, {bottom}")
    
    # Determine the color based on the class name
    if classes[classId].strip().lower() == 'helmet':
        color = (255, 178, 50)  # Orange for helmet
    elif classes[classId].strip().lower() == 'person':
        color = (255, 0, 0)  # Blue for person
   
    # Draw the rectangle with the determined color
    cv.rectangle(frame, (left, top), (right, bottom), color, 3)
    
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
def postprocess(frame, outs, classes):
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
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            detections.append((classIds[i], confidences[i], (left, top, left + width, top + height)))
            drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height, classes)
    else:
        print("No indices found")
    return detections

# Process inputs
winName = 'Helmet and Person Detection'
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
        helmetNet.setInput(blob)
        personNet.setInput(blob)
        
        # Runs the forward pass to get output of the output layers
        helmetOuts = helmetNet.forward(getOutputsNames(helmetNet))
        personOuts = personNet.forward(getOutputsNames(personNet))
        
        # Process the helmet detections
        helmetDetections = postprocess(frame, helmetOuts, helmetClasses)
        
        # Process the person detections
        personDetections = postprocess(frame, personOuts, personClasses)
        
        # Draw detections and check for persons without helmets
        helmetDetected = False
        for (classId, conf, box) in helmetDetections:
            helmetDetected = True
        
        if not helmetDetected:
            for (classId, conf, box) in personDetections:
                if personClasses[classId] == 'person':
                    drawPred(frame, classId, conf, *box, personClasses)
                    print("Person detected without helmet")

        # Put efficiency information
        t, _ = helmetNet.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # Display the frame
        cv.imshow(winName, frame)

    cap.release()

cv.destroyAllWindows()
