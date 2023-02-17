import cv2
import numpy as np

# Load the COCO class labels our YOLO model was trained on
labelsPath = "yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# Derive the paths to the YOLO weights and model configuration
weightsPath = "yolo-coco/yolov4.weights"
configPath = "yolo-coco/yolov4.cfg"

# Load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
# net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# net = cv2.dnn.Dnn.readNet(weightsPath,configPath)
net = cv2.dnn.readNet(weightsPath, configPath)



# Get the output layer names of the YOLO model
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize the video stream
vs = cv2.VideoCapture(0)

# Loop over the frames of the video stream
while True:
    # Read a frame from the video stream
    ret, frame = vs.read()

    # Prepare the frame for object detection by resizing and normalizing
    # it to have a width of 608 pixels (the width the YOLO model was trained on)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (608, 608), swapRB=True, crop=False)

    # Perform forward pass through the YOLO network
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # Initialize lists to store the bounding boxes, confidences and class IDs
    boxes = []
    confidences = []
    classIDs = []

    # Loop over the output layers of the YOLO model
    for output in layerOutputs:
        # Loop over the detections
        for detection in output:
            # Extract the class ID and confidence of the current detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Only consider detections that are above a certain confidence level
            if confidence > 0.5:
                # Extract the bounding box coordinates of the current detection
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")

                # Use the center (x, y)-coordinates to derive the top and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                # Update the list of bounding boxes, confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Perform non-maximum suppression to eliminate duplicate and overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Ensure at least one detection exists
    if len(idxs) > 0:
        # Loop over the indexes we are keeping
        for i in idxs.flatten():
            # Extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Draw a bounding box rectangle and label the detected object
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Apply a brightness enhancement on the segmented region
            roi = frame[y:y+h, x:x+w]
            roi = cv2.addWeighted(roi, 1.2, np.zeros(roi.shape, roi.dtype), 0, 25)
            
    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()
vs.release()
