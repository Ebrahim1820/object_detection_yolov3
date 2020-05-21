import cv2
import numpy as np
import glob
import imutils
import random


CLASSES_OBJ = "yolov3_model/obj.names"
WEIGHTS = "yolov3_model/yolov3_custom_last.weights"
CONFIG_FILE = "yolov3_model/yolov3_testing.cfg"
SCALE = 0.00392
IMAGES_PATH = glob.glob(r"images/*.*")
WIDTH, HIGHT = 416, 416
FONT = cv2.FONT_ITALIC
CONF_THRESHOLD, NMS_THRESHOLD = 0.5, 0.4

# Load Yolo


def load_yolo_model(weight, conf):
    # read pre-trained model and config file
    net = cv2.dnn.readNet(weight, conf)
    # Name custom objects
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # return our neural network model and last output layers
    return net, output_layers

# create input blob


def create_blob(img, net, output_layers):
    blob = cv2.dnn.blobFromImage(img, SCALE, (WIDTH, HIGHT),
                                 (0, 0, 0), True, crop=False)
    # set input blob for the network
    net.setInput(blob)
    # run inference through the network and gather predictions from output layers
    outs = net.forward(output_layers)
    # return predictions
    return outs


def load_img(IMAGES_PATH):
    random.shuffle(IMAGES_PATH)
    # Loading image
    for img in IMAGES_PATH:
        img = cv2.imread(img)
        img = imutils.resize(img, width=max(500, img.shape[1]))
        height, width, _ = img.shape

        return img, height, width

# loop over predictions and extract bounding boxes


def detect_obj(outs, height, width):
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # for each detecion from each output layer get the confidence, class id,
            # bounding box params and ignore weak detections (confidence < 0.5)
            if confidence > 0.3:
                # object detected
                #print("[INFO] Object has been detected..!")
                # extract center coordinates and Bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # add coordinates to the list
                boxes.append([x, y, w, h])
                # add object classes and confidence
                class_ids.append(class_id)
                confidences.append(float(confidence))

     # returns all the coordinates and classes of object and confidences
    return boxes, class_ids, confidences


def nms(boxes, confidences, conf_threshold, nms_threshold):
    # apply non-max suppression to remove overlap bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    return indices

def draw_bounding_box(img, font, boxes, classes, class_ids, confidences, colors):

    indices = nms(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    # loop over bounding boxes and plot image
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x - 10, y - 10), font, 0.5, (255, 0, 0), 2)
     # return result
    return img


def main(IMAGES_PATH, CLASSES_OBJ):
    classes = []
    with open(CLASSES_OBJ, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    net, output_layers = load_yolo_model(WEIGHTS, CONFIG_FILE)

    img, height, width = load_img(IMAGES_PATH)

    outs = create_blob(img, net, output_layers)
    boxes, class_ids, confidences = detect_obj(outs, height, width)

    img = draw_bounding_box(img, FONT, boxes, classes, class_ids, confidences, colors)

    return img


#-------------------------------#
if __name__ == "__main__":

    # while check:
    print('[INFO] Loading...!')
    i = 0
    while True:
        # detect object
        img = main(IMAGES_PATH, CLASSES_OBJ)
        cv2.imshow("Image", img)
        path = 'Results/image{:>05}.jpg'.format(i)

        print('[INFO] To save Image press S or press Esc..!')
        k = cv2.waitKey(0)
        if k == 27:  # wait for ESC key to exit
            print('[INFO] Exit Program..! ')
            cv2.destroyAllWindows()
            break
        # Save image
        elif k == ord('s'):  # wait for 's' key to save and exit
            cv2.imwrite(path, img)
            print('[INFO] Successfully saved image..! ')
            i += 1
            cv2.destroyAllWindows()
