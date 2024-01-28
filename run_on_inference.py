import tensorflow as tf
import cv2
import numpy as np

# Load the frozen graph
def load_frozen_graph(frozen_graph_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

# Load label map
def load_label_map(label_map_path):
    category_index = {}
    with tf.gfile.GFile(label_map_path, 'r') as fid:
        for line in fid:
            if line.startswith("name:"):
                name = line.strip().split('"')[1]
            elif line.startswith("id:"):
                category_id = int(line.split(":")[1].strip())
                category_index[category_id] = {'id': category_id, 'name': name}
    return category_index

# Draw bounding boxes on the image
def draw_boxes(image, boxes, classes, scores, category_index, threshold=0.5):
    height, width, _ = image.shape
    for i in range(len(boxes)):
        box = tuple(boxes[i].tolist())
        class_id = int(classes[i])
        score = scores[i]

        if score >= threshold:
            y_min, x_min, y_max, x_max = box
            y_min = int(y_min * height)
            x_min = int(x_min * width)
            y_max = int(y_max * height)
            x_max = int(x_max * width)

            # Draw bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Display label and score
            label = f"{category_index[class_id]['name']}: {score:.2f}"
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == "__main__":
    # Replace with the actual paths
    frozen_graph_path = 'path/to/save/frozen_graph/frozen_inference_graph.pb'
    label_map_path = 'path/to/label_map.pbtxt'
    image_path = 'path/to/your/inference_image.jpg'

    # Set the confidence threshold (adjust as needed)
    confidence_threshold = 0.5

    # Load the frozen graph
    detection_graph = load_frozen_graph(frozen_graph_path)

    # Load the label map
    category_index = load_label_map(label_map_path)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
           # Read the image
            image = cv2.imread(image_path)
            image_expanded = np.expand_dims(image, axis=0)
            
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Run inference
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: np.expand_dims(image, axis=0)}
            )

            # Draw bounding boxes on the image with threshold
            draw_boxes(image, boxes[0], classes[0], scores[0], category_index, threshold=confidence_threshold)

            # Display the result
            cv2.imshow('Object Detection Result', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
