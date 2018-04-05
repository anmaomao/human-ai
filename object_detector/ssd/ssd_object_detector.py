import numpy as np
import tensorflow as tf

from .label_map_util import load_labelmap, convert_label_map_to_categories, create_category_index

NUM_CLASSES = 90


class SSDObjectDetector:
    def __init__(self, model_config):
        """
        Initialize SSD model for object detection
        :param model_config: all configuration needed for initialize the model
        """
        path_to_ckpt = model_config['graph_path']
        path_to_labels = model_config['label_path']

        detection_boxes_tensor = model_config['box_tensor']
        input_tensor_name = model_config['input_tensor']
        detection_score_tensor = model_config['score_tensor']
        detection_class_tensor = model_config['class_tensor']
        with tf.Graph().as_default() as graph:
            self.session = tf.Session()
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                image_tensor = graph.get_tensor_by_name(input_tensor_name)
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = graph.get_tensor_by_name(detection_boxes_tensor)
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = graph.get_tensor_by_name(detection_score_tensor)
                detection_classes = graph.get_tensor_by_name(detection_class_tensor)
                self.prediction_fn = lambda img: self.session.run(
                    [detection_boxes, detection_scores, detection_classes],
                    feed_dict={image_tensor: img})

        label_map = load_labelmap(path_to_labels)
        categories = convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                     use_display_name=True)
        self.category_index = create_category_index(categories)

    def _relative_to_absolution_location(self, relative_location, img_width, img_height):
        """
        change the portion relative location to absolute location
        :param relative_location: portion relative location of the bounding box
        :param img_width: width of image
        :param img_height: height of image
        :return: absolute location of the bounding box
        """
        top = int(relative_location[0] * img_height)
        left = int(relative_location[1] * img_width)
        bottom = int(relative_location[2] * img_height)
        right = int(relative_location[3] * img_width)
        return [top, left, bottom, right]

    def detect_objects(self, img_data, threshold=0.3):
        """
        Detect over 80 type of objects from given image
        :param img_data: np array of image data
        :param threshold: probability threshold for choosing the objects
        :return: a list of bounding boxes,
                 a list of probability scores,
                 a list of object class name
        """
        width = img_data.shape[1]
        height = img_data.shape[0]
        resized_img = np.expand_dims(img_data, axis=0)
        boxes, scores, classes = self.prediction_fn(resized_img)
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        valid_indices = [idx for idx, item in enumerate(scores) if item > threshold]
        valid_object_classes = [item for idx, item in enumerate(classes) if idx in valid_indices]
        valid_boxes = [self._relative_to_absolution_location(box, width, height) for idx, box in enumerate(boxes) if
                       idx in valid_indices]
        return valid_boxes, scores[valid_indices], valid_object_classes

    def detect_human(self, img_data, threshold=0.3):
        """
        Detect human-ai from given image
        :param img_data: np array of image data
        :param threshold: probability threshold for choosing the objects
        :return: a list of bounding boxes,
                 a list of probability scores,
        """
        boxes, scores, classes = self.detect_objects(img_data, threshold=threshold)
        valid_indices = [idx for idx, item in enumerate(classes) if item == 1]
        valid_boxes = [box for idx, box in enumerate(boxes) if idx in valid_indices]
        valid_scores = [score for idx, score in enumerate(scores) if idx in valid_indices]

        return valid_boxes, valid_scores
