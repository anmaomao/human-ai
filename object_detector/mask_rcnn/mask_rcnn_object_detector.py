import tensorflow as tf

import object_detector.mask_rcnn.model as modellib
from object_detector.mask_rcnn import mask_rcnn_model_config
import keras.backend as K


class MaskRCNNObjectDetector:
    def __init__(self, model_config):
        """
        Initialize SSD model for object detection
        :param model_config: all configuration needed for initialize the model
        """
        model_path = model_config['model_path']
        default_config = mask_rcnn_model_config.COCOMaskRCNNModelConfig()
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            K.set_session(self.sess)
            self.model = modellib.MaskRCNN(mode="inference", model_dir=model_path, config=default_config)
            # Load weights trained on MS-COCO
            self.model.load_weights(model_path, by_name=True)
            self.model.keras_model._make_predict_function()
            # Without explicitly linking the session the weights for the dense layer added below don't get loaded
            # and so the model returns random results which vary with each model you upload because of random seeds.
            # Run detection
            self.detect_fn = lambda img: self.model.detect([img], verbose=0)


    def process_result(self, output, threshold):
        locations = output['rois'].tolist()
        classes_ids = output['class_ids'].tolist()
        scores = output['scores'].tolist()
        valid_indices = [idx for idx, item in enumerate(scores) if item > threshold]
        valid_locations = [item for idx, item in enumerate(locations) if idx in valid_indices]
        valid_scores = [item for idx, item in enumerate(scores) if idx in valid_indices]
        valid_classes = [item for idx, item in enumerate(classes_ids) if idx in valid_indices]

        return valid_locations, valid_scores, valid_classes


    def detect_objects(self, img_data, threshold=0.5):
        """
        Detect over 80 type of objects from given image
        :param img_data: np array of image data
        :return: a list of bounding boxes,
                 a list of probability scores,
                 a list of object class name
        """
        result = self.detect_fn(img_data)[0]
        processed_result = self.process_result(result, threshold)
        return processed_result


    def detect_human(self, img_data, threshold=0.5):
        """
        Detect human-ai from given image
        :param img_data: np array of image data
        :return: a list of bounding boxes,
                 a list of probability scores,
        """
        locations, scores, classes = self.detect_objects(img_data, threshold)
        valid_indices = [idx for idx, item in enumerate(classes) if item == 1]
        valid_locations = [item for idx, item in enumerate(locations) if idx in valid_indices]
        valid_scores = [item for idx, item in enumerate(scores) if idx in valid_indices]
        return valid_locations, valid_scores
