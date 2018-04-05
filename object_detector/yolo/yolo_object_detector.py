"""Run a YOLO_v2 style detection model on test images."""
import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model

from object_detector.yolo.keras_yolo import yolo_eval, yolo_head

IMG_SIZE = 608


class YOLOObjectDetector:
    def __init__(self, model_configs):
        model_path = model_configs['model_path']
        anchors_path = model_configs['anchors_path']
        classes_path = model_configs['classes_path']
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]

        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            K.set_session(self.sess)
            # Missing this was the source of one of the most challenging an insidious bugs that I've ever encountered.
            # Without explicitly linking the session the weights for the dense layer added below don't get loaded
            # and so the model returns random results which vary with each model you upload because of random seeds.
            self.yolo_model = load_model(model_path)

            self.yolo_outputs = yolo_head(self.yolo_model.output, anchors, len(class_names))
            input_image_shape = K.placeholder(shape=(2,))
            boxes, scores, classes = yolo_eval(
                self.yolo_outputs,
                input_image_shape,
                score_threshold=model_configs['score_threshold'],
                iou_threshold=model_configs['iou_threshold'])

            self.prediction_fn = lambda img_data, shape: self.sess.run(
                [boxes, scores, classes],
                feed_dict={
                    self.yolo_model.input: img_data,
                    input_image_shape: shape
                })

    def image_preprocessing(self, image_data):
        """

        :param image_data:
        :return:
        """
        resized_image = cv2.resize(image_data, (IMG_SIZE, IMG_SIZE))
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
        resized_image = np.array(resized_image, dtype='float32')
        resized_image /= 255.
        resized_image = np.expand_dims(resized_image, 0)
        resized_image = np.flip(resized_image, axis=3)
        return resized_image

    def detect_objects(self, image_data, threshold=0.3):
        """

        :param image_data:
        :return:
        """
        image_shape = image_data.shape[0:2]
        resized_image = self.image_preprocessing(image_data)
        boxes, scores, classes = self.prediction_fn(resized_image, image_shape)
        valid_indices = [idx for idx, item in enumerate(scores) if item > threshold]
        return boxes[valid_indices], scores[valid_indices], classes[valid_indices]

    def detect_human(self, image_data, threshold=0.3):
        """

        :param image_data:
        :return:
        """
        boxes, scores, classes = self.detect_objects(image_data, threshold)
        valid_indices = [idx for idx, value in enumerate(classes) if value == 0]
        return boxes[valid_indices], scores[valid_indices]
