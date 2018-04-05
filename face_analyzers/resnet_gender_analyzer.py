import cv2
import tensorflow as tf

TOWER_NAME = 'tower'
IMG_SIZE = 64
CROP_IMG_SIZE = 227


class ResnetGenderAnalyzer:
    def __init__(self, model_config):
        """
        Initialize wide resnet model for gender analysis on face
        :param model_config:
               model_config (dict) - 'model_path' is the path to the trained model
                                  'lbl_encoding' is the encoding of the labels
                                                 in the trained model.
        """
        export_path = model_config['model_path']
        predict_tensor_name = 'prediction_g/Softmax:0'
        input_tensor_name = 'image:0'
        self._lbl_encoding = model_config['lbl_encoding']
        with tf.Graph().as_default() as graph:
            sess = tf.Session()
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_path)
            predict_tensor = graph.get_tensor_by_name(predict_tensor_name)

            self.prediction_fn = lambda resized_faces: sess.run([predict_tensor, 'prediction_a/Softmax:0'],
                                                               feed_dict={input_tensor_name: resized_faces})

    def analyze_gender(self, face_data_list):
        """
        Estimate the gender by analyzing face data
        :param face_data_list: np array of a list of face data
        :return: a tuple of gender estimation, probability
        """
        resized_faces = [cv2.resize(face_data, (IMG_SIZE, IMG_SIZE)) for face_data in face_data_list]
        pred_results = self.prediction_fn(resized_faces)[0]
        decoded_gender_pred = [dict([(name, output[idx]) for name, idx in
                                     self._lbl_encoding.items()]) for output in pred_results]
        return decoded_gender_pred
