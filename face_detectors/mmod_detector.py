# from .base_detector import BaseDetector
# import dlib
#
#
# class MMODDetector(BaseDetector):
#     """
#     Use Max Margin Object Detection Algorithm for face detection, implemented inside dlib library
#     """
#
#     def __init__(self,
#                  model_config,
#                  up):
#         """
#         Initialize the MMOD detector
#         :param model_config: contains path to where the model is stored
#         :param up: whether adjust the sampling window upper or not, if no, minimum window size is 80 * 80, turning up
#         on will take around 6* time to detect but have a much better recall rate
#         """
#         if up:
#             self.upsampling = 1
#         else:
#             self.upsampling = 0
#
#         model_file = model_config['model_file']
#         self.detector = dlib.cnn_face_detection_model_v1(model_file)
#
#     def detect_faces(self, img, threshold=1):
#         """
#         Detect faces from images and return the bounding boxes
#         :param img: input images
#         :return: bounding boxes
#         """
#         result = self.detector(img, self.upsampling)
#         return [[r.rect.left(), r.rect.top(), r.rect.right(), r.rect.bottom(), r.confidence] for r in result]
