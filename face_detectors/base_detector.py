from abc import ABCMeta, abstractmethod


class BaseDetector:
    __metaclass__ = ABCMeta

    @abstractmethod
    def detect_faces(self, img, threshold):
        """
        Get a list of face locations (may associate with face landmarks) from the image
        :param img: input image
        :return: a list of face location
        """
        pass

    @staticmethod
    def extract_face_crop(img, bbox, margin=0.2):
        """
        Crop faces and return the face images
        :param img: input image
        :param bbox: face bounding box
        :param margin:
        :return: a list of cropped face images
        """
        img_height = img.shape[0]
        img_width = img.shape[1]
        box_width = min(int(bbox[2] * img_width), img.shape[1]) - max(0, int(bbox[0] * img_width))
        box_height = min(int(bbox[3] * img_height), img.shape[0]) - max(0, int(bbox[1] * img_height))
        top = max(0, int(bbox[1] * img_height - margin * box_height))
        bottom = min(int(bbox[3] * img_height + margin * box_height), img.shape[0])
        left = max(0, int(bbox[0] * img_width - margin * box_width))
        right = min(int(bbox[2] * img_width + margin * box_width), img.shape[1])
        return img[top:bottom, left:right]

    @staticmethod
    def is_tiny_faces(face, min_height, min_width):
        """
        Given face data, check if the face is a tiny face based on given threshold
        :param face: bounding box of face
        :param min_height: minimum face height
        :param min_width: minimum face width
        :return:
        """
        height = len(face)
        width = len(face[0])
        if height >= min_height and width >= min_width:
            return False
        return True
