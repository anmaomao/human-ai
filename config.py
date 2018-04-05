from os import environ

DEBUG = True

# Max size for accepted images in bytes
MAX_CONTENT_LENGTH = 2048 * 1024  # 2MB

SSD_OBJECT_MODEL_CONFIG = {
    # Model path
    'graph_path': './data/ssd/frozen_inference_graph.pb',
    'label_path': './data/ssd/mscoco_label_map.pbtxt',

    # Tensor name
    'input_tensor': 'image_tensor:0',
    'box_tensor': 'detection_boxes:0',
    'score_tensor': 'detection_scores:0',
    'class_tensor': 'detection_classes:0'
}

YOLO_OBJECT_MODEL_CONFIG = {

    # Model and required file path
    # 'model_path': './data/yolo/yolov2-coco.pb',
    'model_path': './data/yolo/yolo.h5',
    'meta_path': './data/yolo/yolov2-coco.meta',
    'classes_path': './data/yolo/coco_classes.txt',
    'anchors_path': './data/yolo/yolo_anchors.txt',

    # Threshold
    'score_threshold': 0.3,
    'iou_threshold': 0.5}

MASK_RCNN_OBJECT_MODEL_CONFIG = {

    # Model and required file path
    # 'model_path': './data/yolo/yolov2-coco.pb',
    'model_path': './data/mask_rcnn/mask_rcnn_coco.h5'}

MTCNN_DETECTOR_MODEL_CONFIG = {
    # Multi-task ConvNet face detector model path
    'model_folder': './data/mtcnn',

    # Prediction parameters
    'threshold': [0.6, 0.7, 0.7],
    'factor': 0.709,
    'minsize': 40

}

TINY_FACE_DETECTOR_MODEL_CONFIG = {
    'model_path': './data/tiny_face_model.pkl',
    'prob_thresh': 0.5,
    'nms_thresh': 0.1

}

MMOD_FACE_DETECTOR_MODEL_CONFIG = {
    'model_file': './data/mmod_human_face_detector.dat'
}

INCEPTION_GENDER_MODEL_CONFIG = {
    # Model path
    'model_path': './data/gender_inception_saved_model',

    # Tensor name
    'predict_tensor': 'prediction:0',
    'input_tensor': 'image:0',

    # Label encoding in the trained model
    'lbl_encoding': {'female': 1, 'male': 0}
}
RESNET_GENDER_MODEL_CONFIG = {
    # Model path
    'model_path': './data/gender_wide_resnet_saved_model',

    # Tensor name
    'predict_tensor': 'prediction_g',
    'input_tensor': 'image',

    # Label encoding in the trained model
    'lbl_encoding': {'female': 0, 'male': 1}
}

INCEPTION_AGE_MODEL_CONFIG = {
    # Model path
    'model_path': 'data/age_inception_saved_model',

    # Tensor name
    'predict_tensor': 'prediction:0',
    'input_tensor': 'image:0',

    # Label encoding in the trained model
    'lbl_encoding': {(0, 2): 0, (4, 6): 1, (8, 12): 2, (15, 20): 3, (25, 32): 4,
                     (38, 43): 5, (48, 55): 6, (60, 80): 7}
}
