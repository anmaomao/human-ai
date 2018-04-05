# coding:utf-8

from PIL import Image
import requests as re
from io import BytesIO
from timeit import default_timer as timer
import numpy as np
import flask
from flask import request
from webargs import fields, ValidationError
from webargs.flaskparser import parser

DEFAULT_MIN_FACE_HEIGHT = 20
DEFAULT_MIN_FACE_WIDTH = 10

app = flask.current_app
analyze_blueprint = flask.Blueprint("analyze", __name__)


def _analysis_response(params):
    """
    Get object(human-ai), face locations and gender + age information
    :param params: model parameters
    :return: analysis response
    """

    min_face_height = params['min_face_height']
    min_face_width = params['min_face_width']

    profiling = {}
    img_url = params['img_url']
    t1 = timer()
    img_response = re.get(img_url)
    image_data = np.asarray(Image.open(BytesIO(img_response.content)))
    profiling['load_img'] = timer() - t1

    response = {}
    humans = []
    t1 = timer()

    boxes, scores = app.human_detector.detect_human(image_data)

    profiling['detect_human'] = timer() - t1

    if len(boxes) == 0:
        response = {
            'num_human': 0,
            'humans': [],
            'num_face': 0,
            'faces': []}
        app.report_profiling_to_statsd(profiling, 'analyze')
        return response

    for idx in range(0, len(boxes)):
        # box = {
        #         'top': int(boxes[idx][0]),
        #         'left': int(boxes[idx][1]),
        #         'bottom': int(boxes[idx][2]),
        #         'right': int(boxes[idx][3]),
        #     }
        human = {
            # 'location': box,
            'prob': float(scores[idx])
        }
        humans.append(human)

    response['num_human'] = len(humans)
    response['humans'] = humans

    t1 = timer()
    detected_faces = app.face_detector.detect_faces(image_data)

    profiling['detect_face'] = timer() - t1

    if detected_faces is None or len(detected_faces) == 0:
        response['faces'] = []
        response['num_face'] = 0
        app.report_profiling_to_statsd(profiling, 'analyze')
        return response
    faces = []

    t_gender = 0
    t_age = 0

    for bbox in detected_faces:
        # location = {
        #     'left': int(bbox[0]),
        #     'top': int(bbox[1]),
        #     'right': int(bbox[2]),
        #     'bottom': int(bbox[3])
        # }
        face = app.face_detector.extract_face_crop(image_data, bbox)
        if not app.face_detector.is_tiny_faces(face, min_face_height, min_face_width):

            gender_probs = app.gender_analyzer.analyze_gender(face)
            t_gender += timer() - t1
            if gender_probs['female'] > gender_probs['male']:
                gender = 'female'
                gender_proba = gender_probs['female']
            else:
                gender = 'male'
                gender_proba = gender_probs['male']

            gender = {
                'gender': gender,
                'prob': float(gender_proba),
            }

            t1 = timer()
            age_estimation, age_prob = app.age_analyzer.analyze_age(face)
            t_age += timer() - t1
            age_prob = round(age_prob, 2)
            age = {
                'age': int(age_estimation),
                'prob': float(age_prob),
            }
            faces.append(
                {
                    'gender': gender,
                    # 'location': location,
                    'age': age,
                    'confidence': float(bbox[4])
                }
            )

    num_face = len(detected_faces)
    num_analyzed_face = len(faces)
    response['faces'] = faces
    response['num_face'] = num_face
    if num_face:
        profiling['avg_analyze_gender'] = t_gender / num_analyzed_face
        profiling['avg_analyze_age'] = t_age / num_analyzed_face
    profiling['analyze_gender'] = t_gender
    profiling['analyze_age'] = t_age

    app.report_profiling_to_statsd(profiling, 'analyze')

    return response


def _validate_min_face_size(size):
    if size <= 0:
        raise ValidationError('Minimum height and/or width should always no less than 0')


@analyze_blueprint.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze client
    :return: return the analyzing result in json format
    """
    params = parser.parse({
        'img_url': fields.String(required=True),
        'min_face_height': fields.Int(validate=_validate_min_face_size, missing=DEFAULT_MIN_FACE_HEIGHT),
        'min_face_width': fields.Int(validate=_validate_min_face_size, missing=DEFAULT_MIN_FACE_WIDTH),
    }, request)

    response = _analysis_response(params)
    return flask.jsonify(response)
