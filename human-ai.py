#!/usr/bin/env python

import flask
from datadog import statsd
from flask import jsonify
from flask import request

from web import analyze_blueprint
from object_detector import MaskRCNNObjectDetector
from face_detectors import TinyFaceDetector
from face_analyzers import InceptionGenderAnalyzer, InceptionAgeAnalyzer


class HumanAI(flask.Flask):
    """Flask app handling request on analyzing human related information"""

    def __init__(self, *args, **kwargs):
        super(HumanAI, self).__init__(*args, **kwargs)

        config = self.config

        config.from_object('config')
        config.from_envvar("HUMAN_AI_CONFIG_PATH", silent=True)
        # self.mtcnn = MTCNNDetector(model_config=config['MTCNN_DETECTOR_MODEL_CONFIG'])
        # self.mmod = MMODDetector(model_config=config['MMOD_DETECTOR_MODEL_CONFIG'], up=False)
        # self.mmod_up = MMODDetector(model_config=config['MMOD_DETECTOR_MODEL_CONFIG'], up=True)
        self.face_detector = TinyFaceDetector(model_configs=config['TINY_FACE_DETECTOR_MODEL_CONFIG'])

        # self.gender_analyzer = ResnetGenderAnalyzer(config['RESNET_GENDER_MODEL_CONFIG'])

        self.gender_analyzer = InceptionGenderAnalyzer(config['INCEPTION_GENDER_MODEL_CONFIG'])
        self.age_analyzer = InceptionAgeAnalyzer(config['INCEPTION_AGE_MODEL_CONFIG'])

        self.human_detector = MaskRCNNObjectDetector(config['MASK_RCNN_OBJECT_MODEL_CONFIG'])
        # self.human_detector = YOLOObjectDetector(config['YOLO_OBJECT_MODEL_CONFIG'])
        # self.human_detector = SSDObjectDetector(config['SSD_OBJECT_MODEL_CONFIG'])
        statsd.event(title='Vision-AI App Started', text='',
                     alert_type='info', tags=['visionai.started'])

    def report_profiling_to_statsd(self, profiling, tag):
        """
        Report status to datadog
        :param profiling: profile values
        :param tag: tags for the profile
        :return:
        """
        metric = 'human-ai.request.duration'
        path = flask.request.path[1:].replace('/', '_')
        tags = ['type:' + (path or 'index'), 'tag:' + tag]

        # datadog prints warnings when there is no agent running,
        # suppress them here with this ugly check
        debug_or_testing = self.config.get('TESTING', False) or self.config.get('DEBUG', False)

        if debug_or_testing:
            return

        try:
            for k, v in profiling.items():
                statsd.histogram(metric, v, tags + [tag + k])
        except:
            pass


app = HumanAI('humanai')

app.register_blueprint(analyze_blueprint)


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(422)
@app.errorhandler(Exception)
def handle_all_exceptions(error):
    message = getattr(error, 'message', 'Invalid request')
    status_code = getattr(error, 'code', 500)

    try:
        # Validation handling for webargs library
        message = error.exc.messages
        status_code = 400  # override 422 from webargs
    except (AttributeError, KeyError):
        pass

    response = jsonify(error=message)

    # pass error to handle_after_request for logging
    response.error = error
    response.status_code = status_code
    return response


# called for all requests after the errorhandler function
@app.after_request
def handle_after_request(response):
    path = request.path[1:].replace('/', '_')
    code = response.status_code
    tags = [
        'type:' + (path or 'index'),
        'status:' + str(code)
    ]

    try:
        tags.append('error:' + type(response.error).__name__)
    except AttributeError:
        pass

    try:
        statsd.increment('humanai.requests.completed.count', tags=tags)
    except:
        pass
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
