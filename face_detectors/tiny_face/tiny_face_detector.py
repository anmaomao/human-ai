# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pylab as pl
import tensorflow as tf
from scipy.special import expit

import face_detectors.tiny_face.tiny_face_model as tiny_face_model
from face_detectors.base_detector import BaseDetector

MAX_INPUT_DIM = 2000.0
IMG_SIZE = 500

class TinyFaceDetector(BaseDetector):
    def __init__(self, model_configs):
        model_path = model_configs['model_path']
        self.prob_thresh = model_configs['prob_thresh']
        self.nms_thresh = model_configs['nms_thresh']  # main
        # placeholder of input images. Currently batch size of one is supported.
        x = tf.placeholder(tf.float32, [1, None, None, 3])  # n, h, w, c
        bboxes = tf.placeholder(tf.float32, [None, 5])
        num_bboxes = tf.placeholder(tf.int32)
        self.nms = tf.image.non_max_suppression(
            bboxes[:, :4],
            bboxes[:, 4],
            max_output_size=num_bboxes, iou_threshold=self.nms_thresh)
        # Create the tiny face model which weights are loaded from a pretrained model.
        model = tiny_face_model.TinyFaceModel(model_path)
        self.score_final = model.tiny_face(x)
        self.average_image = model.get_data_by_key("average_image")
        self.clusters = model.get_data_by_key("clusters")
        self.clusters_h = self.clusters[:, 3] - self.clusters[:, 1] + 1
        self.clusters_w = self.clusters[:, 2] - self.clusters[:, 0] + 1
        self.normal_idx = np.where(self.clusters[:, 4] == 1)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.score_final_tf_fn = lambda img: self.sess.run(self.score_final, feed_dict={x: img})
            self.box_final_fn = lambda boxes, num_bbox: self.sess.run(self.nms,
                                                                      feed_dict={bboxes: boxes, num_bboxes: num_bbox})

    def box_to_relative_portion(self, bbox):
        newbox = []
        newbox.append(bbox[0] * 1.0 / IMG_SIZE)
        newbox.append(bbox[1] * 1.0 / IMG_SIZE)
        newbox.append(bbox[2] * 1.0 / IMG_SIZE)
        newbox.append(bbox[3] * 1.0 / IMG_SIZE)
        newbox.append(bbox[4])
        return newbox

    def detect_faces(self, origin_img_data, threshold=5):
        img_data = cv2.resize(origin_img_data, (IMG_SIZE, IMG_SIZE))

        def _calc_scales():
            raw_h, raw_w = img_data.shape[0], img_data.shape[1]
            min_scale = min(np.floor(np.log2(np.max(self.clusters_w[self.normal_idx] / raw_w))),
                            np.floor(np.log2(np.max(self.clusters_h[self.normal_idx] / raw_h))))
            max_scale = min(1.0, -np.log2(max(raw_h, raw_w) / MAX_INPUT_DIM))
            scales_down = pl.frange(min_scale, 0, 1.)
            scales_up = pl.frange(0.5, max_scale, 0.5)
            scales_pow = np.hstack((scales_down, scales_up))
            scales = np.power(2.0, scales_pow)
            return scales

        scales = _calc_scales()

        # initialize output
        bboxes = np.empty(shape=(0, 5))

        # process input at different scales
        for s in scales:
            img = cv2.resize(img_data, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            img = img - self.average_image
            img = img[np.newaxis, :]

            # we don't run every template on every scale ids of templates to ignore
            tids = list(range(4, 12)) + ([] if s <= 1.0 else list(range(18, 25)))
            ignoredTids = list(set(range(0, self.clusters.shape[0])) - set(tids))

            # run through the net
            score_final_tf = self.score_final_tf_fn(img)
            # collect scores
            score_cls_tf, score_reg_tf = score_final_tf[:, :, :, :25], score_final_tf[:, :, :, 25:125]
            prob_cls_tf = expit(score_cls_tf)
            prob_cls_tf[0, :, :, ignoredTids] = 0.0

            def _calc_bounding_boxes():
                # threshold for detection
                _, fy, fx, fc = np.where(prob_cls_tf > self.prob_thresh)

                # interpret heatmap into bounding boxes
                cy = fy * 8 - 1
                cx = fx * 8 - 1
                ch = self.clusters[fc, 3] - self.clusters[fc, 1] + 1
                cw = self.clusters[fc, 2] - self.clusters[fc, 0] + 1

                # extract bounding box refinement
                Nt = self.clusters.shape[0]
                tx = score_reg_tf[0, :, :, 0:Nt]
                ty = score_reg_tf[0, :, :, Nt:2 * Nt]
                tw = score_reg_tf[0, :, :, 2 * Nt:3 * Nt]
                th = score_reg_tf[0, :, :, 3 * Nt:4 * Nt]

                # refine bounding boxes
                dcx = cw * tx[fy, fx, fc]
                dcy = ch * ty[fy, fx, fc]
                rcx = cx + dcx
                rcy = cy + dcy
                rcw = cw * np.exp(tw[fy, fx, fc])
                rch = ch * np.exp(th[fy, fx, fc])

                scores = score_cls_tf[0, fy, fx, fc]
                tmp_bboxes = np.vstack((rcx - rcw / 2, rcy - rch / 2, rcx + rcw / 2, rcy + rch / 2))
                tmp_bboxes = np.vstack((tmp_bboxes / s, scores))
                tmp_bboxes = tmp_bboxes.transpose()
                return tmp_bboxes

            tmp_bboxes = _calc_bounding_boxes()
            bboxes = np.vstack((bboxes, tmp_bboxes))

        refind_idx = self.box_final_fn(bboxes, bboxes.shape[0])
        refind_idx = [idx for idx in refind_idx if bboxes[idx][4] > threshold]
        refined_bboxes = bboxes[refind_idx]
        refined_bboxes = [self.box_to_relative_portion(b) for b in refined_bboxes]
        return refined_bboxes
