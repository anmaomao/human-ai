import re

import tensorflow as tf
from tensorflow.contrib.layers import *
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
from tensorflow.python.saved_model import signature_constants

TOWER_NAME = 'tower'
IMG_SIZE = 256
CROP_IMG_SIZE = 227

checkpoint_path = "data/age_net/"
export_path = './data/age_inception_saved_model'
nlabels = 8


def make_multi_crops(face_data):
    """
    Crop and flip face image corners and center and make it a batch
    :param face_data: face data
    :return: a batch of cropped face data
    """

    crops = []
    shape = face_data.get_shape().as_list()
    h = shape[0]
    w = shape[1]
    hl = h - CROP_IMG_SIZE
    wl = w - CROP_IMG_SIZE

    crop = tf.image.resize_images(face_data, (CROP_IMG_SIZE, CROP_IMG_SIZE))
    crops.append(tf.image.per_image_standardization(crop))
    crops.append(tf.image.flip_left_right(crop))

    corners = [(0, 0), (0, wl), (hl, 0), (hl, wl), (int(hl / 2), int(wl / 2))]
    for corner in corners:
        ch, cw = corner
        cropped = tf.image.crop_to_bounding_box(face_data, ch, cw, CROP_IMG_SIZE, CROP_IMG_SIZE)
        crops.append(tf.image.per_image_standardization(cropped))
        flipped = tf.image.flip_left_right(cropped)
        crops.append(tf.image.per_image_standardization(flipped))

    image_batch = tf.stack(crops)
    return image_batch


def inception_v3(nlabels, images, pkeep, is_training):
    batch_norm_params = {
        "is_training": is_training,
        "trainable": True,
        # Decay for the moving averages.
        "decay": 0.9997,
        # Epsilon to prevent 0s in variance.
        "epsilon": 0.001,
        # Collection containing the moving mean and moving variance.
        "variables_collections": {
            "beta": None,
            "gamma": None,
            "moving_mean": ["moving_vars"],
            "moving_variance": ["moving_vars"],
        }
    }
    weight_decay = 0.00004
    stddev = 0.1
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    with tf.variable_scope("InceptionV3", "InceptionV3", [images]) as scope:
        with tf.contrib.slim.arg_scope(
                [tf.contrib.slim.conv2d, tf.contrib.slim.fully_connected],
                weights_regularizer=weights_regularizer,
                trainable=True):
            with tf.contrib.slim.arg_scope(
                    [tf.contrib.slim.conv2d],
                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=batch_norm,
                    normalizer_params=batch_norm_params):
                net, end_points = inception_v3_base(images, scope=scope)
                with tf.variable_scope("logits"):
                    shape = net.get_shape()
                    net = avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
                    net = tf.nn.dropout(net, pkeep, name='droplast')
                    net = flatten(net, scope="flatten")

    with tf.variable_scope('output') as scope:
        weights = tf.Variable(tf.truncated_normal([2048, nlabels], mean=0.0, stddev=0.01), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nlabels], dtype=tf.float32), name='biases')
        output = tf.add(tf.matmul(net, weights), biases, name=scope.name)
        _activation_summary(output)
    return output


def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def get_checkpoint(checkpoint_path):
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        # Restore checkpoint as described in top of this program
        print(ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        return ckpt.model_checkpoint_path, global_step
    else:
        print('No checkpoint file found at [%s]' % checkpoint_path)
        exit(-1)


model_checkpoint_path, global_step = get_checkpoint(checkpoint_path)

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session()

# Required for helpers.read_tfrecords num_epochs param
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())
#
# serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
# feature_configs = {'image': tf.FixedLenFeature(shape=[CROP_IMG_SIZE * CROP_IMG_SIZE * 3], dtype=tf.float32)}
# tf_example = tf.parse_example(serialized_tf_example, feature_configs)
# images = tf.identity(tf_example['image'], name='image')  # use tf.identity() to assign name
# reshaped = tf.reshape(
#     images,
#     [1, CROP_IMG_SIZE, CROP_IMG_SIZE, 3],
#     name='reshaped'
# )
image = tf.placeholder(tf.float32, [IMG_SIZE, IMG_SIZE, 3], name='image')
batch = make_multi_crops(image)
logits = inception_v3(nlabels, batch, 1, False)
softmax_output = tf.nn.softmax(logits, name='prediction')

saver = tf.train.Saver()
saver.restore(sess, model_checkpoint_path)

builder = tf.saved_model.builder.SavedModelBuilder(export_path)

tensor_info_x = tf.saved_model.utils.build_tensor_info(image)
tensor_info_y = tf.saved_model.utils.build_tensor_info(softmax_output)

prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'image': tensor_info_x},
        outputs={'prediction': tensor_info_y},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            prediction_signature
    },
    legacy_init_op=legacy_init_op)

builder.save()

print('Done exporting!')
