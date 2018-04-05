"""
Load WideResNet model and save it as SavedModel format to use for tf-serving in the future
"""
import tensorflow as tf
from keras import backend as K
from wide_resnet import WideResNet

weight_path = './data/wide_resnet.hdf5'
export_path = './data/gender_wide_resnet_saved_model'

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=False)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Missing this was the source of one of the most challenging an insidious bugs that I've ever encountered.
# Without explicitly linking the session the weights for the dense layer added below don't get loaded
# and so the model returns random results which vary with each model you upload because of random seeds.
K.set_session(sess)

# Use this only for export of the model.
# This must come before the instantiation of ResNet50
K._LEARNING_PHASE = tf.constant(0)
K.set_learning_phase(0)

model = WideResNet()()
# Training Not included; We're going to load pretrained weights
model.load_weights(weight_path)

# Import the libraries needed for saving models
# Note that in some other tutorials these are framed as coming from tensorflow_serving_api which is no longer correct
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants

# I want the full prediction tensor out, not classification. This format: {"image": model.input} took me a while to track down
prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def({"image": model.input},
                                                                                {"prediction_g": model.output[0],
                                                                                 "prediction_a": model.output[1]})

# export_path is a directory in which the model will be created
builder = saved_model_builder.SavedModelBuilder(export_path)
legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

# Initialize global variables and the model
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

# Add the meta_graph and the variables to the builder
builder.add_meta_graph_and_variables(
    sess, [tag_constants.SERVING],
    signature_def_map={
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            prediction_signature,
    },
    legacy_init_op=legacy_init_op)
# save the graph
builder.save()

print('Done exporting!')
