[![Build Status](https://travis-ci.com/500px/vision-ai.svg?token=s87GfvdYBuoMmUAGs4TA&branch=master)](https://travis-ci.com/500px/vision-ai)

# Vision AI Service

An HTTP service to perform image classification and analysis. Visit the [project wiki](https://github.com/500px/vision-ai/wiki) for a system architecture overview.

## Setup

- [Install Caffe](http://caffe.berkeleyvision.org/installation.html) (version [rc5](https://github.com/BVLC/caffe/releases/tag/rc5)) with python bindings and add to `PYTHONPATH`
- Run the following commands inside the project folder

```bash
pip2 install virtualenv                 # Install virtualenv if haven't already
virtualenv -p /usr/bin/python2.7 .venv  # Create clean virtual environment
source .venv/bin/activate               # Activate virtual environment
pip install -r requirements.txt         # Install dependencies

# Download model files (locations can be changed in config.py)
aws s3 sync s3://500px-vision/0_models/imagenet/ data/imagenet
aws s3 sync s3://500px-vision/0_models/bvlc_googlenet/ data/bvlc_googlenet/
aws s3 cp s3://500px-vision/0_models/vision_ai/joint_embeddings/w2v/w2v_vocab.pkl data/w2v/
aws s3 sync s3://500px-vision/0_models/vision_ai/joint_embeddings/ data/

aws s3 cp s3://500px-vision/0_models/vision_ai/20151005_lsh_pca_config_image_embed.pkl data/
aws s3 cp s3://500px-vision/0_models/vision_ai/20170212_lsh_pca_config_joint_embed.pkl data/

aws s3 cp s3://500px-vision/0_models/vision_ai/nsfw/nsfw_lr.pkl data/
aws s3 cp s3://500px-vision/0_models/vision_ai/spam/spam_lr.pkl data/
```

- Generate gRPC code from TensorFlow Serving protocol buffers

```bash
pip install grpcio-tools==1.1.0
sh compile_tfs_proto.sh 0.5.1 /Users/JVillella/Development/vision-ai # abs path only
```

## Running in development

- Run the TensorFlow docker container with our [platform](https://github.com/500px/platform) docker-compose
aws s3 cp s3://500px-vision/0_models/vision_ai/joint_embeddings/tensorflow-serving.conf data/

- Run the docker container and docker cp the configure file, and update the path in conf file.
docker exec -it 'docker ps | grep joint-embedding | awk '{print $1}'' bash


```bash
cd <platform-dir> # https://github.com/500px/platform
docker-init
docker-compose pull # If you haven't already
docker-compose up

cd <vision-ai> # This project
docker-init
```

- Start Vision AI Service

```bash
source .venv/bin/activate
python visionai.py
```

## Testing

```bash
python -m unittest discover tests/
```

## Tweak controlled vocabulary

1. Download CSV for revised controlled vocab ([example](https://docs.google.com/spreadsheets/d/1fc7aWfggs-TqrmZLcYarOtrwnsTmKuAQLVvVeTuB3b0/edit#gid=0))
2. Generate new w2v vocab by running [Vision AI Controlled Vocabulary notebook](./notes/joint-embeddings/vision-ai-controlled-vocabulary.ipynb)
3. Upload output to S3, e.x.

```bash
aws s3 cp --dryrun w2v_vocab.pkl s3://500px-vision/0_models/vision_ai/joint_embeddings/w2v/w2v_vocab.pkl
```

## Uploading new models

To upload a new TensorFlow model you'll need to freeze and export the graph. You can do so by running our [export notebook](https://github.com/500px/vision-ai/blob/502e085c4bf0b7db8d792fb195b42b86a300c00b/notes/joint-embeddings/export_model_tensorflow_serving.ipynb). Make sure to upload this new graph (`s3://500px-vision/0_models/vision_ai/joint_embeddings/export/<version>`) under a new directory with a higher version number. TensorFlow Serving will pick it up automatically. Also, consider if we need to backfill our images through this new model to recompute their embeddings and overwrite those in the vision RDS.

## Bulk tagging

See the bulk tagging [instructions](notes/vision-apis/README.md) for setup, and running.

---

## API

### Classification

```
POST /classify
```

##### Parameters

- `image` _(required)_ — JPEG image to classify. Max size 0.5Mb.
- `model` - controls whether to use the image embeddings model (`image-embeddings`; default) or joint embeddings model (`joint-embeddings`).
- `resize` — controls whether to resize the image. `0` (default) - no resizing performed; the input image size must match the model's input dimensions specified in [config.py](https://github.com/500px/vision-ai/blob/master/config.py). `1` - resize the input image to match the model's input dimensions. Only applies to `model=image-embeddings`.
- `oversample` - controls whether to oversample the image. Oversampling generates 10 crops, (4 + 1 center) * 2 mirror from the input image. The final predictions are the average of all crops which have slightly higher accuracy with the price of CPU/GPU time. Only applies to `model=image-embeddings`.

##### Response

JSON containing:

- `model_id` - model id that was used for classification.
- `classes` - list of objects containing a class id, its corresponding label, and a probability, sorted by descending probability.
- `top_k` - Number of labels to return.
- `embeddings` - high dimensional vector that represents the image.
- `lsh` - locality sensitive hash for `embeddings` vector.

##### Examples

Calculate joint embeddings for image.

```bash
curl -F "image=@cat.jpg" \
     -F "labels=cat" \
     -F "labels=kitty" \
     -F "model=joint-embeddings" \
     localhost:5000/classify
```

```json
{
  "classes": [
    {
      "class_id": 1511,
      "label": "kitty",
      "probability": 0.8619504837844371
    },
    {
      "class_id": 287,
      "label": "cat",
      "probability": 0.85824586419635
    },
    ...
  ],
  "embeddings": [
    132,
    104,
    106,
    105,
    ...
  ],
  "lsh": 10931330,
  "model_id": "20170221_devise"
}
```
