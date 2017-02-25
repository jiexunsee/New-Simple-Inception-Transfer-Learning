# A classifier using transfer learning on the inception model

This is a response to Siraj's challenge of the week.

This is a classifier made with basic TensorFlow, using Transfer Learning from the Inception model. Data is taken from [this](https://www.kaggle.com/c/dogs-vs-cats) Kaggle competition, as recommended by Siraj.

## Pipeline
JPG images --> Inception --> 2048-dimensional vector --> Fully connected layer --> Prediction

## How to use

### Create bottlenecks
First, create bottleneck files for use in the training of the one-layer neural network.

`python producebottlenecks.py`

This will download the inception network, run your images in the train directory through the inception network, then output the images as bottleneck text files in the bottlenecks folder.

### Train on the bottleneck files

`python classifier.py`

This will train a single layer neural network on the bottleneck files. The trained graph will be saved as savedgraph.pb. Also, you can view TensorBoard using `tensorboard --logdir=log` on the command line.

### Test on any image

`python testclassify.py --image_dir=path/to/image`

## Notes
Only 1000 images of cats and 1000 images of dogs are in the train folder. I did not use all the images as that would take quite a lot of time to produce bottleneck files for. 1000 images of each category are more than enough to produce a very high accuracy.

The code for downloading the inception model and producing the bottleneck text files was taken from the TensorFlow github -> examples -> image_retraining. However, I only took the relevant portions, hence making the code much shorter and easier to understand.
