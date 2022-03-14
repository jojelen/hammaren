Hammaren
========

Utility library that serves as a template for computer-vision projects.

## Install

### Dependencies

- python3.7
- OpenCV
- TensorflowLite (optional)

#### Tflite EdgeTPU

We use the [coral API](https://coral.ai/docs/accelerator/get-started/#pycoral-on-linux)
for running models on the EdgeTPU. To install these dependencies, on Debian,
run
```
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update
sudo apt-get install libedgetpu1-std
sudo apt-get install python3-pycoral
```

## Useful stuff

### Supported capture resolutions in v4l2

Video4Linux is a backend in OpenCV. See supported resolutions:
```
v4l2-ctl -d /dev/video0 --list-formats-ext
```

