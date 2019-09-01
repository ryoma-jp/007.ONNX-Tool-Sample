# 007.ONNX-Tool-Sample

MobileNet v1のPre-Trainedモデルを用いたONNXモデル変換サンプル

* TensorFlow → ONNX：○
* Keras → ONNX：○
* PyTorch → ONNX：× … PyTorchはPre-Trained ModelがMobileNet v2
https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
※ Pre-Trainedでないものならありそう
https://modelzoo.co/model/pytorch-mobilenet
* Caffe2 → ONNX：× … Caffe2はPre-Trained ModelがMobileNet v2
https://github.com/caffe2/models



