#! -*- coding: utf-8 -*-

"""
  [main]
    python main.py --help
"""

#---------------------------------
# モジュールのインポート
#---------------------------------
import os
import argparse
import urllib.request
import tarfile

import onnx

import tensorflow as tf
import tf2onnx

from keras.applications.mobilenet import MobileNet
import keras2onnx

#---------------------------------
# 定数定義
#---------------------------------

#---------------------------------
# 関数
#---------------------------------

#---------------------------------
# クラス
#---------------------------------

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
	# --- local functions ---
	"""
	  関数名: _arg_parser
	  説明：引数を解析して値を取得する
	"""
	def _arg_parser():
		parser = argparse.ArgumentParser(description='Mobile Net v1モデルをダウンロードしてONNX変換するツール', formatter_class=argparse.RawTextHelpFormatter)

		# --- 引数を追加 ---
		parser.add_argument('--save_dir', dest='save_dir', type=str, default=None, help='モデル保存先のディレクトリ', required=True)

		args = parser.parse_args()

		return args

	# --- 引数処理 ---
	args = _arg_parser()

	# --- TensorFlow ---
	print('[INFO] TensorFlow mobilenet_v1_1.0_224 downloading ...')
	save_dir = os.path.join(args.save_dir, 'tensorflow/mobilenet_v1_1.0_224')
	save_file = os.path.join(save_dir, 'mobilenet_v1_1.0_224.tgz')
	os.makedirs(save_dir, exist_ok=True)
	url = 'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz'
	urllib.request.urlretrieve(url, '{}'.format(save_file))

	print('[INFO] DONE')
	print('[INFO] TensorFlow mobilenet_v1_1.0_224 extracting ...')
	with tarfile.open(save_file, 'r:gz') as f:
def is_within_directory(directory, target):
	
	abs_directory = os.path.abspath(directory)
	abs_target = os.path.abspath(target)

	prefix = os.path.commonprefix([abs_directory, abs_target])
	
	return prefix == abs_directory

def safe_extract(tar, path=".", members=None, *, numeric_owner=False):

	for member in tar.getmembers():
		member_path = os.path.join(path, member.name)
		if not is_within_directory(path, member_path):
			raise Exception("Attempted Path Traversal in Tar File")

	tar.extractall(path, members, numeric_owner=numeric_owner) 
	

safe_extract(f, path=save_dir)
	print('[INFO] DONE')

	print('[INFO] TensorFlow mobilenet_v1_1.0_224 frozen to ONNX ...')
	with tf.Graph().as_default():
		with open(os.path.join(save_dir, 'mobilenet_v1_1.0_224_frozen.pb'), 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			tf.import_graph_def(graph_def, name='')
		onnx_graph = tf2onnx.tfonnx.process_tf_graph(tf.get_default_graph(), \
			input_names=['input:0'], \
			output_names=['MobilenetV1/Predictions/Reshape_1:0'])
		model_proto = onnx_graph.make_model('test')
	onnx_dir = os.path.join(save_dir, 'onnx_model')
	os.makedirs(onnx_dir, exist_ok=True)
	with open(os.path.join(onnx_dir, 'model.onnx'), 'wb') as f:
		f.write(model_proto.SerializeToString())
	print('[INFO] DONE')


	# --- Keras ---
	print('[INFO] Keras MobileNet v1 downloading ...')
	model = MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)

	save_dir = os.path.join(args.save_dir, 'keras/mobilenet_v1')
	save_file = os.path.join(save_dir, 'mobilenet_v1.h5')
	os.makedirs(save_dir, exist_ok=True)
	model.save(save_file)
	print('[INFO] DONE')

	print('[INFO] Keras MobileNet v1 to ONNX ...')
	save_dir = os.path.join(args.save_dir, 'keras/mobilenet_v1/onnx_model')
	os.makedirs(save_dir, exist_ok=True)
	save_file = os.path.join(save_dir, 'model.onnx')
	onnx_model = keras2onnx.convert_keras(model, model.name)
	onnx.save_model(onnx_model, save_file)
	print('[INFO] DONE')

