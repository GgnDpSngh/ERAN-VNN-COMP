import onnx

from onnx_tf.backend import prepare
import argparse

parser = argparse.ArgumentParser(description='Convert onnx')
parser.add_argument('--input', type=str, help='Input path')
parser.add_argument('--output', type=str, help='Output path')

args = parser.parse_args()
onnx_model = onnx.load(args.input)  # load onnx model
tf_rep = prepare(onnx_model)  # prepare tf representation
tf_rep.export_graph(args.output)  # export the model
