python3 -u . --netname  relu_image_models/ffn_relu/mnist-net_256x2.onnx --dataset mnist --domain deeppoly --epsilon 0.02 --complete True --timeout_milp 90 --subset comp > results/FFN_relu/mnist_256x2_eps0.02.txt

python3 -u . --netname  relu_image_models/ffn_relu/mnist-net_256x2.onnx --dataset mnist --domain deeppoly --epsilon 0.05 --complete True --timeout_milp 90 --subset comp > results/FFN_relu/mnist_256x2_eps0.05.txt

python3 -u . --netname  relu_image_models/ffn_relu/mnist-net_256x4.onnx --dataset mnist --domain deeppoly --epsilon 0.02 --complete True --timeout_milp 90 --subset comp > results/FFN_relu/mnist_256x4_eps0.02.txt

python3 -u . --netname  relu_image_models/ffn_relu/mnist-net_256x4.onnx --dataset mnist --domain deeppoly --epsilon 0.05 --complete True --timeout_milp 90 --subset comp  > results/FFN_relu/mnist_256x4_eps0.05.txt

python3 -u . --netname  relu_image_models/ffn_relu/mnist-net_256x6.onnx --dataset mnist --domain deeppoly --epsilon 0.02  --timeout_milp 90 --subset comp --complete True > results/FFN_relu/mnist_256x6_eps0.02.txt

python3 -u . --netname  relu_image_models/ffn_relu/mnist-net_256x6.onnx --dataset mnist --domain deeppoly --epsilon 0.05 --complete True --timeout_milp 90 --subset comp > results/FFN_relu/mnist_256x6_eps0.05.txt

