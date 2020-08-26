python3 -u . --netname sigmoid_tanh_models/logsig_200_50_onnx.onnx --dataset mnist --domain deeppoly --epsilon 5 > results/neel/logsig_200_50.txt

python3 -u . --netname sigmoid_tanh_models/logsig_200_100_50_onnx.onnx --dataset mnist --domain deeppoly --epsilon 5 > results/neel/logsig_200_100_50.txt

python3 -u . --netname sigmoid_tanh_models/tansig_200_50_onnx.onnx --dataset mnist --domain deeppoly --epsilon 5 > results/neel/tansig_200_50.txt

python3 -u . --netname sigmoid_tanh_models/tansig_200_100_50_onnx.onnx --dataset mnist --domain deeppoly --epsilon 5 > results/neel/tansig_200_100_50.txt
