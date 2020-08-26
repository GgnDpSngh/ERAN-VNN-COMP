python3 -u . --netname sigmoid_tanh_models/logsig_200_50_onnx.onnx --dataset mnist --domain deeppoly --epsilon 5 --subset comp > results/neel/logsig_200_50_eps5.txt

python3 -u . --netname sigmoid_tanh_models/logsig_200_50_onnx.onnx --dataset mnist --domain deeppoly --epsilon 12 --subset comp > results/neel/logsig_200_50_eps12.txt

python3 -u . --netname sigmoid_tanh_models/logsig_200_100_50_onnx.onnx --dataset mnist --domain deeppoly --epsilon 5 --subset comp > results/neel/logsig_200_100_50_eps5.txt

python3 -u . --netname sigmoid_tanh_models/logsig_200_100_50_onnx.onnx --dataset mnist --domain deeppoly --epsilon 12 --subset comp > results/neel/logsig_200_100_50_eps12.txt

python3 -u . --netname sigmoid_tanh_models/tansig_200_50_onnx.onnx --dataset mnist --domain deeppoly --epsilon 5 --subset comp > results/neel/tansig_200_50_eps5.txt

python3 -u . --netname sigmoid_tanh_models/tansig_200_50_onnx.onnx --dataset mnist --domain deeppoly --epsilon 12 --subset comp > results/neel/tansig_200_50_eps12.txt

python3 -u . --netname sigmoid_tanh_models/tansig_200_100_50_onnx.onnx --dataset mnist --domain deeppoly --epsilon 5 --subset comp > results/neel/tansig_200_100_50_eps5.txt

python3 -u . --netname sigmoid_tanh_models/tansig_200_100_50_onnx.onnx --dataset mnist --domain deeppoly --epsilon 12 --subset comp > results/neel/tansig_200_100_50_eps12.txt
