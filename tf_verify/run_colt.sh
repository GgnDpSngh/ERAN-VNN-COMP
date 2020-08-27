timeout 300 python3 -u . --netname relu_image_models/colt/mnist_0.1.onnx --dataset mnist --domain deeppoly --epsilon 0.1  --mean 0.1307 --std 0.3081 --complete True --timeout_milp 50 > results/colt/mnist_0.1.txt

timeout 300 python3 -u . --netname relu_image_models/colt/mnist_0.3.onnx --dataset mnist --domain refinepoly --epsilon 0.3  --mean 0.1307 --std 0.3081  --timeout_milp 50  --sparse_n 4 > results/colt/mnist_0.3.txt

timeout 300 python3 -u . --netname relu_image_models/colt/cifar10_8_255.onnx --dataset cifar10 --domain refinepoly --epsilon 0.03137254901960784  --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010  --timeout_lp 200  --sparse_n 13 > results/colt/cifar_8_255.txt

timeout 300 python3 -u . --netname relu_image_models/colt/cifar10_2_255.onnx --dataset cifar10 --domain refinepoly --epsilon 0.00784313725490196 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --timeout_lp 200  --sparse_n 13 > results/colt/cifar_2_255.txt





