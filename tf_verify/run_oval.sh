python3 -u . --domain deeppoly --netname relu_image_models/oval/cifar_base_kw.onnx --dataset cifar10 --mean 0.485 0.456 0.406 --std 0.225 0.225 0.225 --timeout_milp 3590  --complete True --subset base100 --target ../data/base_prop.csv --epsfile ../data/base_eps.csv --normalized_region False > results/oval/base.txt

python3 -u . --domain deeppoly --netname relu_image_models/oval/cifar_wide_kw.onnx --dataset cifar10 --mean 0.485 0.456 0.406 --std 0.225 0.225 0.225 --timeout_milp 3590 --complete True --subset wide100 --target ../data/wide_prop.csv --epsfile ../data/wide_eps.csv --normalized_region False > results/oval/wide.txt

python3 -u . --domain refinepoly --netname relu_image_models/oval/cifar_deep_kw.onnx --dataset cifar10 --mean 0.485 0.456 0.406 --std 0.225 0.225 0.225 --timeout_milp 3590  --sparse_n 10 --subset deep100 --target ../data/deep_prop.csv --epsfile ../data/deep_eps.csv --normalized_region False > results/oval/deep.txt



