#!/bin/sh

for i in $(seq 0 24);
do
        timeout 900 python3 -u pgd.py --im $i --model tansig_200_50_onnx.pb  --epsilon 5 --pgd_epsilon 0.5 >> tansig_200_50_eps5.txt
done


for i in $(seq 0 24); 
do
	timeout 900 python3 -u pgd.py --im $i --model tansig_200_50_onnx.pb  --epsilon 12 --pgd_epsilon 0.5 >> tansig_200_50_eps12.txt
done

for i in $(seq 0 24);
do
        timeout 900 python3 -u pgd.py --im $i --model tansig_200_100_50_onnx.pb  --epsilon 5 --pgd_epsilon 0.5 >> tansig_200_100_50_eps5.txt
done


for i in $(seq 0 24);
do
        timeout 900 python3 -u pgd.py --im $i --model tansig_200_100_50_onnx.pb  --epsilon 12 --pgd_epsilon 0.5 >> tansig_200_100_50_eps12.txt
done
