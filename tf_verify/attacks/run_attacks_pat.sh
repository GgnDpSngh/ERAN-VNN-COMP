#!/bin/sh

for i in $(seq 0 24);
do
        timeout 900 python3 -u pgd_pat.py --im $i --model mnist-net_256x4.pb  --epsilon 0.05 --pgd_epsilon 0.1 >> mnist-net_256x4_eps_05.txt
done


for i in $(seq 0 24); 
do
	timeout 900 python3 -u pgd_pat.py --im $i --model mnist-net_256x6.pb  --epsilon 0.05 --pgd_epsilon 0.1 >> mnist-net_256x6_eps_05.txt
done




