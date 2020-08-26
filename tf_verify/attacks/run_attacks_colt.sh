#!/bin/sh

for i in $(seq 0 100);
do
        timeout 300 python3 -u pgd_mislav.py --im $i --model mnist_0.3.pb --mean 0.1307 --std 0.3081 --epsilon 0.3 --pgd_epsilon 0.1 >> mnist_03.txt
done





