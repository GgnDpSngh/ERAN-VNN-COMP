!/bin/bash
#L_infinity attack
rm results/acasxu/prop*
for net in ../data/acasxu/nets/*.onnx;
do

timeout 300 python3 -u . --netname $net --specnumber 1 --domain deeppoly --dataset acasxu --complete True --timeout_milp 10 >> results/acasxu/prop1.txt
        
	
done

for net in ../data/acasxu/nets/*.onnx;
do

        timeout 300 python3 -u . --netname $net --specnumber 2 --domain deeppoly --dataset acasxu --complete True --timeout_milp 10 >> results/acasxu/prop2.txt
        
	
done

for net in ../data/acasxu/nets/*.onnx;
do

        timeout 300 python3 -u . --netname $net --specnumber 3 --domain deeppoly --dataset acasxu --complete True --timeout_milp 10 >> results/acasxu/prop3.txt
        
	
done

for net in ../data/acasxu/nets/*.onnx;
do

        timeout 300 python3 -u . --netname $net --specnumber 4 --domain deeppoly --dataset acasxu --complete True  --timeout_milp 10 >> results/acasxu/prop4.txt
        
	
done


timeout 300 python3 -u . --netname ../data/acasxu/nets/ACASXU_run2a_1_1_batch_2000.onnx --specnumber 5 --domain deeppoly --dataset acasxu --complete True --timeout_milp 10  > results/acasxu/prop5.txt

timeout 300 python3 -u . --netname ../data/acasxu/nets/ACASXU_run2a_1_1_batch_2000.onnx --specnumber 6 --domain deeppoly --dataset acasxu --complete True --timeout_milp 10 > results/acasxu/prop6.txt

timeout 300 python3 -u . --netname ../data/acasxu/nets/ACASXU_run2a_1_9_batch_2000.onnx --specnumber 7 --domain deeppoly --dataset acasxu --complete True --timeout_milp 1  > results/acasxu/prop7.txt

timeout 300 python3 -u . --netname ../data/acasxu/nets/ACASXU_run2a_2_9_batch_2000.onnx --specnumber 8 --domain deeppoly --dataset acasxu --complete True --timeout_milp 10  > results/acasxu/prop8.txt

timeout 300 python3 -u . --netname ../data/acasxu/nets/ACASXU_run2a_3_3_batch_2000.onnx --specnumber 9 --domain deeppoly --dataset acasxu --complete True --timeout_milp 10  > results/acasxu/prop9.txt

timeout 300 python3 -u . --netname ../data/acasxu/nets/ACASXU_run2a_4_5_batch_2000.onnx --specnumber 10 --domain deeppoly --dataset acasxu --complete True --timeout_milp 10  > results/acasxu/prop10.txt
