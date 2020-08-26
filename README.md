ERAN for VNN COMP 2020 <img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg">
========

![High Level](https://raw.githubusercontent.com/eth-sri/eran/master/overview.png)

ETH Robustness Analyzer for Neural Networks (ERAN) is a state-of-the-art sound, precise, scalable, and extensible analyzer based on [abstract interpretation](https://en.wikipedia.org/wiki/Abstract_interpretation) for the complete and incomplete verification of MNIST, CIFAR10, and ACAS Xu based networks. ERAN produces state-of-the-art precision and performance for both complete and incomplete verification and can be tuned to provide best precision and scalability (see recommended configuration settings at the bottom). ERAN is developed at the [SRI Lab, Department of Computer Science, ETH Zurich](https://www.sri.inf.ethz.ch/) as part of the [Safe AI project](http://safeai.ethz.ch/). The goal of ERAN is to automatically verify safety properties of neural networks with feedforward, convolutional, and residual layers against input perturbations (e.g.,  L∞-norm attacks, geometric transformations, etc). 

ERAN supports networks with ReLU, Sigmoid and Tanh activations and is sound under floating point arithmetic. It employs custom abstract domains which are specifically designed for the setting of neural networks and which aim to balance scalability and precision. Specifically, ERAN supports the following four analysis:

* DeepZ [NIPS'18]: contains specialized abstract Zonotope transformers for handling ReLU, Sigmoid and Tanh activation functions.

* DeepPoly [POPL'19]: based on a domain that combines floating point Polyhedra with Intervals.

* RefineZono [ICLR'19]: combines DeepZ analysis with MILP and LP solvers for more precision. 

* RefinePoly [NeurIPS'19]: combines DeepPoly analysis with MILP and k-ReLU framework for state-of-the-art precision while maintaining scalability.

All analysis are implemented using the [ELINA](http://elina.ethz.ch/) library for numerical abstractions. More details can be found in the publications below. 



USER MANUAL
--------------------
For a detailed desciption of the options provided and the implentation of ERAN, you can download the [user manual](https://files.sri.inf.ethz.ch/eran/docs/eran_manual.pdf).

Requirements 
------------
GNU C compiler, ELINA, Gurobi's Python interface,

python3.6 or higher, tensorflow 1.11 or higher, numpy.


Installation
------------


The dependencies for ERAN can be installed step by step as follows (sudo rights might be required):

```
sudo ./install.sh
source gurobi_setup_path.sh

```


Note that to run ERAN with Gurobi one needs to obtain an academic license for gurobi from https://user.gurobi.com/download/licenses/free-academic.

To install the remaining python dependencies (numpy and tensorflow), type:

```
pip3 install -r requirements.txt
```

ERAN may not be compatible with older versions of tensorflow (we have tested ERAN with versions >= 1.11.0), so if you have an older version and want to keep it, then we recommend using the python virtual environment for installing tensorflow.


Reproducing results
-------------------
We provide results computed on our machines in the folder "tf_verify/results/<category>".  The produced files contain the verification result (SAT, UNSAT, or UNKNOWN) and the runtime in seconds. This files in this folder will be overwritten by our scripts. One can copy it to another folder for reference. 


```
cd tf_verify

./run_acasxu.sh
./run_ffn_relu.sh
./run_colt.sh
./run_oval.sh
```

Note that for parsing the Sigmoid and Tanh based onnx networks the code ERAN needs to be manually adjusted as follows:

* change line 471 of "onnx_translator.py" from
```
input_name = node.input[0]
```

to 

```
input_name = node.input[1]
```

After this change run:

```
./run_neel.sh
```


Our code for PGD attacks require tensorflow models so we converted the onnx files into ".pb" type. PGD attacks for the Sigmoid, Tanh, ReLU based fully-connected networks and "mnist_0.3.onnx" can be run as follows:

```
cd attacks

./run_attacks_neel.sh
./run_attacks_pat.sh
./run_attacks_colt.sh
```

In our results, we only report those attacks for which the overall time of running the verifier and attack is less than the specified timeout.

Publications
-------------
*  [Certifying Geometric Robustness of Neural Networks](https://www.sri.inf.ethz.ch/publications/balunovic2019geometric)

   Mislav Balunovic,  Maximilian Baader, Gagandeep Singh, Timon Gehr,  Martin Vechev
   
   NeurIPS 2019.


*  [Beyond the Single Neuron Convex Barrier for Neural Network Certification](https://www.sri.inf.ethz.ch/publications/singh2019krelu).
    
    Gagandeep Singh, Rupanshu Ganvir, Markus Püschel, and Martin Vechev. 
   
    NeurIPS 2019.

*  [Boosting Robustness Certification of Neural Networks](https://www.sri.inf.ethz.ch/publications/singh2019refinement).

    Gagandeep Singh, Timon Gehr, Markus Püschel, and Martin Vechev. 

    ICLR 2019.


*  [An Abstract Domain for Certifying Neural Networks](https://www.sri.inf.ethz.ch/publications/singh2019domain).

    Gagandeep Singh, Timon Gehr, Markus Püschel, and Martin Vechev. 

    POPL 2019.


*  [Fast and Effective Robustness Certification](https://www.sri.inf.ethz.ch/publications/singh2018effective). 

    Gagandeep Singh, Timon Gehr, Matthew Mirman, Markus Püschel, and Martin Vechev. 

    NeurIPS 2018.





Contributors
--------------

* [Gagandeep Singh](https://www.sri.inf.ethz.ch/people/gagandeep) (lead contact) - gsingh@inf.ethz.ch

* Makarchuk Hleb - hleb.makarchuk@inf.ethz.ch

* Dimitar I. Dimitrov (https://www.sri.inf.ethz.ch/people/dimitadi) - dimitadi@inf.ethz.ch

* [Markus Püschel](https://acl.inf.ethz.ch/people/markusp/) - pueschel@inf.ethz.ch

* [Martin Vechev](https://www.sri.inf.ethz.ch/vechev.php) - martin.vechev@inf.ethz.ch

License and Copyright
---------------------

* Copyright (c) 2020 [Secure, Reliable, and Intelligent Systems Lab (SRI), Department of Computer Science ETH Zurich](https://www.sri.inf.ethz.ch/)
* Licensed under the [Apache License](https://www.apache.org/licenses/LICENSE-2.0)
