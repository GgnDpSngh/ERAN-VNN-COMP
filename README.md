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
Note the Sigmoid and Tanh based onnx networks do not parse with the provided code of ERAN. For running with ERAN, one has to 


```
cd tf_verify

./run_acasxu.sh
./run_ffn_relu.sh
./run_colt.sh
./run_oval.sh
```

The results are collected in "results/<category>" where the produced files contain the verification result (SAT, UNSAT, or UNKNOWN) and the runtime in seconds.

PGD attacks for the Sigmoid, Tanh, ReLU based fully-connected networks and "mnist_0.3.onnx" can be run as follows:

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




Neural Networks and Datasets
---------------

We provide a number of pretrained MNIST and CIAFR10 defended and undefended feedforward and convolutional neural networks with ReLU, Sigmoid and Tanh activations trained with the PyTorch and TensorFlow frameworks. The adversarial training to obtain the defended networks is performed using PGD and [DiffAI](https://github.com/eth-sri/diffai). 

| Dataset  |   Model  |  Type   | #units | #layers| Activation | Training Defense| Download |
| :-------- | :-------- | :-------- | :-------------| :-------------| :------------ | :------------- | :---------------:|
| MNIST   | 3x50 | fully connected | 110 | 3    | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_3_50.tf)|
|         | 3x100 | fully connected | 210 | 3    | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_3_100.tf)|
|         | 5x100 | fully connected | 510 | 5    | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_5_100.tf)|
|         | 6x100 | fully connected | 510 | 6    | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_6_100.tf)|
|         | 9x100 | fully connected | 810 | 9    | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_9_100.tf)|
|         | 6x200 | fully connected | 1,010 | 6   | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_6_200.tf)|
|         | 9x200 | fully connected | 1,610 | 9   | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_9_200.tf)|
|         | 6x500 | fully connected | 3,000 | 6   | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnRELU__Point_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 | 6   | ReLU  | PGD &epsilon;=0.1 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnRELU__PGDK_w_0.1_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 |  6  | ReLU | PGD &epsilon;=0.3 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnRELU__PGDK_w_0.3_6_500.pyt)|
|         | 6x500 | fully connected | 3,000  | 6   | Sigmoid | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnSIGMOID__Point_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 |  6  | Sigmoid | PGD &epsilon;=0.1 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnSIGMOID__PGDK_w_0.1_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 | 6   | Sigmoid | PGD &epsilon;=0.3 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnSIGMOID__PGDK_w_0.3_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 | 6 |    Tanh | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnTANH__Point_6_500.pyt)|
|         | 6x500 |  fully connected| 3,000 | 6   | Tanh | PGD &epsilon;=0.1 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnTANH__PGDK_w_0.1_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 | 6   |  Tanh | PGD &epsilon;=0.3 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnTANH__PGDK_w_0.3_6_500.pyt)|
|         | 4x1024 | fully connected | 3,072 | 4   | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_4_1024.tf)|
|         |  ConvSmall | convolutional | 3,604 | 3  | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convSmallRELU__Point.pyt)|
|         |  ConvSmall | convolutional | 3,604 | 3  | ReLU | PGD | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convSmallRELU__PGDK.pyt) |
|         |  ConvSmall | convolutional | 3,604 | 3  | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convSmallRELU__DiffAI.pyt) |
|         | ConvMed | convolutional | 5,704 | 3  | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGRELU__Point.pyt) |
|         | ConvMed | convolutional | 5,704 | 3   | ReLU | PGD &epsilon;=0.1 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGRELU__PGDK_w_0.1.pyt) |
|         | ConvMed | convolutional | 5,704 | 3   | ReLU | PGD &epsilon;=0.3 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGRELU__PGDK_w_0.3.pyt) |
|         | ConvMed | convolutional | 5,704 | 3   | Sigmoid | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGSIGMOID__Point.pyt) |
|         | ConvMed | convolutional | 5,704 | 3   | Sigmoid | PGD &epsilon;=0.1 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGSIGMOID__PGDK_w_0.1.pyt) | 
|         | ConvMed | convolutional | 5,704 | 3   | Sigmoid | PGD &epsilon;=0.3 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGSIGMOID__PGDK_w_0.3.pyt) | 
|         | ConvMed | convolutional | 5,704 | 3   | Tanh | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGTANH__Point.pyt) |
|         | ConvMed | convolutional | 5,704 | 3   | Tanh | PGD &epsilon;=0.1 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGTANH__PGDK_w_0.1.pyt) | 
|         | ConvMed | convolutional | 5,704 | 3   |  Tanh | PGD &epsilon;=0.3 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGTANH__PGDK_w_0.3.pyt) |
|         | ConvMaxpool | convolutional | 13,798 | 9 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_conv_maxpool.tf)|
|         | ConvBig | convolutional | 48,064 | 6  | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convBigRELU__DiffAI.pyt) |
|         | ConvSuper | convolutional | 88,544 | 6  | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convSuperRELU__DiffAI.pyt) |
|         | Skip      | Residual | 71,650 | 9 | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/skip__DiffAI.pyt) |
| CIFAR10 | 4x100 | fully connected | 410 | 4 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_relu_4_100.tf) |
|         | 6x100 | fully connected | 610 | 6 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_relu_6_100.tf) |
|         | 9x200 | fully connected | 1,810 | 9 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_relu_9_200.tf) |
|         | 6x500 | fully connected | 3,000 | 6   | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnRELU__Point_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 | 6   | ReLU | PGD &epsilon;=0.0078 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnRELU__PGDK_w_0.0078_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 | 6   | ReLU | PGD &epsilon;=0.0313 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnRELU__PGDK_w_0.0313_6_500.pyt)| 
|         | 6x500 | fully connected | 3,000 | 6   | Sigmoid | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnSIGMOID__Point_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 | 6   | Sigmoid | PGD &epsilon;=0.0078 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnSIGMOID__PGDK_w_0.0078_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 | 6   | Sigmoid | PGD &epsilon;=0.0313 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnSIGMOID__PGDK_w_0.0313_6_500.pyt)| 
|         | 6x500 | fully connected | 3,000 | 6   | Tanh | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnTANH__Point_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 | 6   | Tanh | PGD &epsilon;=0.0078 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnTANH__PGDK_w_0.0078_6_500.pyt)|
|         | 6x500 | fully connected | 3,000 | 6   | Tanh | PGD &epsilon;=0.0313 |  [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnTANH__PGDK_w_0.0313_6_500.pyt)| 
|         | 7x1024 | fully connected | 6,144 | 7 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_relu_7_1024.tf) |
|         | ConvSmall | convolutional | 4,852 | 3 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convSmallRELU__Point.pyt)|
|         | ConvSmall   | convolutional  | 4,852 | 3  | ReLU  | PGD | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convSmallRELU__PGDK.pyt)|
|         | ConvSmall  | convolutional | 4,852 | 3  | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convSmallRELU__DiffAI.pyt)|
|         | ConvMed | convolutional | 7,144 | 3 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGRELU__Point.pyt) |
|         | ConvMed | convolutional | 7,144 | 3   | ReLU | PGD &epsilon;=0.0078 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGRELU__PGDK_w_0.0078.pyt) |
|         | ConvMed | convolutional | 7,144 | 3   | ReLU | PGD &epsilon;=0.0313 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGRELU__PGDK_w_0.0313.pyt) | 
|         | ConvMed | convolutional | 7,144 | 3   | Sigmoid | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGSIGMOID__Point.pyt) |
|         | ConvMed | convolutional | 7,144 | 3   | Sigmoid | PGD &epsilon;=0.0078 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGSIGMOID__PGDK_w_0.0078.pyt) |
|         | ConvMed | convolutional | 7,144 | 3   | Sigmoid | PGD &epsilon;=0.0313 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGSIGMOID__PGDK_w_0.0313.pyt) | 
|         | ConvMed | convolutional | 7,144 | 3   | Tanh | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGTANH__Point.pyt) |
|         | ConvMed | convolutional | 7,144 | 3   | Tanh | PGD &epsilon;=0.0078 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGTANH__PGDK_w_0.0078.pyt) |
|         | ConvMed | convolutional | 7,144 | 3   | Tanh | PGD &epsilon;=0.0313 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGTANH__PGDK_w_0.0313.pyt) |  
|         | ConvMaxpool | convolutional | 53,938 | 9 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_conv_maxpool.tf)|
|         | ConvBig | convolutional | 62,464 | 6 | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convBigRELU__DiffAI.pyt) | 
|         | ResNet18 | Residual | 558K | 18 | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ResNet18_DiffAI.pyt) | 

We provide the first 100 images from the testset of both MNIST and CIFAR10 datasets in the 'data' folder. Our analyzer first verifies whether the neural network classifies an image correctly before performing robustness analysis. In the same folder, we also provide ACAS Xu networks and property specifications.

Experimental Results
--------------
We ran our experiments for the feedforward networks on a 3.3 GHz 10 core Intel i9-7900X Skylake CPU with a main memory of 64 GB whereas our experiments for the convolutional networks were run on a 2.6 GHz 14 core Intel Xeon CPU E5-2690 with 512 GB of main memory. We first compare the precision and performance of DeepZ and DeepPoly vs [Fast-Lin](https://github.com/huanzhang12/CertifiedReLURobustness) on the MNIST 6x100 network in single threaded mode. It can be seen that DeepZ has the same precision as Fast-Lin whereas DeepPoly is more precise while also being faster.

![High Level](https://files.sri.inf.ethz.ch/eran/plots/mnist_6_100.png)

In the following, we compare the precision and performance of DeepZ and DeepPoly on a subset of the neural networks listed above in multi-threaded mode. In can be seen that DeepPoly is overall more precise than DeepZ but it is slower than DeepZ on the convolutional networks. 




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
