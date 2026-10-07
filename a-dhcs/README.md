# Federated Learning (http://cea.ceaj.org/CN/10.3778/j.issn.1002-8331.2601-0005)

This is partly the reproduction of the paper of [一种自适应的高效压缩联邦学习方法](http://cea.ceaj.org/CN/10.3778/j.issn.1002-8331.2601-0005)   
Only experiments on MNIST and CIFAR10 (both IID and non-IID) is produced by far.
针对不同的数据集和多种学习模型，本文提出的A-DHCS方法均能超越其它同类参数压缩方法，不仅降低了模型参数的上传量、加快了训练收敛、提升了全局模型的准确率，还显著地减轻了客户端的计算负担。具体而言，相较于标准的非压缩算法FedAvg，在获得相近准确率时，本文提出方法的客户端上传数据总量仅为FedAvg的1/3230 (IID数据)、1/4207(Non-IID数据)。

Note: The scripts will be slow without the implementation of parallel computing. 

## Requirements
python>=3.6  
pytorch>=0.4

## Run

The MLP and CNN models are produced by:
> python [main_nn.py](main_nn.py)

Federated learning with MLP and CNN is produced by:
> python [main_fed.py](main_fed.py)

See the arguments in [options.py](utils/options.py). 

For example:
> python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0  

`--all_clients` for averaging over all client models

NB: for CIFAR-10, `num_channels` must be 3.

## Results
### MNIST
Results are shown in Table 1 and Table 2, with the parameters C=0.1, B=10, E=5.

Table 1. results of 10 epochs training with the learning rate of 0.01

| Model     | Acc. of IID | Acc. of Non-IID|
| -----     | -----       | ----           |
| FedAVG-MLP|  96%        | 95%            |


Table 2. results of 50 epochs training with the learning rate of 0.01

| Model     | Acc. of IID | Acc. of Non-IID|
| -----     | -----       | ----           |
| FedAVG-MLP| 97%         | 93%            |
| FedAVG-CNN| 98%         | 93%            |

各方法通信效率关系：
                                  表3 使用IID数据时客户端上传通信成本与准确率的关系（MB）
     Relationship between client upload communication costs and accuracy when using IID data
准确率	FedAvg	T-FedAvg	动态阈值1bit-cs	CFedSI	A-DHCS	fedcams	CAFL
80%	224.92	9.23	         0.07	        1.23	0.11	4.98	25.86
85%	262.40	14.99	         0.11	        1.64	0.11	7.66	35.98
90%	599.78	79.57	         0.28	        5.73	0.12	16.53	94.46
95%	2624.04		         5.80	        37.26	0.74	31.67	
96%	4198.46			                63.06	1.30	34.08
	
                                  表4 使用Non-IID数据时客户端上传通信成本与准确率的关系（MB）
   The relationship between client upload communication cost and accuracy when using Non-IID data
准确率	FedAvg	T-FedAvg	动态阈值1bit-cs	CFedSI	A-DHCS	fedcams	CAFL
75%	196.54	36.90	        0.25	        2.87	 0.30	10.30	96.71
80%	432.38	53.05	        0.46	        2.87     0.38	11.93	109.08
85%	628.92	91.10	        1.58	        5.73     0.49	18.66	147.31
90%	1965.36			                18.02	 1.45	34.11	
93%	7153.91			                39.31	 1.91		
95%					                 3.93		

## Ackonwledgements
Acknowledgements give to [jialongj](https://github.com/jialongj/pysyft).

## References
McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Artificial Intelligence and Statistics (AISTATS), 2017.

## Cite As
秦乐, 蒋佳龙, 谢平, 饶鑫平, 易玉根. 一种自适应的高效压缩联邦学习方法[J]. 计算机工程与应用, DOI: 10.3778/j.issn.1002-8331.2601-0005.


