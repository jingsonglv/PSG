# Path-aware Siamese Graph Neural Network for Link Prediction of OGB (PSG-OGB)
This repository provides evaluation codes of PSG on ogbl-ddi dataset for OGB link property prediction task. The idea of PSG is described in the following article:
>**Path-aware Siamese Graph Neural Network for Link Prediction (https://arxiv.org/pdf/2208.05781.pdf)**



## Requirements
The code is implemented with PyTorch and PyTorch Geometric. Requirments:  
1. python=3.8.13
2. pytorch=1.8.0
3. ogb=1.3.3
4. torch-geometric=2.0.4



## Train and Predict
### ogbl-ddi:  

	nohup python main.py --data_name=ogbl-ddi --emb_hidden_channels=512 --gnn_hidden_channels=512 --mlp_hidden_channels=512 --num_neg=3 --dropout=0.3  --node_emb=512 --num_samples=3  --encoder=sage-edge --lr=0.001 --epochs=500 --runs=10 --device=0&


## Results
The performances of PSG together with other GNN-based methods on OGB-DDI task are listed as below:

| Algorithm                                    | Test Hits@20  | Val Hits@20             |
| ---------- | :-----------:  | :-----------: |
| SEAL                                | 0.3056 ± 0.0386          | 0.2849 ± 0.0269         |
| GCN       | 0.3707 ± 0.0507         | 0.5550 ± 0.0208          |
| GraphSAGE | 0.5390 ± 0.0474         | 0.6262 ± 0.0037        |
| GCN+JKNet    | 0.6056 ± 0.0869         | 0.6776 ± 0.0095          |
| DEA + JKNet   | 0.7672 ± 0.0265 | 0.6713 ± 0.0071  |
| CFLP (w/ JKNet)    | 0.8608 ± 0.0198 | 0.8405 ± 0.0284 |
| GraphSAGE + Edge Attr   | 0.8781 ± 0.0474 | 0.8044 ± 0.0404 |
| PLNLP   | 0.9088 ± 0.0313 | 0.8242 ± 0.0253 |
| PSG(epochs = 400)   | **0.9118 ± 0.0235** | 0.8161 ± 0.0232 |
| **PSG(epochs = 500)**   | **0.9284 ± 0.0047** | **0.8306 ± 0.0134** |
| PSG(epochs = 600)   | **0.9189 ± 0.0298** | **0.8562 ± 0.0181** |
| PSG(epochs = 750)   | 0.9065 ± 0.0391 | **0.8651 ± 0.0156** |
| PSG(epochs = 1000)   | 0.9072 ± 0.0246 | **0.8673 ± 0.0253** |
| PSG(epochs = 1500)   | **0.9125 ± 0.0214** | **0.8681 ± 0.0161** |

PSG achieves **top-1** performance on ogbl-ddi in current OGB Link Property Prediction Leader Board until **Aug 12, 2022** (https://ogb.stanford.edu/docs/leader_linkprop/), which improves the Hits@20 performance by **2.2%** than the state-of-the-art algorithm PLNLP on the test set. 

## Reference
This work is mainly based on GraphSAGE [1], PLNLP [2] and GraphSAGE + Edge Attr [3, 4], which can be found as follows:

[1] W. Hamilton, Z. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” Advances in neural information processing systems, vol. 30, 2017.

[2] Z. Wang, Y. Zhou, L. Hong, Y. Zou, and H. Su, “Pairwise learning for neural link prediction,” arXiv preprint arXiv:2112.02936, 2021.

[3] S. Lu and J. Yang, “Link prediction with structural information,” https://github.com/lustoo/OGB_link_prediction/blob/main/Link%20prediction%20with%20structural%20information.pdf, 2021.

[4] P. Li, Y. Wang, H. Wang, and J. Leskovec, “Distance encoding: Design provably more powerful neural networks for graph representation learning,” Advances in
Neural Information Processing Systems, vol. 33, pp. 4465–4478, 2020.


