# Building a Digital Twin for Network Optimization using Graph Neural Networks
#### Miquel Ferriol-Galmés, José Suárez-Varela, Jordi Paillissé, Bo Wu, Shihan Xiao, Xiangle Cheng, Pere Barlet-Ros and Albert Cabellos-Aparicio

### Abstract
Network modeling is a critical component of Quality of Service (QoS) optimization. Current networks implement Service Level Agreements (SLA) by careful configuration of both routing and queue scheduling policies. However, existing modeling techniques are not able to produce accurate estimates of relevant SLA metrics, such as delay or jitter, in networks with complex QoS-aware queueing policies (e.g., strict priority, Weighted Fair Queueing, Deficit Round Robin). Recently, Graph Neural Networks (GNNs) have become a powerful tool to model networks since they are specifically designed to work with graph-structured data. In this paper, we propose a GNN-based network model able to understand the complex relationship between _(i)_ the queueing policy (scheduling algorithm and queue sizes), _(ii)_ the network topology, _(iii)_ the routing configuration, and _(iv)_ the input traffic matrix. We call our model TwinNet, a _Digital Twin_ that can accurately estimate relevant SLA metrics for network optimization. TwinNet can generalize to its input parameters, operating successfully in topologies, routing, and queueing configurations never seen during training. We evaluate TwinNet over a wide variety of scenarios with synthetic traffic and validate it with real traffic traces. Our results show that TwinNet can provide accurate estimates of end-to-end path delays in 106 unseen real-world topologies, under different queuing configurations with a Mean Absolute Percentage Error (MAPE) of 3.8%, as well as a MAPE of 6.3% error when evaluated with a real testbed. We also showcase the potential of the proposed model for SLA-driven network optimization and what-if analysis.

**If you decide to apply the concepts presented or base on the provided code, please do refer our paper.**

```
@article{ferriol2022building,
  title={Building a digital twin for network optimization using graph neural networks},
  author={Ferriol-Galm{\'e}s, Miquel and Su{\'a}rez-Varela, Jos{\'e} and Pailliss{\'e}, Jordi and Shi, Xiang and Xiao, Shihan and Cheng, Xiangle and Barlet-Ros, Pere and Cabellos-Aparicio, Albert},
  journal={Computer Networks},
  volume={217},
  pages={109329},
  year={2022},
  publisher={Elsevier}
}
```
## Quick Start
### Project structure
The project is divided into two main blocks: [GNN](./GNN) and [Optimizer](./optimizer). Each block has its own 
directory and contains its own files. 

In the [GNN](./GNN) repository we can find three main files:
- `main.py`: contains the code for the training/validation of the model.
- `model.py`: contains the Tensorflow-based implementation of TwinNet.
- `read_dataset.py`: contains the code for reading the dataset and transforming it to the graph that is fed into the GNN.
- `config.ini`: contains the different paths to the dataset and the logs directories, as well as all the configurable hyperparameters of the model.

In the [Optimizer](./optimizer) repository we can find four main files, each one for one of the different optimization experiments:
- `optimize_routing.py`: Routing Optimization experiments.
- `optimize_scheduling.py`: Scheduling Optimization experiments.
- `optimize_scheduling_routing.py`: Routing and Scheduling Optimization experiments.
- `link_failure.py`: Link Failure experiments.
- `network_upgrade.py`: Network Upgrade experiments.

You can find more information about the reproducibility of the experiments inside each one of the directories 
([GNN](./GNN/README.md), [Optimizer](./optimizer/README.md)).

## Main Contributors
#### M. Ferriol-Galmés, J. Suárez-Varela, P. Barlet-Ros, A. Cabellos-Aparicio.

[Barcelona Neural Networking center](https://bnn.upc.edu/), Universitat Politècnica de Catalunya

#### Do you want to contribute to this open-source project? Please, read our guidelines on [How to contribute](CONTRIBUTING.md)

## License
See [LICENSE](LICENSE) for full of the license text.

```
Copyright 2022 Universitat Politècnica de Catalunya

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
