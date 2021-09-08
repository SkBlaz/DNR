# Deep Node Ranking

This is the repository of the DNR paper. Short abstract:


`Network node embedding is an active research subfield of complex network analysis. This paper contributes a novel approach to learning network node embeddings and direct node classification using a node ranking scheme coupled with an autoencoder-based neural network architecture. The main advantages of the proposed Deep Node Ranking (DNR) algorithm are competitive or better classification performance, significantly higher learning speed and lower space requirements when compared to state-of-the-art approaches on 15 real-life node classification benchmarks. Furthermore, it enables exploration of the relationship between symbolic and the derived sub-symbolic node representations, offering insights into the learned node space structure.
To avoid the space complexity bottleneck in a direct node classification setting, DNR computes stationary distributions of personalized random walks from given nodes in mini-batches, scaling seamlessly to larger networks. The scaling laws associated with DNR were also investigated on 1488  synthetic Erd\H{o}s-R\'enyi networks, demonstrating its scalability to tens of millions of links.`


## DNR library
The core algorithm is implemented as a simple-to-use Python library. Simply
```
pip install dnrlib
```
to install the library.

## Example use

A self-contained example is as follows:

```
import dnrlib
import scipy.io

input_matrix = "./datasets/ions.mat"
task_name = input_matrix.split("/")[-1].replace(".mat",".emb")
mat = scipy.io.loadmat(input_matrix)

adjacency = mat['network']

dnr_class = dnrlib.DNR(device = "cpu", batch_size = 64, algorithm = "DNR", num_epoch = 100, hidden_size = 1, num_pivot_nodes = None, n_components = 256)

embedding = dnr_class.fit_transform(adjacency)    

print(embedding.shape)
```

Please inspect the source for any additional export/import methods that might come handy.

## Tests
To check the original mode of operation, please run
```
python -m pytest ./tests
```

## Data
Examples of freely (under a given license) available data are given in `./datasets` folder. For all datasets, please write to us (licensing constraints).


## Hyperparameters


| Hyperparameter                   | Default value | Possible values                                                  |
|----------------------------------|---------------|------------------------------------------------------------------|
| num_pivot_nodes                     | None           | int (None = use all nodes) |
| verbose            | True             | bool  |
| batch_size                        | 64 | int |
| num_epoch                          | 100         | int            |
| learning_rate | 0.01  | float                                     |
| stopping_nn (stopping criterion)                          | 10          | int       |
| algorithm                      | "DNR"          | ['DNR','DNR-symbolic']  |
| damping (damping factor)                | 0.86             | float             |
| epsilon (convergence constraint)		           | 1e-6	       | float |
| scaling_constant (numerical stability)                              | 10         | float  |
| spread_step                              | 20         | int |
| max_steps (ranking)                             | 100000         | int |
| hidden_size (nn)                             | 2         | int |
| dropout                             | 0.6         | float |
| spread_percent                             | 0.3         | float |
| device (Torch)                             | 'cpu'         | ['cuda','cpu'] |
| memoization                             | True         | bool |
| upper_memory_bound_gb                             | 16         | int |
| try_shrink                             | True         | bool |
| n_components                             | 128         | int |


## Other useful methods

`obj.write_output_file(fname` -> creates a standard `.emb` file useful for benchmarking.

## Examples
For additional examples, please consult `./examples`