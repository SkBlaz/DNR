import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import dnrlib
import scipy.io

input_matrix = "../datasets/cora.mat"
task_name = input_matrix.split("/")[-1].replace(".mat", ".emb")
mat = scipy.io.loadmat(input_matrix)

adjacency = mat['network']
labels1 = np.argmax(mat['group'], axis=1)

dnr_class = dnrlib.DNR(device="cpu",
                       batch_size=64,
                       algorithm="DNR",
                       num_epoch=100,
                       hidden_size=1,
                       num_pivot_nodes=None,
                       n_components=256)

embedding = dnr_class.fit_transform(adjacency)

task_name = "example_task"
reducer = umap.UMAP(n_neighbors=15, min_dist=0.5)
projection = reducer.fit_transform(embedding)
g = sns.scatterplot(projection[:, 0],
                    projection[:, 1],
                    hue=labels1,
                    palette="Set2",
                    s=60,
                    legend=None)
plt.gca().set_aspect('equal', 'datalim')
plt.axis('off')
plt.savefig(f"../figures/embedding_{task_name}.pdf", dpi=400)
