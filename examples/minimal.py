import dnrlib
import scipy.io

input_matrix = "../datasets/citeseer.mat"
task_name = input_matrix.split("/")[-1].replace(".mat", ".emb")
mat = scipy.io.loadmat(input_matrix)

adjacency = mat['network']

dnr_class = dnrlib.DNR(device="cpu",
                       batch_size=64,
                       algorithm="DNR",
                       num_epoch=100,
                       hidden_size=1,
                       num_pivot_nodes=None,
                       n_components=256)

embedding = dnr_class.fit_transform(adjacency)

print(embedding.shape)
