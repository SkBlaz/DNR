## The core DNR class
import numpy as np
import torch
import scipy.sparse as sp
import operator
import tqdm
from sklearn.preprocessing import normalize
import time
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger(__name__).setLevel(logging.INFO)

try:
    from .class_neural import DNRNet
except:
    from class_neural import DNRNet

np.random.seed(123321)


class DNR:
    def __init__(self,
                 num_pivot_nodes=None,
                 verbose=True,
                 batch_size=64,
                 num_epoch=100,
                 learning_rate=0.01,
                 stopping_nn=10,
                 algorithm="DNR",
                 damping=0.86,
                 epsilon=1e-6,
                 scaling_constant=10,
                 spread_step=20,
                 max_steps=100000,
                 hidden_size=2,
                 spread_percent=0.3,
                 device="cuda",
                 memoization=True,
                 dropout=0.6,
                 upper_memory_bound_gb=16,
                 try_shrink=True,
                 n_components=128):
        """
        Core DNR class. Includes methods for walk sampling and subsequent representation distillation.
        
        :param int num_pivot_nodes: How many pivot nodes to consider for sampling -> None implies whole P matrix
        :param bool verbose: Whether to perform logging
        :param int batch_size: The size of the rank batches used during training.
        :param int num_epoch: The number of epochs (upper bound) for neural network training
        :param float learning_rate: The learning rate hyperparameter
        :param int stopping_nn: The stopping criterion for the neural network
        :param str algorithm: The representation learning algorithm to be considered (DNR-symbolic or DNR)
        :param float damping: The damping factor for PPR
        :param float epsilon: The error bound for the power iteration
        :param int scaling_constant: Scaling constant for numerical stability
        :param int spread_step: Spread step hyperparameter
        :param int max_steps: Maximum number of steps for the PageRank iteration
        :param int hidden_size: The hidden layer size
        :param float spread_percent: The spreading percent hyperparameter
        :param str device: "cuda" or "cpu"
        :param bool memoization: Whether to store the PPR matrix if possible
        :param float dropout: The dropout of the neural network
        :param int upper_memory_bound_gb: Maximum amount of ram in GB allowed
        :param bool try_shrink: Whether to try to shrink the adjacency before the PPR iterations
        :param int n_components: The latent dimension of the output embedding
        """

        self.num_pivot_nodes = num_pivot_nodes
        self.scaling_constant = scaling_constant
        self.upper_memory_bound_gb = upper_memory_bound_gb
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.memoization = memoization
        self.device = device
        self.epsilon = epsilon
        self.try_shrink = try_shrink
        self.spread_step = spread_step
        self.damping = damping
        self.spread_percent = spread_percent
        self.max_steps = max_steps
        self.stopping_nn = stopping_nn
        self.n_components = n_components
        self.verbose = verbose
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.algorithm = algorithm

        if self.verbose:
            logging.info(f"Num. pivot nodes set to: {num_pivot_nodes}")

    def get_global_node_ranking(self, network):
        """
        The method to obtain the global node ranking suitable for subsequent shrinking steps.

        :param scipy.sparse network: An adjacency matrix (sparse) of a network

        """

        start_time = time.monotonic()
        if self.num_pivot_nodes is None:
            self.num_pivot_nodes = network.shape[1]

        if self.num_pivot_nodes < network.shape[1]:
            self.global_pagerank_ = sorted(
                self.sparse_pagerank(network).items(),
                key=operator.itemgetter(1),
                reverse=True)
            self.global_top_nodes_ = [
                x[0] for x in self.global_pagerank_[0:self.num_pivot_nodes]
            ]

        else:
            self.global_top_nodes_ = np.arange(network.shape[1])
        start_time = time.monotonic() - start_time

        if self.verbose:
            logging.info(
                f"Computed global ranking for pivoting. {np.round(start_time, 2)}s"
            )

    def estimate_memory_bound(self, network, dtype=32, out="GB"):
        """
        Memory bound estimation method.

        :param scipy.sparse network: The input network
        :param int dtype: The data type
        :param str out: The output format unit
        :return: Estimate of the space required
        """

        sizes = {v: i for i, v in enumerate("BYTES KB MB GB TB".split())}
        if not self.num_pivot_nodes is None:
            dim2 = self.num_pivot_nodes
        else:
            dim2 = network.shape[1]
        return network.shape[0] * dim2 * dtype / 8 / 1024.**sizes[out]

    def fit_transform(self, network, labelspace=None):
        """
        The sklearn-like fit-transform method.

        :param scipy.sparse network: The input adjacency matrix
        :param obj labelspace: Optional labelspace (for the future versions of the algorithm perhaps)
        """

        fit_time = time.monotonic()
        self.num_nodes = network.shape[0]
        if self.verbose:
            numedge = len(np.nonzero(network)[0])
            logging.info(
                f"Loaded a graph with |N| = {self.num_nodes}; |E| = {numedge}")

        ## Normalize the network just once.
        network = self.stochastic_normalization(network)

        ## Estimate whether memoization will overload the system
        memory_consumption = self.estimate_memory_bound(
            network) * 2  ## network + neural network overhead estimate

        if self.verbose:
            logging.info(
                f"Estimated memory consumption: {np.round(memory_consumption, 2)}GB"
            )

        ## Revert to minibatch P-PRS computation
        if memory_consumption >= self.upper_memory_bound_gb:
            if self.verbose:
                logging.info(
                    f"Estimated memory in GB: {memory_consumption}. Upper memory bound set via upper_memory_bound_gb parameter: {self.upper_memory_bound_gb}. Switching to iterative representation construction - this will be slower, albeit very space efficient."
                )
                self.memoization = False

        ## Identify pivot nodes
        self.get_global_node_ranking(network)

        if self.algorithm == "DNR-symbolic":

            output_representations = np.zeros(
                (network.shape[0], self.num_pivot_nodes))
            for start_node in tqdm.tqdm(list(range(self.num_nodes)),
                                        total=self.num_nodes,
                                        disable=not self.verbose):
                representation = self.pr_kernel(network, start_node)
                output_representations[representation[0]] = representation[1]

            self.embedding = output_representations

            if self.verbose:

                mdensity = np.round(
                    np.count_nonzero(output_representations) /
                    (output_representations.shape[0] *
                     output_representations.shape[1]), 3)
                logging.info(
                    f"Obtained the P-PRS representation matrix with shape {self.embedding.shape} and density of {mdensity}"
                )

        elif self.algorithm == "DNR":

            pivot_nodes = len(self.global_top_nodes_)
            self.model = DNRNet(input_size=pivot_nodes,
                                output_size=pivot_nodes,
                                verbose=True,
                                device=self.device,
                                hidden_size=self.hidden_size,
                                embedding_dim=self.n_components,
                                dropout=self.dropout)

            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.learning_rate)
            self.num_params = sum(p.numel() for p in self.model.parameters())
            self.loss = torch.nn.SmoothL1Loss()
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.9)

            if self.verbose:
                logging.info(
                    f"Initialized architecture with {self.num_params} parameters and {self.hidden_size+1} hidden layers."
                )

            job_batches = np.array_split(list(range(network.shape[0])),
                                         network.shape[0] / self.batch_size)

            min_loss = np.inf
            stopping_crit = 0
            t = tqdm.trange(self.num_epoch,
                            desc='Loss',
                            leave=True,
                            disable=not self.verbose)

            if self.memoization:

                if self.verbose:
                    logging.info(
                        f"Probability matrix memoization. Estimated upper memory bound of {self.upper_memory_bound_gb}GB not exceeded."
                    )

                output_representations = np.zeros(
                    (network.shape[0], self.num_pivot_nodes))

                ## TODO: parallelize this
                for start_node in tqdm.tqdm(list(range(self.num_nodes)),
                                            disable=not self.verbose):
                    representation = self.pr_kernel(network, start_node)
                    output_representations[
                        representation[0]] = representation[1]

                self.prob_rep = output_representations

            self.model.train()

            self.losses = []

            for epoch in range(self.num_epoch):

                epoch_loss = torch.tensor(0).float()
                for batch in job_batches:

                    if self.memoization:

                        ppr_batch = torch.from_numpy(
                            normalize(self.prob_rep[batch],
                                      norm="l2")).float().to(self.device)

                    else:

                        ppr_batch = []

                        for node_id in batch:
                            rep_id, rep = self.pr_kernel(network, node_id)
                            ppr_batch.append(normalize(rep, norm="l2"))
                            ppr_batch = torch.from_numpy(
                                np.array(ppr_batch)).float().to(self.device)

                    ppr_batch = ppr_batch * self.scaling_constant
                    reconstruction = self.model(ppr_batch)

                    loss = self.loss(reconstruction, ppr_batch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    epoch_loss += float(loss)

                t.update()
                self.losses.append(epoch_loss.detach().cpu().numpy().item())
                if self.verbose:

                    t.set_description(
                        f"Epoch: {epoch+1}; loss: {np.round(epoch_loss, 5)}")
                    if epoch_loss >= min_loss:
                        stopping_crit += 1
                    else:
                        min_loss = epoch_loss

                if stopping_crit > self.stopping_nn:
                    if self.verbose:
                        logging.info("Stopping reached, DNRNet converged.")

                    break

            self.embedding = np.zeros((network.shape[0], self.n_components))

            if self.verbose:
                logging.info("Final set of forward passes.")

            self.model.eval()

            for start_node in range(network.shape[0]):

                if self.memoization:
                    ppr_vec = torch.from_numpy(
                        self.prob_rep[start_node]).float(
                        ) * self.scaling_constant
                    representation = self.model.get_representation(
                        ppr_vec).detach().numpy()

                else:
                    ppr_vec = self.pr_kernel(
                        network, start_node) * self.scaling_constant
                    representation = self.model.get_representation(ppr_vec)

                self.embedding[start_node] = representation

        elif self.algorithm == "Random":
            self.embedding = np.random.random(
                (self.num_nodes, self.n_components))

        fit_time = time.monotonic() - fit_time
        self.fitting_time = fit_time

        if self.verbose:
            logging.info(
                f"Obtained final embedding of dimensions: {self.embedding.shape}"
            )
            logging.info(
                f"The process required: {np.round(self.fitting_time, 2)}s")

        return self.embedding

    def write_output_file(self, ofile):
        """
        A method which produces a word2vec-like .emb file.
        
        :param str ofile: The target output file.

        """

        with open(ofile, 'w') as f:
            f.write(
                str(self.num_nodes) + " " + str(self.embedding.shape[1]) +
                "\n")
            for j in range(self.embedding.shape[0]):
                prediction = self.embedding[j]
                outst = str(j) + " " + " ".join(str(x)
                                                for x in prediction) + "\n"
                f.write(outst)
        if self.verbose:
            logging.info(f"Wrote the embedding to: {ofile}")

    def sparse_pagerank(self, M, max_iter=10000, tol=1.0e-6, alpha=0.85):
        """
        The sparse PageRank algorithm (for pivoting)

        :param scipy.sparse M: The input matrix
        :param int max_iter: Maximum number of iterations
        :param float tol: The tolerance criterion
        :param float alpha: The damping factor
        :return dict: PageRanks for individual nodes

        """

        N = M.shape[1]
        nodelist = np.arange(M.shape[1])
        S = np.array(M.sum(axis=1)).flatten()
        S[S != 0] = 1.0 / S[S != 0]
        Q = sp.spdiags(S.T, 0, *M.shape, format='csr')
        M = Q * M
        x = np.repeat(1.0 / N, N)
        p = np.repeat(1.0 / N, N)
        dangling_weights = p
        is_dangling = np.where(S == 0)[0]
        for _ in range(max_iter):
            xlast = x
            x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + \
                (1 - alpha) * p
            err = np.absolute(x - xlast).sum()
            if err < N * tol:
                return dict(zip(nodelist, map(float, x)))

    def stochastic_normalization(self, matrix):
        """
        Stochastic normalization of the adjacency matrix - needed for proper ranking.
        
        :param scipy.sparse matrix: The sparse matrix
        :return scipy.sparse: The normalized matrix

        """

        matrix = matrix.tolil()
        try:
            matrix.setdiag(0)
        except TypeError:
            matrix.setdiag(np.zeros(matrix.shape[0]))
        matrix = matrix.tocsr()
        d = matrix.sum(axis=1).getA1()
        nzs = np.where(d > 0)
        d[nzs] = 1 / d[nzs]
        matrix = (sp.diags(d, 0).tocsc().dot(matrix)).transpose()
        return matrix

    def sparse_page_rank(self,
                         matrix,
                         start_nodes,
                         epsilon=1e-6,
                         max_steps=100000,
                         damping=0.5,
                         spread_step=10,
                         spread_percent=0.3,
                         try_shrink=True):
        """
        The P-PRS algorithm with pivoting

        :param scipy.sparse matrix: The input adjacency matrix
        :param list start_nodes: The list of starting nodes
        :param float epsilon: The Error tolerance of the final estimation
        :param int max_steps: The maximum number of allowed steps
        :param float damping: The daming hyperparameter
        :param int spread_step: The spreading step hyperparameter
        :param float spread_percent: The spreading percent hyperparameter
        :param bool try_shrink: Whether to perform the shrinking step
        :return np.array: A probability distribution w.r.t. start_nodes
        """

        assert (len(start_nodes)) > 0
        size = matrix.shape[0]
        if start_nodes is None:
            start_nodes = range(size)
            nz = size
        else:
            nz = len(start_nodes)
        start_vec = np.zeros((size, 1))
        start_vec[start_nodes] = 1
        start_rank = start_vec / len(start_nodes)
        rank_vec = start_vec / len(start_nodes)
        shrink = False
        which = np.zeros(0)
        if len(self.global_top_nodes_) < matrix.shape[1]:
            which = np.full(matrix.shape[1], False)
            fo_neighbors = np.where(matrix[start_nodes].todense() > 0)[1]

            ## Preserve the ordering
            tmp_top_nodes = np.array(
                [x for x in self.global_top_nodes_ if not x in fo_neighbors])
            ## Add top ranked (unique) to the local neighborhood - start with local neighborhood and complete with global pivots
            tmp_top_nodes = np.concatenate([tmp_top_nodes, fo_neighbors
                                            ])[-self.num_pivot_nodes:]
            which[tmp_top_nodes] = True
            start_rank = start_rank[which]
            rank_vec = rank_vec[which]
            matrix = matrix[:, which][which, :]
            start_vec = start_vec[which]
            size = len(tmp_top_nodes)
        else:
            which = np.zeros(0)
        if try_shrink:
            v = start_vec / len(start_nodes)
            steps = 0
            while nz < size * spread_percent and steps < spread_step:
                steps += 1
                v += matrix.dot(v)
                nz_new = np.count_nonzero(v)
                if nz_new == nz:
                    shrink = True
                    break
                nz = nz_new
            rr = np.arange(matrix.shape[0])
            which = (v[rr] > 0).reshape(size)
            if shrink:
                start_rank = start_rank[which]
                rank_vec = rank_vec[which]
                matrix = matrix[:, which][which, :]
        diff = np.Inf
        steps = 0
        while diff > epsilon and steps < max_steps:  # not converged yet
            steps += 1
            new_rank = matrix.dot(rank_vec)
            rank_sum = np.sum(new_rank)
            if rank_sum < 0.999999999:
                new_rank += start_rank * (1 - rank_sum)
            new_rank = damping * new_rank + (1 - damping) * start_rank
            new_diff = np.linalg.norm(rank_vec - new_rank, 1)
            diff = new_diff
            rank_vec = new_rank
        if try_shrink and shrink:
            ret = np.zeros(size)
            rank_vec = rank_vec.T[0]  ## this works for both python versions
            ret[which] = rank_vec

            if start_nodes[0] < len(ret):
                ret[start_nodes] = 0
            return ret.flatten()
        else:
            if start_nodes[0] < len(rank_vec):
                rank_vec[start_nodes] = 0
            return rank_vec.flatten()

    def pr_kernel(self, adjacency, index_row):
        """
        A helper function for the PageRank iteration.
        
        :param scipy.sparse adjacency: The input adjacency matrix
        :param int index_row: Node ID to be learned for
        :return list: The probability distribution w.r.t. index_row

        """

        pr = self.sparse_page_rank(adjacency, [index_row],
                                   epsilon=self.epsilon,
                                   max_steps=self.max_steps,
                                   damping=self.damping,
                                   spread_step=self.spread_step,
                                   spread_percent=self.spread_percent,
                                   try_shrink=self.try_shrink)

        norm = np.linalg.norm(pr, 2)
        if norm > 0:
            pr = pr / np.linalg.norm(pr, 2)
            return [index_row, pr]
        else:
            return [index_row, np.zeros(len(self.global_top_nodes_))]


if __name__ == "__main__":

    import scipy.io

    imps = ["../multi_class/citeseer.mat"]
    for input_matrix in imps:

        task_name = input_matrix.split("/")[-1].replace(".mat", ".emb")
        mat = scipy.io.loadmat(input_matrix)
        adjacency = mat['network']
        labels = mat['group']
        labels1 = np.argmax(labels, axis=1)

        dnr_class = DNR(device="cpu",
                        batch_size=64,
                        algorithm="DNR",
                        num_epoch=100,
                        hidden_size=1,
                        num_pivot_nodes=None,
                        n_components=256)

        embedding = dnr_class.fit_transform(adjacency)

        print(embedding.shape)
