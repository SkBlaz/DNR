import torch

torch.manual_seed(123321)


class DNRNet(torch.nn.Module):
    """
    The core DNRNet PyTorch class. It includes the implementation of the inverted forward pass idea.
    """
    def __init__(self,
                 input_size,
                 output_size,
                 verbose=False,
                 device="cuda",
                 hidden_size=3,
                 dropout=0.5,
                 embedding_dim=256):
        """

        param int input_size: Input size
        param int output_size: Output size
        param bool verbose: Output info during training?
        param str device: Default 'cuda', can be 'cpu' too.
        param int hidden_size: The number of hidden layers
        param float dropout: The regularization amount
        param int embedding_dim: The embedding dimension

        """

        super(DNRNet, self).__init__()

        ## Containers and parameters
        self.hidden_size = hidden_size
        self.modules_initial = []
        self.modules_representation = []
        self.modules_final = []

        ## Input projection
        self.modules_initial.append(torch.nn.Linear(input_size, embedding_dim))

        ## Initial embedding
        self.modules_representation.append(torch.nn.ELU())
        self.modules_representation.append(torch.nn.Dropout(dropout))

        ## Embedding layers
        for _ in range(self.hidden_size):
            self.modules_representation.append(
                torch.nn.Linear(embedding_dim, embedding_dim))
            self.modules_representation.append(torch.nn.ELU())

        ## Final projection
        self.modules_final.append(torch.nn.Linear(embedding_dim, output_size))

        ## Param space initialization
        self.sequential_initial = torch.nn.Sequential(*self.modules_initial)
        self.sequential_embedding = torch.nn.Sequential(
            *self.modules_representation)
        self.sequential_final = torch.nn.Sequential(*self.modules_final)

        if verbose:
            print(self.sequential_initial)
            print(self.sequential_embedding)
            print(self.sequential_final)

    def forward(self, x):

        ## Initial projection
        out = self.sequential_initial(x)

        ## Symmetric encoder pass
        out1 = self.sequential_embedding(out)
        out2 = self.sequential_embedding[::-1](out)

        ## Expected representation
        out = (out1 + out2) / 2

        out = self.sequential_final(out)
        return out

    def get_representation(self, x):
        """
        An auxilliary method for obtaining the final representation (from all intermediary layers
        """

        out = self.sequential_initial(x)
        tmp_out = []

        ## This could probably be done better with hooks.
        for j in range(1, len(self.sequential_embedding)):
            if j % 2 == 0:
                tmp = self.sequential_embedding[0:j](out)
                tmp_out.append(tmp)

        ## Construct the final representation
        final_rep = tmp_out[0]

        ## Traverse and average the outputs
        for k in range(1, len(tmp_out)):
            final_rep += tmp_out[k]
        out = final_rep / len(tmp_out)

        return out
