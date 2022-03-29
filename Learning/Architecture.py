# 2022/03/29~
# Mikhail Hayhoe, mhayhoe@seas.upenn.edu
# Adapted from code by:
# Landon Butler, landonb3@seas.upenn.edu
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu

import torch

torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
import torch.sparse
import alegnn.utils.graphML as gml


class BatchedSparseGraphFilter():
    pass


class SelectionGNN_CliqueLine(nn.Module):
    """
    SelectionGNN: implement the selection GNN architecture, with a clique
                expansion module followed by a line expansion  module

    Initialization:

        SelectionGNN(dimNodeSignals, nFilterTaps, bias, # Graph Filtering
                     nonlinearity, # Nonlinearity
                     nSelectedNodes, poolingFunction, poolingSize, # Pooling
                     dimLayersMLP, # MLP in the end)

        Input:
            /** Graph convolutional layers **/
            dimNodeSignals (list of int): dimension of the signals at each layer
                (i.e. number of features at each node, or size of the vector
                 supported at each node)
            nFilterTaps (list of int): number of filter taps on each layer
                (i.e. nFilterTaps-1 is the extent of neighborhoods that are
                 reached, for example K=2 is info from the 1-hop neighbors)
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nFilterTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nFilterTaps) = L.

            /** Activation function **/
            nonlinearity (torch.nn): module from torch.nn non-linear activations

            /** Pooling **/
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML or in torch.nn):
                summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer

            /** Readout layers **/
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied


        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics.

    Forward call:

        SelectionGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes
            GSOs (list): list of GSOs, each torch.sparse tensors of size
                batchSize x numberNodes x numberNodes
        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x 1

    Other methods:

        .changeGSO(S, nSelectedNodes = [], poolingSize = []): takes as input a
        new graph shift operator S as a tensor of shape
            (dimEdgeFeatures x) numberNodes x numberNodes
        Then, next time the SelectionGNN is run, it will run over the graph
        with GSO S, instead of running over the original GSO S. This is
        particularly useful when training on one graph, and testing on another
        one. The number of selected nodes and the pooling size will not change
        unless specifically consider those as input. Those lists need to have
        the same length as the number of layers. There is no need to define
        both, unless they change.
        >> Obs.: The number of nodes in the GSOs need not be the same, but
            unless we want to risk zero-padding beyond the original number
            of nodes (which just results in disconnected nodes), then we might
            want to update the nSelectedNodes and poolingSize accordingly, if
            the size of the new GSO is different.

        y, yGNN = .splitForward(x): gives the output of the entire GNN y,
        which is of shape batchSize x dimLayersMLP[-1], as well as the output
        of all the GNN layers (i.e. before the MLP layers), yGNN of shape
        batchSize x nSelectedNodes[-1] x dimFeatures[-1]. This can be used to
        isolate the effect of the graph convolutions from the effect of the
        readout layer.
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nFilterTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 nEdgeFeatures):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nFilterTaps) + 1
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nFilterTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nFilterTaps)
        # We will never be using coarsening, for simplicity
        assert not self.coarsening, "Coarsening is not supported for this architecture."
        # Store the values (using the notation in the paper):
        self.L = len(nFilterTaps)  # Number of graph filtering layers
        self.F = dimNodeSignals  # Features
        self.K = nFilterTaps  # Filter taps
        self.E = nEdgeFeatures  # Number of edge feature
        self.N = [GSO.shape[1]] + nSelectedNodes  # Number of nodes
        self.alpha = poolingSize
        self.lineOffset = self.L  # Offset for when to start the line graph layers

        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias  # Boolean
        # Store the rest of the variables
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.dimLayersMLP = dimLayersMLP
        # And now, we're finally ready to create the architecture:
        # \\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        gfl = []  # Graph Filtering Layers for clique expansion
        for l in range(self.L):
            # \\ Graph filtering stage:
            gfl.append(gml.GraphFilter(self.F[l], self.F[l + 1], self.K[l],
                                       self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.

            gfl[3 * l].addGSO(self.S_clique)
            # \\ Nonlinearity
            gfl.append(self.sigma())
            # \\ Pooling
            gfl.append(self.rho(self.N[l], self.N[l + 1], self.alpha[l]))
            # Same as before, this is 3*l+2
            gfl[3 * l + 2].addGSO(self.S_clique)

        for l in range(self.L):
            # \\ Graph filtering stage:
            gfl.append(gml.GraphFilter(self.F[l], self.F[l + 1], self.K[l],
                                       self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.

            gfl[3 * (l + self.lineOffset)].addGSO(self.S_line)
            # \\ Nonlinearity
            gfl.append(self.sigma())
            # \\ Pooling
            gfl.append(self.rho(self.N[l], self.N[l + 1], self.alpha[l]))
            # Same as before, this is 3*l+2
            gfl[3 * (l + self.lineOffset) + 2].addGSO(self.S_line)
        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl)  # Graph Filtering Layers
        # \\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0:  # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias=self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP) - 1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l + 1],
                                    bias=self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.

    def changeGSO(self, GSO_clique, GSO_line, nSelectedNodes=[], poolingSize=[]):

        # We use this to change the GSO, using the same graph filters.

        # Check that the new GSOs have the correct shape
        assert len(GSO_clique.shape) == 2 or len(GSO_clique.shape) == 3
        if len(GSO_clique.shape) == 2:
            assert GSO_clique.shape[0] == GSO_clique.shape[1]
            GSO = GSO_clique.reshape([1, GSO_clique.shape[0], GSO_clique.shape[1]])  # 1 x N x N
        else:
            assert GSO_clique.shape[1] == GSO_clique.shape[2]  # E x N x N

        assert len(GSO_line.shape) == 2 or len(GSO_line.shape) == 3
        if len(GSO_line.shape) == 2:
            assert GSO_line.shape[0] == GSO_line.shape[1]
            GSO = GSO_line.reshape([1, GSO_line.shape[0], GSO_line.shape[1]])  # 1 x M x M
        else:
            assert GSO_line.shape[1] == GSO_line.shape[2]  # E x M x M

        # Get dataType and device of the current GSO, so when we replace it, it
        # is still located in the same type and the same device.
        dataType = self.S.dtype
        if 'device' in dir(self.S):
            device = self.S.device
        else:
            device = None

        # Now, if we don't have coarsening, then we need to reorder the GSO,
        # and since this GSO reordering will affect several parts of the non
        # coarsening algorithm, then we will do it now
        # Reorder the GSO
        self.S, self.order = self.permFunction(GSO)
        # Change data type and device as required
        self.S = changeDataType(self.S, dataType)
        if device is not None:
            self.S = self.S.to(device)

        # Before making decisions, check if there is a new poolingSize list
        if len(poolingSize) > 0:
            # (If it's coarsening, then the pooling size cannot change)
            # Check it has the right length
            assert len(poolingSize) == self.L
            # And update it
            self.alpha = poolingSize

        # Now, check if we have a new list of nodes (this only makes sense
        # if there is no coarsening, because if it is coarsening, the list with
        # the number of nodes to be considered is ignored.)
        if len(nSelectedNodes) > 0:
            # If we do, then we need to change the pooling functions to select
            # less nodes. This would allow to use graphs of different size.
            # Note that the pooling function, there is nothing learnable, so
            # they can easily be re-made, re-initialized.
            # The first thing we need to check, is that the length of the
            # number of nodes is equal to the number of layers (this list
            # indicates the number of nodes selected at the output of each
            # layer)
            assert len(nSelectedNodes) == self.L
            # Then, update the N that we have stored
            self.N = [GSO.shape[1]] + nSelectedNodes
            # And get the new pooling functions
            for l in range(self.L):
                # For each layer, add the pooling function
                self.GFL[3 * l + 2] = self.rho(self.N[l], self.N[l + 1],
                                               self.alpha[l])
                self.GFL[3 * l + 2].addGSO(self.S)
        elif len(nSelectedNodes) == 0:
            # Just update the GSO
            for l in range(self.L):
                self.GFL[3 * l + 2].addGSO(self.S)

        # And update in the LSIGF that is still missing (recall that the
        # ordering for the non-coarsening case has already been done)
        for l in range(self.L):
            self.GFL[3 * l].addGSO(self.S)  # Graph convolutional layer

    def splitForward(self, x):

        # Reorder the nodes from the data
        x = x[:, :, self.order]

        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Let's call the graph filtering layer
        y = self.GFL(x)
        # Flatten the output
        yFlat = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(yFlat), y
        # If self.MLP is a sequential on an empty list it just does nothing.

    def forward(self, x):

        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)

        return output

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S_clique = self.S_clique.to(device)
        self.S_line = self.S_line.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.GFL[3 * l].addGSO(self.S_clique)
            self.GFL[3 * l + 2].addGSO(self.S_clique)
            self.GFL[3 * (l + self.lineOffset)].addGSO(self.S_line)
            self.GFL[3 * (l + self.lineOffset) + 2].addGSO(self.S_line)

