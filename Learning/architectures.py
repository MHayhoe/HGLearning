# 2022/03/29~
# Mikhail Hayhoe, mhayhoe@seas.upenn.edu
# Adapted from code by:
# Landon Butler, landonb3@seas.upenn.edu
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu

import torch
import torch_sparse
import alegnn
import torch.nn as nn
# import torch.optim as optim
import torch.sparse
import alegnn.utils.graphML as gml
import sparseGraphML as spgml
import alegnn.utils.graphTools
from alegnn.utils.dataTools import changeDataType
import numpy as np
import math
from opt_einsum import contract as sparse_einsum

torch.set_default_dtype(torch.float64)


class PoolCliqueToLine(nn.Module):
    """
    MaxPoolLocal Creates a pooling layer on graphs by selecting nodes

    Initialization:

        MaxPoolLocal(in_dim, out_dim, number_hops)

        Inputs:
            in_dim (int): number of nodes at the input
            out_dim (int): number of nodes at the output
            number_hops (int): number of hops to pool information

        Output:
            torch.nn.Module for a local max-pooling layer.

        Observation: The selected nodes for the output are always the top ones.

    Add a neighborhood set:

    Add graph shift operator:

        GraphFilter.addGSO(GSO) Before being used, we need to define the GSO
        that will determine the neighborhood that we are going to pool.

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        v = MaxPoolLocal(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x dim_features x in_dim

        Outputs:
            y (torch.tensor): pooled data; shape:
                batch_size x dim_features x out_dim
    """

    def __init__(self, incidenceMatrix, do_sparse):

        super().__init__()
        self.device = incidenceMatrix.device
        '''
        self.do_sparse = do_sparse
        if self.do_sparse:
            self.Bt = incidenceMatrix.T.to_sparse()
            self.Dinv = torch.sparse.sum(self.Bt, 1).to_dense()**(-1)
            self.edge_list = self.makeEdgeList(incidenceMatrix)
        else:
            self.Bt = incidenceMatrix.T
            self.Dinv = torch.sum(self.Bt, 1)**(-1)
        '''
        self.Bt = incidenceMatrix.T
        self.nInputNodes = incidenceMatrix.shape[0]
        self.nOutputNodes = incidenceMatrix.shape[1]
        self.edge_list = self.makeEdgeList(incidenceMatrix)
        self.signal_step = 8

    def forward(self, x):
        # x should be of shape batchSize x dimNodeSignals x nInputNodes
        # B will be of shape nInputNodes x nOutputNodes.
        # From the clique expansion to the line expansion, this will be N x M,
        # i.e., the number of nodes by the number of hyperedges.
        assert x.shape[2] == self.nInputNodes

        # We need to map the node signals from the current graph to those of the
        # next one according to the provided incidence matrix, B.
        # First, let us add a new dimension to x for the hyperedges. We will map
        # the values of all nodes into the columns corresponding to the hyperedges,
        # and then pool appropriately by passing over the dimension of the nodes
        # (i.e., axis=2). Since x is of shape batchSize x dimNodeSignals x nInputNodes,
        # and B is of shape nInputNodes x nOutputNodes, v will have shape
        # batchSize x dimNodeSignals x nInputNodes x nOutputNodes, and by pooling in
        # the third dimension we obtain a signal v of shape
        # batchSize x dimNodeSignals x nOutputNodes.

        # v, _ = torch.max(torch.einsum('bfn,nm->bfnm', x, self.B), dim=2)

        v = torch.zeros((x.shape[0], x.shape[1], self.nOutputNodes), device=self.device)
        # for m in range(self.nOutputNodes):
        #    v[:, :, m], _ = torch.max(x[:, :, self.edge_list[m]], dim=2)
        for b in range(x.shape[0]):
            for k in range(x.shape[1] // self.signal_step):
                v[b, k*self.signal_step : (k+1)*self.signal_step, :], _ = \
                        torch.max(x[b, k*self.signal_step : (k+1)*self.signal_step, None] * self.Bt, dim=2)
        # if self.do_sparse:
        #     v = torch.sparse.sum(x[:, :, None] * self.Bt, 3).to_dense() * self.Dinv
        # else:
        #     v = torch.sum(x[:, :, None] * self.Bt, dim=3) * self.Dinv

        return v

    # Make an edge list from an incidence matrix
    def makeEdgeList(self, B):
        return [torch.where(B[:,m] > 0)[0].to(self.device) for m in range(self.nOutputNodes)]

    def extra_repr(self):
        reprString = "in_dim=%d, out_dim=%d, pooling between GSOs" % (
            self.nInputNodes, self.nOutputNodes)
        return reprString


def changeDataTypeAndDevice(X, dataType, device):
    # Change data type and device as required
    X = changeDataType(X, dataType)
    if device is not None:
        X = X.to(device)
    return X


def getDataTypeAndDevice(X):
    dataType = X.dtype
    if 'device' in dir(X):
        device = X.device
    else:
        device = None
    return dataType, device


class LocalGNNCliqueLine(nn.Module):
    """
    LocalGNN: implement the selection GNN architecture where all operations are
        implemented locally, i.e. by means of neighboring exchanges only. More
        specifically, it has graph convolutional layers, but the readout layer,
        instead of being an MLP for the entire graph signal, it is a linear
        combination of the features at each node.
        >> Obs.: This precludes the use of clustering as a pooling operation,
            since clustering is not local (it changes the given graph).

    Initialization:

        LocalGNN(dimSignals, nFilterTaps, bias, # Graph Filtering
                 nonlinearity, # Nonlinearity
                 nSelectedNodes, poolingFunction, poolingSize, # Pooling
                 dimReadout, # Local readout layer
                 GSO, order = None # Structure)

        Input:
            /** Graph convolutional layers **/
            dimSignals (list of int): dimension of the signals at each layer
                (i.e. number of features at each node, or size of the vector
                 supported at each node)
            nFilterTaps (list of int): number of filter taps on each layer
                (i.e. nFilterTaps-1 is the extent of neighborhoods that are
                 reached, for example K=2 is info from the 1-hop neighbors)
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimSignals) = L+1. Slightly
                different, nFilterTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nFilterTaps) = L.

            /** Activation function **/
            nonlinearity (torch.nn): module from torch.nn non-linear activations

            /** Pooling **/
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer

            /** Readout layers **/
            dimReadout (list of int): number of output hidden units of a
                sequence of fully connected layers applied locally at each node
                (i.e. no exchange of information involved).

            /** Graph structure **/
            GSO (np.array): graph shift operator of choice.
            incidence_matrices (np.array): maps nodes from the current graph to the next.
                In the case of the clique expansion to line expansion, this is the
                incidence matrix of the original hypergraph.
            targets (np.array): hyperedges corresponding to targets. If this is
                provided, it will be used to select only those outputs corresponding
                to the source hyperedges when calling self.forward()
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array

        Output:
            nn.Module with a Local GNN architecture with the above specified
            characteristics.

    Forward call:

        LocalGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimReadout[-1] x nSelectedNodes[-1]

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
        which is of shape batchSize x dimReadout[-1], as well as the output
        of all the GNN layers (i.e. before the readout layers), yGNN of shape
        batchSize x nSelectedNodes[-1] x dimFeatures[-1]. This can be used to
        isolate the effect of the graph convolutions from the effect of the
        readout layer.

        y = .singleNodeForward(x, nodes): outputs the value of the last layer
        at a single node. x is the usual input of shape batchSize x dimFeatures
        x numberNodes. nodes is either a single node (int) or a collection of
        nodes (list or numpy.array) of length batchSize, where for each element
        in the batch, we get the output at the single specified node. The
        output y is of shape batchSize x dimReadout[-1].
    """

    def __init__(self,
                 # Graph filtering
                 dimSignals, nFilterTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimReadout,
                 # Structure
                 GSOs, incidence_matrices, do_sparse=False, targets=None):
        # Initialize parent:
        super().__init__()
        # dimSignals should be a list and of size 1 more than nFilter taps.
        numGSOs = len(GSOs)
        for i in range(numGSOs):
            assert len(dimSignals[i]) == len(nFilterTaps[i]) + 1
        # Number of incidence matrices should be 1 less than the number of GSOs
        assert len(incidence_matrices) == numGSOs - 1
        # Check whether the GSOs have features or not, and whether they are sparse.
        # After that, always handle it as a matrix of dimension N x N x E.
        for i in range(numGSOs):
            GSO = GSOs[i]
            assert len(GSO.shape) == 2 or len(GSO.shape) == 3
            if len(GSO.shape) == 2:
                assert GSO.shape[0] == GSO.shape[1]
                GSOs[i] = torch.unsqueeze(GSO, axis=0)  # 1 x N x N
            else:
                assert GSO.shape[0] == GSO.shape[1]  # E x N x N
            if i < numGSOs - 1:
                B = incidence_matrices[i]
                if B.is_sparse:
                    self.B_is_sparse[i] = True
                    B = B.coalesce()
                assert B.ndim == 2 and B.shape[0] == GSO.shape[1]  # N x M
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        if nSelectedNodes is None:
            nSelectedNodes = [[GSOs[i].shape[1] for _ in range(len(nFilterTaps[i]))] for i in range(numGSOs)]
        else:
            for i in range(numGSOs):
                assert len(nSelectedNodes[i]) == len(nFilterTaps[i])
        # poolingSize also has to be a list of the same size
        for i in range(numGSOs):
            assert len(poolingSize[i]) == len(nFilterTaps[i])
        # Store the values (using the notation in the paper):
        self.L = [len(nTaps) for nTaps in nFilterTaps]  # Number of graph filtering layers
        self.F = [dims for dims in dimSignals]  # Features
        self.K = [nTaps for nTaps in nFilterTaps]  # Filter taps
        self.E = [GSO.shape[0] for GSO in GSOs]  # Number of edge features
        # For the incidence matrices
        self.B = []
        # For the GSOs
        self.S = []
        for i in range(numGSOs):
            self.S.append(GSOs[i])
            if 'torch' not in repr(self.S[i].dtype):
                self.S[i] = torch.tensor(self.S[i])
            # Permute the incidence matrices to match the GSOs
            if i < numGSOs - 1:
                self.B.append(incidence_matrices[i])
                if 'torch' not in repr(self.B[i].dtype):
                    self.B[i] = torch.tensor(self.B[i])

        self.alpha = [poolSizes for poolSizes in poolingSize]
        self.N = [[GSOs[i].shape[1]] + nSelectedNodes[i] for i in range(numGSOs)]  # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias  # Boolean
        # Store the rest of the variables
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.dimReadout = dimReadout
        self.targets = targets
        # And now, we're finally ready to create the architecture:
        # \\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        gfl = []  # Graph Filtering Layers
        offset = 0
        for i in range(numGSOs):
            for l in range(self.L[i]):
                # \\ Graph filtering stage:
                gfl.append(gml.GraphFilter(self.F[i][l], self.F[i][l + 1], self.K[i][l],
                                               self.E[i], self.bias))
                # There is a 3*l below here, because we have three elements per
                # layer: graph filter, nonlinearity and pooling, so after each layer
                # we're actually adding elements to the (sequential) list.
                gfl[3 * l + offset].addGSO(self.S[i])
                # \\ Nonlinearity
                gfl.append(self.sigma())
                # \\ Pooling
                gfl.append(self.rho(self.N[i][l], self.N[i][l + 1], self.alpha[i][l]))
                # Same as before, this is 3*l+2
                gfl[3 * l + 2 + offset].addGSO(self.S[i])
            # Add the pooling layer between GNNs and update offset, if another incidence matrix is provided
            if i < len(self.B):
                gfl.append(PoolCliqueToLine(self.B[i], do_sparse))
                # Update the offset, since we're starting a new GNN
                offset += 3*self.L[i] + 1
        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl)  # Graph Filtering Layers
        # \\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimReadout) > 0:  # Maybe we don't want to readout anything
            # The first layer has to connect whatever was left of the graph
            # filtering stage to create the number of features required by
            # the readout layer
            fc.append(nn.Linear(self.F[-1][-1], dimReadout[0], bias=self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss, or we add a softmax.)
            for l in range(len(dimReadout) - 1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimReadout[l], dimReadout[l + 1],
                                    bias=self.bias))
        # And we're done
        self.Readout = nn.Sequential(*fc)
        # so we finally have the architecture.
        # Lastly, for efficiency we pre-compute some terms for computing the integral Lipschitz constant
        self.construct_IL_terms()

    def changeGSO(self, GSOs, Bs, nSelectedNodes=[], poolingSize=[]):

        # We use this to change the GSO, using the same graph filters.

        # Check that the new GSO has the correct shape
        numGSOs = len(GSOs)
        for i in range(numGSOs):
            GSO = GSOs[i]
            assert len(GSO.shape) == 2 or len(GSO.shape) == 3
            if len(GSO.shape) == 2:
                assert GSO.shape[0] == GSO.shape[1]
                GSOs[i] = torch.unsqueeze(GSO, axis=0)  # 1 x N x N
            else:
                assert GSO.shape[1] == GSO.shape[2]  # E x N x N
            if i < numGSOs - 1:
                B = Bs[i]
                assert B.ndim == 2 and B.shape[0] == GSO.shape[0]  # N x M

        # Loop through all GSOs provided
        for i in range(numGSOs):
            # Get dataType and device of the current GSO, so when we replace it, it
            # is still located in the same type and the same device.
            dataType, device = getDataTypeAndDevice(self.S[i])
            # Reorder the new GSO
            self.S[i], self.order[i] = self.permFunction(GSOs[i])
            # Change data type and device as required
            self.S[i] = changeDataTypeAndDevice(self.S[i], dataType, device)
            if i < numGSOs - 1:
                dataType, device = getDataTypeAndDevice(self.B[i])
                self.B[i] = B[i][self.order, :]
                self.B[i] = changeDataTypeAndDevice(self.B[i], dataType, device)

        # Before making decisions, check if there is a new poolingSize list
        if len(poolingSize) > 0:
            # Check it has the right length
            assert len(poolingSize) == self.L
            # And update it
            self.alpha = poolingSize

        # Now, check if we have a new list of nodes
        if len(nSelectedNodes) > 0:
            # If we do, then we need to change the pooling functions to select
            # fewer nodes. This would allow using graphs of different size.
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
            offset = 0
            for i in range(numGSOs):
                for l in range(self.L[i]):
                    # For each layer, add the pooling function
                    self.GFL[3 * l + 2] = self.rho(self.N[i][l], self.N[i][l + 1],
                                                   self.alpha[i][l])
                offset += 3*self.L[i] + 1

        # And update the GSOs
        offset = 0
        for i in range(numGSOs):
            for l in range(self.L[i]):
                self.GFL[3 * l + offset].addGSO(self.S[i])  # Graph convolutional layer
                self.GFL[3 * l + 2 + offset].addGSO(self.S[i])
            offset += 3 * self.L[i] + 1
        # Lastly, for efficiency we pre-compute some terms for computing the integral Lipschitz constant
        self.construct_IL_terms()

    # Construct terms required to compute the integral Lipschitz constant ahead of time for efficiency
    def construct_IL_terms(self):
        self.IL_terms = []
        for layer in self.GFL:
            if isinstance(layer, gml.GraphFilter):
                lambda_max = torch.max(torch.abs(torch.linalg.eigvalsh(layer.S)))
                self.IL_terms.append(torch.Tensor([0] + [k * lambda_max ** k for k in range(1, layer.K)])
                                        .repeat(layer.F, layer.E, layer.G).reshape(layer.F, layer.E, layer.K, layer.G))

    # Ensure the integral Lipschitz constant constraint is not violated.
    # Otherwise, the loss could be NaN (due to the log barrier penalties)
    def enforce_IL_condition(self, IL_constant):
        # Check if the IL terms have already been created, and compute them if not.
        try:
            self.IL_terms
        except NameError:
            self.construct_IL_terms()

        # Check the integral Lipschitz constants of all filtering layers, and shrink them if
        # the constraint has been violated.
        # We use no grad to make sure this does not impact the gradients to be computed later
        with torch.no_grad():
            layer_counter = 0
            for layer in self.GFL:
                if isinstance(layer, gml.GraphFilter):
                    C_vals = torch.abs(torch.sum(layer.weight * self.IL_terms[layer_counter], dim=2))
                    C_vals = C_vals.unsqueeze(dim=2).repeat_interleave(layer.K, dim=2)

                    # Get the indices of any filter coefficients that violate the condition
                    violated_inds = C_vals >= IL_constant

                    # Shrink these filter coefficients so that the integral Lipschitz constant will be at most 90% of
                    # the constrained value
                    layer.weight[violated_inds] = layer.weight[violated_inds] / C_vals[violated_inds] * IL_constant * 0.9
                    layer_counter += 1

    # Find the largest (or the mean) integral Lipschitz (IL) constant for all IL graph filtering layers.
    def compute_IL_constant(self, return_all=False):
        try:
            self.IL_terms
        except NameError:
            self.construct_IL_terms()

        C_tensor = torch.Tensor().to(self.IL_terms[0].device)
        layer_counter = 0
        for layer in self.GFL:
            if isinstance(layer, gml.GraphFilter):
                C_vals = torch.abs(torch.sum(layer.weight * self.IL_terms[layer_counter], dim=2))
                C_tensor = torch.cat((C_tensor, C_vals.flatten()))
                del C_vals
                layer_counter += 1
        if return_all:
            return C_tensor
        else:
            return torch.max(C_tensor)

    def splitForward(self, x):

        # Now we compute the forward call
        assert x.ndim == 3
        assert x.shape[1] == self.F[0][0]
        assert x.shape[2] == self.N[0][0]

        # Convert to dense, if required
        if x.is_sparse:
            x = x.to_dense()
        # Let's call the graph filtering layer
        yGFL = self.GFL(x)
        # Change the order, for the readout
        y = yGFL.permute(0, 2, 1)  # B x N[-1] x F[-1]
        # And, feed it into the Readout layer
        y = self.Readout(y)  # B x N[-1] x dimReadout[-1]
        # Reshape and return
        # return y.permute(0, 2, 1), yGFL
        if self.targets is not None:
            if self.targets.ndim == 1:
                return y[:, self.targets, :], yGFL
            elif self.targets.ndim == 2:
                return y[self.targets[:,0], self.targets[:,1], :], yGFL
            else:
                raise ValueError('Targets must be 1D or 2D array/tensor')
        else:
            return y, yGFL
        # IF PERMUTED: B x dimReadout[-1] x N[-1], B x dimFeatures[-1] x N[-1]

    # Splits samples into batches and runs forward on all of them
    def forwardBatch(self, data, samplesType, batchSize=32):
        nSamples = data.indices[samplesType].shape[0]
        device = self.S[0].device
        output = torch.tensor([], device=device)

        for b in range(nSamples // batchSize):
            thisBatchIndices = list(range(b * batchSize, (b + 1) * batchSize))
            xBatch, _ = data.getSamples(samplesType, thisBatchIndices)
            self.targets = data.getTargets(samplesType, thisBatchIndices)
            yBatch = self.forward(xBatch.to(device))
            output = torch.cat((output, yBatch))

        if nSamples % batchSize > 0:
            thisBatchIndices = list(range((b + 1) * batchSize, nSamples))
            xBatch, _ = data.getSamples(samplesType, thisBatchIndices)
            self.targets = data.getTargets(samplesType, thisBatchIndices)
            yBatch = self.forward(xBatch.to(device))
            output = torch.cat((output, yBatch))
        self.targets = None
        return output

    def forward(self, x):

        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward function that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)

        return output

    # TODO: This is NOT updated and should not be used!!!
    def singleNodeForward(self, x, nodes):

        # x is of shape B x F[0] x N[-1]
        batchSize = x.shape[0]
        # nodes is either an int, or a list/np.array of ints of size B
        assert type(nodes) is int \
               or type(nodes) is list \
               or type(nodes) is np.ndarray

        # Let us start by building the selection matrix
        # This selection matrix has to be a matrix of shape
        #   B x N[-1] x 1
        # so that when multiplying with the output of the forward, we get a
        #   B x dimRedout[-1] x 1
        # and we just squeeze the last dimension

        # TODO: The big question here is if multiplying by a matrix is faster
        # than doing torch.index_select

        # Let's always work with numpy arrays to make it easier.
        if type(nodes) is int:
            # Change the node number to accommodate the new order
            nodes = self.order.index(nodes)
            # If it's int, make it a list and an array
            nodes = np.array([nodes], dtype=np.int)
            # And repeat for the number of batches
            nodes = np.tile(nodes, batchSize)
        if type(nodes) is list:
            newNodes = [self.order.index(n) for n in nodes]
            nodes = np.array(newNodes, dtype=np.int)
        elif type(nodes) is np.ndarray:
            newNodes = np.array([np.where(np.array(self.order) == n)[0][0] \
                                 for n in nodes])
            nodes = newNodes.astype(np.int)
        # Now, nodes is an np.int np.ndarray with shape batchSize

        # Build the selection matrix
        selectionMatrix = np.zeros([batchSize, self.N[-1], 1])
        selectionMatrix[np.arange(batchSize), nodes, 0] = 1.
        # And convert it to a tensor
        selectionMatrix = torch.tensor(selectionMatrix,
                                       dtype=x.dtype,
                                       device=x.device)

        # Now compute the output
        y = self.forward(x)
        # This output is of size B x dimReadout[-1] x N[-1]

        # Multiply the output
        y = torch.matmul(y, selectionMatrix)
        #   B x dimReadout[-1] x 1

        # Squeeze the last dimension and return
        return y.squeeze(2)

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the terms for the integral Lipschitz computations
        self.IL_terms = [s.to(device) for s in self.IL_terms]
        # Move the GSO
        offset = 0
        for i in range(len(self.S)):
            self.S[i] = self.S[i].to(device)
            # And all the other variables derived from it.
            for l in range(self.L[i]):
                self.GFL[3 * l + offset].addGSO(self.S[i])
                self.GFL[3 * l + 2 + offset].addGSO(self.S[i])
            offset += 3 * self.L[i] + 1


class LocalGNNHGLap(LocalGNNCliqueLine):
    def __init__(self,
                 # Graph filtering
                 dimSignals, nFilterTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimReadout,
                 # Structure
                 GSOs, incidence_matrices, targets=None, order=None):
        # Initialize parent:
        super().__init__()
        # dimSignals should be a list and of size 1 more than nFilter taps.
        numGSOs = len(GSOs)
        for i in range(numGSOs):
            assert len(dimSignals[i]) == len(nFilterTaps[i]) + 1
        # Check whether the GSOs have features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        for i in range(numGSOs):
            GSO = GSOs[i]
            assert len(GSO.shape) == 2 or len(GSO.shape) == 3
            if len(GSO.shape) == 2:
                assert GSO.shape[0] == GSO.shape[1]
                GSOs[i] = torch.unsqueeze(GSO, axis=0)  # 1 x N x N
            else:
                assert GSO.shape[1] == GSO.shape[2]  # E x N x N
            if i < numGSOs - 1:
                B = incidence_matrices[i]
                assert B.ndim == 2 and B.shape[0] == GSO.shape[0]  # N x M
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        if nSelectedNodes is None:
            nSelectedNodes = [[GSOs[i].shape[1] for _ in range(len(nFilterTaps[i]))] for i in range(numGSOs)]
        else:
            for i in range(numGSOs):
                assert len(nSelectedNodes[i]) == len(nFilterTaps[i])
        # poolingSize also has to be a list of the same size
        for i in range(numGSOs):
            assert len(poolingSize[i]) == len(nFilterTaps[i])
        # Store the values (using the notation in the paper):
        self.L = [len(nTaps) for nTaps in nFilterTaps]  # Number of graph filtering layers
        self.F = [dims for dims in dimSignals]  # Features
        self.K = [nTaps for nTaps in nFilterTaps]  # Filter taps
        self.E = [GSO.shape[0] for GSO in GSOs]  # Number of edge features
        # For the incidence matrices
        self.B = []
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = alegnn.utils.graphTools.permIdentity
            # This is overridden if coarsening is selected, since the ordering
            # function is native to that pooling method.
        self.S = []
        self.order = []
        for i in range(numGSOs):
            newS, newOrder = self.permFunction(GSOs[i])
            self.S.append(newS)
            self.order.append(newOrder)
            if 'torch' not in repr(self.S[i].dtype):
                self.S[i] = torch.tensor(self.S[i])
            # Permute the incidence matrices to match the GSOs
            self.B.append(incidence_matrices[i][newOrder, :])
            if 'torch' not in repr(self.B[i].dtype):
                self.B[i] = torch.tensor(self.B[i])

        self.alpha = [poolSizes for poolSizes in poolingSize]
        self.N = [[GSOs[i].shape[1]] + nSelectedNodes[i] for i in range(numGSOs)]  # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias  # Boolean
        # Store the rest of the variables
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.dimReadout = dimReadout
        self.targets = targets
        # And now, we're finally ready to create the architecture:
        # \\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        gfl = []  # Graph Filtering Layers
        offset = 0
        for i in range(numGSOs):
            for l in range(self.L[i]):
                # \\ Graph filtering stage:
                gfl.append(gml.GraphFilter(self.F[i][l], self.F[i][l + 1], self.K[i][l],
                                           self.E[i], self.bias))
                # There is a 3*l below here, because we have three elements per
                # layer: graph filter, nonlinearity and pooling, so after each layer
                # we're actually adding elements to the (sequential) list.
                gfl[3 * l + offset].addGSO(self.S[i])
                # \\ Nonlinearity
                gfl.append(self.sigma())
                # \\ Pooling
                gfl.append(self.rho(self.N[i][l], self.N[i][l + 1], self.alpha[i][l]))
                # Same as before, this is 3*l+2
                gfl[3 * l + 2 + offset].addGSO(self.S[i])
            # Add the pooling layer between GNNs and update offset, if another incidence matrix is provided
            if i < len(self.B):
                gfl.append(PoolCliqueToLine(self.B[i]))
                # Update the offset, since we're starting a new GNN
                offset += 3*self.L[i] + 1
        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl)  # Graph Filtering Layers
        # \\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimReadout) > 0:  # Maybe we don't want to readout anything
            # The first layer has to connect whatever was left of the graph
            # filtering stage to create the number of features required by
            # the readout layer
            fc.append(nn.Linear(self.F[-1][-1], dimReadout[0], bias=self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss, or we add a softmax.)
            for l in range(len(dimReadout) - 1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimReadout[l], dimReadout[l + 1],
                                    bias=self.bias))
        # And we're done
        self.Readout = nn.Sequential(*fc)
        # so we finally have the architecture.
        # Lastly, for efficiency we pre-compute some terms for computing the integral Lipschitz constant
        self.construct_IL_terms()

    '''
    def changeGSO(self, GSOs, Bs, nSelectedNodes=[], poolingSize=[]):
        # We use this to change the GSO, using the same graph filters.
        # Check that the new GSO has the correct shape
        numGSOs = len(GSOs)
        for i in range(numGSOs):
            GSO = GSOs[i]
            assert len(GSO.shape) == 2 or len(GSO.shape) == 3
            if len(GSO.shape) == 2:
                assert GSO.shape[0] == GSO.shape[1]
                GSOs[i] = torch.unsqueeze(GSO, axis=0)  # 1 x N x N
            else:
                assert GSO.shape[1] == GSO.shape[2]  # E x N x N
            if i < numGSOs - 1:
                B = Bs[i]
                assert B.ndim == 2 and B.shape[0] == GSO.shape[0]  # N x M

        # Loop through all GSOs provided
        for i in range(numGSOs):
            # Get dataType and device of the current GSO, so when we replace it, it
            # is still located in the same type and the same device.
            dataType, device = getDataTypeAndDevice(self.S[i])
            # Reorder the new GSO
            self.S[i], self.order[i] = self.permFunction(GSOs[i])
            # Change data type and device as required
            self.S[i] = changeDataTypeAndDevice(self.S[i], dataType, device)
            if i < numGSOs - 1:
                dataType, device = getDataTypeAndDevice(self.B[i])
                self.B[i] = Bs[i][self.order, :]
                self.B[i] = changeDataTypeAndDevice(self.B[i], dataType, device)

        # Before making decisions, check if there is a new poolingSize list
        if len(poolingSize) > 0:
            # Check it has the right length
            assert len(poolingSize) == self.L
            # And update it
            self.alpha = poolingSize

        # Now, check if we have a new list of nodes
        if len(nSelectedNodes) > 0:
            # If we do, then we need to change the pooling functions to select
            # fewer nodes. This would allow using graphs of different size.
            # Note that the pooling function, there is nothing learnable, so
            # they can easily be re-made, re-initialized.
            # The first thing we need to check, is that the length of the
            # number of nodes is equal to the number of layers (this list
            # indicates the number of nodes selected at the output of each
            # layer)
            assert len(nSelectedNodes) == self.L
            # Then, update the N that we have stored
            self.N = [GSOs.shape[1]] + nSelectedNodes
            # And get the new pooling functions
            offset = 0
            for i in range(numGSOs):
                for l in range(self.L[i]):
                    # For each layer, add the pooling function
                    self.GFL[3 * l + 2] = self.rho(self.N[i][l], self.N[i][l + 1],
                                                   self.alpha[i][l])
                offset += 3*self.L[i] + 1

        # And update the GSOs
        offset = 0
        for i in range(numGSOs):
            for l in range(self.L[i]):
                self.GFL[3 * l + offset].addGSO(self.S[i])  # Graph convolutional layer
                self.GFL[3 * l + 2 + offset].addGSO(self.S[i])
            offset += 3 * self.L[i] + 1
        # Lastly, for efficiency we pre-compute some terms for computing the integral Lipschitz constant
        self.construct_IL_terms()

    # Construct terms required to compute the integral Lipschitz constant ahead of time for efficiency
    def construct_IL_terms(self):
        self.IL_terms = []
        for layer in self.GFL:
            if isinstance(layer, gml.GraphFilter):
                lambda_max = torch.max(torch.abs(torch.linalg.eigvalsh(layer.S)))
                self.IL_terms.append( torch.Tensor([0] + [k * lambda_max ** k for k in range(1, layer.K)]) \
                                 .repeat(layer.F, layer.E, layer.G).reshape(layer.F, layer.E, layer.K, layer.G) )

    # Ensure the integral Lipschitz constant constraint is not violated.
    # Otherwise, the loss could be NaN (due to the log barrier penalties)
    def enforce_IL_condition(self, IL_constant):
        # Check if the IL terms have already been created, and compute them if not.
        try:
            self.IL_terms
        except NameError:
            self.construct_IL_terms()

        # Check the integral Lipschitz constants of all filtering layers, and shrink them if
        # the constraint has been violated.
        # We use no grad to make sure this does not impact the gradients to be computed later
        with torch.no_grad():
            layer_counter = 0
            for layer in self.GFL:
                if isinstance(layer, gml.GraphFilter):
                    C_vals = torch.abs(torch.sum(layer.weight * self.IL_terms[layer_counter], dim=2))
                    C_vals = C_vals.unsqueeze(dim=2).repeat_interleave(layer.K, dim=2)

                    # Get the indices of any filter coefficients that violate the condition
                    violated_inds = C_vals >= IL_constant

                    # Shrink these filter coefficients so that the integral Lipschitz constant will be at most 90% of
                    # the constrained value
                    layer.weight[violated_inds] = layer.weight[violated_inds] / C_vals[violated_inds] * IL_constant * 0.9
                    layer_counter += 1

    # Find the largest (or the mean) integral Lipschitz (IL) constant for all IL graph filtering layers.
    def compute_IL_constant(self, return_all=False):
        try:
            self.IL_terms
        except NameError:
            self.construct_IL_terms()

        C_tensor = torch.Tensor().to(self.IL_terms[0].device)
        layer_counter = 0
        for layer in self.GFL:
            if isinstance(layer, gml.GraphFilter):
                C_vals = torch.abs(torch.sum(layer.weight * self.IL_terms[layer_counter], dim=2))
                C_tensor = torch.cat((C_tensor, C_vals.flatten()))
                layer_counter += 1
        if return_all:
            return C_tensor
        else:
            return torch.max(C_tensor)

    def splitForward(self, x):

        # Now we compute the forward call
        assert x.ndim == 3
        assert x.shape[1] == self.F[0][0]
        assert x.shape[2] == self.N[0][0]
        # Reorder
        x = x[:, :, self.order[0]]  # B x F x N
        # Let's call the graph filtering layer
        yGFL = self.GFL(x)
        # Change the order, for the readout
        y = yGFL.permute(0, 2, 1)  # B x N[-1] x F[-1]
        # And, feed it into the Readout layer
        y = self.Readout(y)  # B x N[-1] x dimReadout[-1]
        # Reshape and return
        # return y.permute(0, 2, 1), yGFL
        if self.targets is not None:
            return y[:, self.targets, :], yGFL
        else:
            return y, yGFL
        # B x dimReadout[-1] x N[-1], B x dimFeatures[-1] x N[-1]

    def forward(self, x):

        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward function that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)

        return output

    # TODO: This is NOT updated and should not be used!!!
    def singleNodeForward(self, x, nodes):

        # x is of shape B x F[0] x N[-1]
        batchSize = x.shape[0]
        # nodes is either an int, or a list/np.array of ints of size B
        assert type(nodes) is int \
               or type(nodes) is list \
               or type(nodes) is np.ndarray

        # Let us start by building the selection matrix
        # This selection matrix has to be a matrix of shape
        #   B x N[-1] x 1
        # so that when multiplying with the output of the forward, we get a
        #   B x dimRedout[-1] x 1
        # and we just squeeze the last dimension

        # TODO: The big question here is if multiplying by a matrix is faster
        # than doing torch.index_select

        # Let's always work with numpy arrays to make it easier.
        if type(nodes) is int:
            # Change the node number to accommodate the new order
            nodes = self.order.index(nodes)
            # If it's int, make it a list and an array
            nodes = np.array([nodes], dtype=np.int)
            # And repeat for the number of batches
            nodes = np.tile(nodes, batchSize)
        if type(nodes) is list:
            newNodes = [self.order.index(n) for n in nodes]
            nodes = np.array(newNodes, dtype=np.int)
        elif type(nodes) is np.ndarray:
            newNodes = np.array([np.where(np.array(self.order) == n)[0][0] \
                                 for n in nodes])
            nodes = newNodes.astype(np.int)
        # Now, nodes is an np.int np.ndarray with shape batchSize

        # Build the selection matrix
        selectionMatrix = np.zeros([batchSize, self.N[-1], 1])
        selectionMatrix[np.arange(batchSize), nodes, 0] = 1.
        # And convert it to a tensor
        selectionMatrix = torch.tensor(selectionMatrix,
                                       dtype=x.dtype,
                                       device=x.device)

        # Now compute the output
        y = self.forward(x)
        # This output is of size B x dimReadout[-1] x N[-1]

        # Multiply the output
        y = torch.matmul(y, selectionMatrix)
        #   B x dimReadout[-1] x 1

        # Squeeze the last dimension and return
        return y.squeeze(2)

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the terms for the integral Lipschitz computations
        self.IL_terms = [s.to(device) for s in self.IL_terms]
        # Move the GSO
        offset = 0
        for i in range(len(self.S)):
            self.S[i] = self.S[i].to(device)
            # And all the other variables derived from it.
            for l in range(self.L[i]):
                self.GFL[3 * l + offset].addGSO(self.S[i])
                self.GFL[3 * l + 2 + offset].addGSO(self.S[i])
            offset += 3 * self.L[i] + 1
    '''

class LocalGNNClique(nn.Module):
    def __init__(self,
                 # Graph filtering
                 dimSignals, nFilterTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimReadout,
                 # Structure
                 GSOs, incidence_matrices, targets=None, order=None):
        # Initialize parent:
        super().__init__()
        # dimSignals should be a list and of size 1 more than nFilter taps.
        numGSOs = len(GSOs)
        for i in range(numGSOs):
            assert len(dimSignals[i]) == len(nFilterTaps[i]) + 1
        # Check whether the GSOs have features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        for i in range(numGSOs):
            GSO = GSOs[i]
            assert len(GSO.shape) == 2 or len(GSO.shape) == 3
            if len(GSO.shape) == 2:
                assert GSO.shape[0] == GSO.shape[1]
                GSOs[i] = torch.unsqueeze(GSO, axis=0)  # 1 x N x N
            else:
                assert GSO.shape[1] == GSO.shape[2]  # E x N x N
            if i < numGSOs - 1:
                B = incidence_matrices[i]
                assert B.ndim == 2 and B.shape[0] == GSO.shape[0]  # N x M
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        if nSelectedNodes is None:
            nSelectedNodes = [[GSOs[i].shape[1] for _ in range(len(nFilterTaps[i]))] for i in range(numGSOs)]
        else:
            for i in range(numGSOs):
                assert len(nSelectedNodes[i]) == len(nFilterTaps[i])
        # poolingSize also has to be a list of the same size
        for i in range(numGSOs):
            assert len(poolingSize[i]) == len(nFilterTaps[i])
        # Store the values (using the notation in the paper):
        self.L = [len(nTaps) for nTaps in nFilterTaps]  # Number of graph filtering layers
        self.F = [dims for dims in dimSignals]  # Features
        self.K = [nTaps for nTaps in nFilterTaps]  # Filter taps
        self.E = [GSO.shape[0] for GSO in GSOs]  # Number of edge features
        # For the incidence matrices
        self.B = []
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = alegnn.utils.graphTools.permIdentity
            # This is overridden if coarsening is selected, since the ordering
            # function is native to that pooling method.
        self.S = []
        self.order = []
        for i in range(numGSOs):
            newS, newOrder = self.permFunction(GSOs[i])
            self.S.append(newS)
            self.order.append(newOrder)
            if 'torch' not in repr(self.S[i].dtype):
                self.S[i] = torch.tensor(self.S[i])
            # Permute the incidence matrices to match the GSOs
            self.B.append(incidence_matrices[i][newOrder, :])
            if 'torch' not in repr(self.B[i].dtype):
                self.B[i] = torch.tensor(self.B[i])

        self.alpha = [poolSizes for poolSizes in poolingSize]
        self.N = [[GSOs[i].shape[1]] + nSelectedNodes[i] for i in range(numGSOs)]  # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias  # Boolean
        # Store the rest of the variables
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.dimReadout = dimReadout
        self.targets = targets
        # And now, we're finally ready to create the architecture:
        # \\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        gfl = []  # Graph Filtering Layers
        offset = 0
        for i in range(numGSOs):
            for l in range(self.L[i]):
                # \\ Graph filtering stage:
                gfl.append(gml.GraphFilter(self.F[i][l], self.F[i][l + 1], self.K[i][l],
                                            self.E[i], self.bias))
                # There is a 3*l below here, because we have three elements per
                # layer: graph filter, nonlinearity and pooling, so after each layer
                # we're actually adding elements to the (sequential) list.
                gfl[3 * l + offset].addGSO(self.S[i])
                # \\ Nonlinearity
                gfl.append(self.sigma())
                # \\ Pooling
                gfl.append(self.rho(self.N[i][l], self.N[i][l + 1], self.alpha[i][l]))
                # Same as before, this is 3*l+2
                gfl[3 * l + 2 + offset].addGSO(self.S[i])
            # Add the pooling layer between GNNs and update offset, if another incidence matrix is provided
            if i < len(self.B):
                gfl.append(PoolCliqueToLine(self.B[i]))
                # Update the offset, since we're starting a new GNN
                offset += 3*self.L[i] + 1
        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl)  # Graph Filtering Layers
        # \\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimReadout) > 0:  # Maybe we don't want to readout anything
            # The first layer has to connect whatever was left of the graph
            # filtering stage to create the number of features required by
            # the readout layer
            fc.append(nn.Linear(self.F[-1][-1], dimReadout[0], bias=self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss, or we add a softmax.)
            for l in range(len(dimReadout) - 1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimReadout[l], dimReadout[l + 1],
                                    bias=self.bias))
        # And we're done
        self.Readout = nn.Sequential(*fc)
        # so we finally have the architecture.
        # Lastly, for efficiency we pre-compute some terms for computing the integral Lipschitz constant
        self.construct_IL_terms()

    def changeGSO(self, GSOs, Bs, nSelectedNodes=[], poolingSize=[]):

        # We use this to change the GSO, using the same graph filters.

        # Check that the new GSO has the correct shape
        numGSOs = len(GSOs)
        for i in range(numGSOs):
            GSO = GSOs[i]
            assert len(GSO.shape) == 2 or len(GSO.shape) == 3
            if len(GSO.shape) == 2:
                assert GSO.shape[0] == GSO.shape[1]
                GSOs[i] = torch.unsqueeze(GSO, axis=0)  # 1 x N x N
            else:
                assert GSO.shape[1] == GSO.shape[2]  # E x N x N
            if i < numGSOs - 1:
                B = Bs[i]
                assert B.ndim == 2 and B.shape[0] == GSO.shape[0]  # N x M

        # Loop through all GSOs provided
        for i in range(numGSOs):
            # Get dataType and device of the current GSO, so when we replace it, it
            # is still located in the same type and the same device.
            dataType, device = getDataTypeAndDevice(self.S[i])
            # Reorder the new GSO
            self.S[i], self.order[i] = self.permFunction(GSOs[i])
            # Change data type and device as required
            self.S[i] = changeDataTypeAndDevice(self.S[i], dataType, device)
            if i < numGSOs - 1:
                dataType, device = getDataTypeAndDevice(self.B[i])
                self.B[i] = B[i][self.order, :]
                self.B[i] = changeDataTypeAndDevice(self.B[i], dataType, device)

        # Before making decisions, check if there is a new poolingSize list
        if len(poolingSize) > 0:
            # Check it has the right length
            assert len(poolingSize) == self.L
            # And update it
            self.alpha = poolingSize

        # Now, check if we have a new list of nodes
        if len(nSelectedNodes) > 0:
            # If we do, then we need to change the pooling functions to select
            # fewer nodes. This would allow using graphs of different size.
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
            offset = 0
            for i in range(numGSOs):
                for l in range(self.L[i]):
                    # For each layer, add the pooling function
                    self.GFL[3 * l + 2] = self.rho(self.N[i][l], self.N[i][l + 1],
                                                   self.alpha[i][l])
                offset += 3*self.L[i] + 1

        # And update the GSOs
        offset = 0
        for i in range(numGSOs):
            for l in range(self.L[i]):
                self.GFL[3 * l + offset].addGSO(self.S[i])  # Graph convolutional layer
                self.GFL[3 * l + 2 + offset].addGSO(self.S[i])
            offset += 3 * self.L[i] + 1
        # Lastly, for efficiency we pre-compute some terms for computing the integral Lipschitz constant
        self.construct_IL_terms()

    # Construct terms required to compute the integral Lipschitz constant ahead of time for efficiency
    def construct_IL_terms(self):
        self.IL_terms = []
        for layer in self.GFL:
            if isinstance(layer, gml.GraphFilter):
                lambda_max = torch.max(torch.abs(torch.linalg.eigvalsh(layer.S)))
                self.IL_terms.append( torch.Tensor([0] + [k * lambda_max ** k for k in range(1, layer.K)]) \
                                 .repeat(layer.F, layer.E, layer.G).reshape(layer.F, layer.E, layer.K, layer.G) )

    # Ensure the integral Lipschitz constant constraint is not violated.
    # Otherwise, the loss could be NaN (due to the log barrier penalties)
    def enforce_IL_condition(self, IL_constant):
        # Check if the IL terms have already been created, and compute them if not.
        try:
            self.IL_terms
        except NameError:
            self.construct_IL_terms()

        # Check the integral Lipschitz constants of all filtering layers, and shrink them if
        # the constraint has been violated.
        # We use no grad to make sure this does not impact the gradients to be computed later
        with torch.no_grad():
            layer_counter = 0
            for layer in self.GFL:
                if isinstance(layer, gml.GraphFilter):
                    C_vals = torch.abs(torch.sum(layer.weight * self.IL_terms[layer_counter], dim=2))
                    C_vals = C_vals.unsqueeze(dim=2).repeat_interleave(layer.K, dim=2)

                    # Get the indices of any filter coefficients that violate the condition
                    violated_inds = C_vals >= IL_constant

                    # Shrink these filter coefficients so that the integral Lipschitz constant will be at most 90% of
                    # the constrained value
                    layer.weight[violated_inds] = layer.weight[violated_inds] / C_vals[violated_inds] * IL_constant * 0.9
                    layer_counter += 1

    # Find the largest (or the mean) integral Lipschitz (IL) constant for all IL graph filtering layers.
    def compute_IL_constant(self, return_all=False):
        try:
            self.IL_terms
        except NameError:
            self.construct_IL_terms()

        C_tensor = torch.Tensor().to(self.IL_terms[0].device)
        layer_counter = 0
        for layer in self.GFL:
            if isinstance(layer, gml.GraphFilter):
                C_vals = torch.abs(torch.sum(layer.weight * self.IL_terms[layer_counter], dim=2))
                C_tensor = torch.cat((C_tensor, C_vals.flatten()))
                layer_counter += 1
        if return_all:
            return C_tensor
        else:
            return torch.max(C_tensor)

    def splitForward(self, x):

        # Now we compute the forward call
        assert x.ndim == 3
        assert x.shape[1] == self.F[0][0]
        assert x.shape[2] == self.N[0][0]
        # Reorder
        x = x[:, :, self.order[0]]  # B x F x N
        # Let's call the graph filtering layer
        yGFL = self.GFL(x)
        # Change the order, for the readout
        y = yGFL.permute(0, 2, 1)  # B x N[-1] x F[-1]
        # And, feed it into the Readout layer
        y = self.Readout(y)  # B x N[-1] x dimReadout[-1]
        # Reshape and return
        # return y.permute(0, 2, 1), yGFL
        if self.targets is not None:
            return y[:, self.targets, :], yGFL
        else:
            return y, yGFL
        # B x dimReadout[-1] x N[-1], B x dimFeatures[-1] x N[-1]

    def forward(self, x):

        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward function that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)

        return output

    # TODO: This is NOT updated and should not be used!!!
    def singleNodeForward(self, x, nodes):

        # x is of shape B x F[0] x N[-1]
        batchSize = x.shape[0]
        # nodes is either an int, or a list/np.array of ints of size B
        assert type(nodes) is int \
               or type(nodes) is list \
               or type(nodes) is np.ndarray

        # Let us start by building the selection matrix
        # This selection matrix has to be a matrix of shape
        #   B x N[-1] x 1
        # so that when multiplying with the output of the forward, we get a
        #   B x dimRedout[-1] x 1
        # and we just squeeze the last dimension

        # TODO: The big question here is if multiplying by a matrix is faster
        # than doing torch.index_select

        # Let's always work with numpy arrays to make it easier.
        if type(nodes) is int:
            # Change the node number to accommodate the new order
            nodes = self.order.index(nodes)
            # If it's int, make it a list and an array
            nodes = np.array([nodes], dtype=np.int)
            # And repeat for the number of batches
            nodes = np.tile(nodes, batchSize)
        if type(nodes) is list:
            newNodes = [self.order.index(n) for n in nodes]
            nodes = np.array(newNodes, dtype=np.int)
        elif type(nodes) is np.ndarray:
            newNodes = np.array([np.where(np.array(self.order) == n)[0][0] \
                                 for n in nodes])
            nodes = newNodes.astype(np.int)
        # Now, nodes is an np.int np.ndarray with shape batchSize

        # Build the selection matrix
        selectionMatrix = np.zeros([batchSize, self.N[-1], 1])
        selectionMatrix[np.arange(batchSize), nodes, 0] = 1.
        # And convert it to a tensor
        selectionMatrix = torch.tensor(selectionMatrix,
                                       dtype=x.dtype,
                                       device=x.device)

        # Now compute the output
        y = self.forward(x)
        # This output is of size B x dimReadout[-1] x N[-1]

        # Multiply the output
        y = torch.matmul(y, selectionMatrix)
        #   B x dimReadout[-1] x 1

        # Squeeze the last dimension and return
        return y.squeeze(2)

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the terms for the integral Lipschitz computations
        self.IL_terms = [s.to(device) for s in self.IL_terms]
        # Move the GSO
        offset = 0
        for i in range(len(self.S)):
            self.S[i] = self.S[i].to(device)
            # And all the other variables derived from it.
            for l in range(self.L[i]):
                self.GFL[3 * l + offset].addGSO(self.S[i])
                self.GFL[3 * l + 2 + offset].addGSO(self.S[i])
            offset += 3 * self.L[i] + 1


class LocalGNNLine(nn.Module):
    def __init__(self,
                 # Graph filtering
                 dimSignals, nFilterTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimReadout,
                 # Structure
                 GSOs, incidence_matrices, targets=None, order=None):
        # Initialize parent:
        super().__init__()
        # dimSignals should be a list and of size 1 more than nFilter taps.
        numGSOs = len(GSOs)
        for i in range(numGSOs):
            assert len(dimSignals[i]) == len(nFilterTaps[i]) + 1
        # Check whether the GSOs have features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        for i in range(numGSOs):
            GSO = GSOs[i]
            assert len(GSO.shape) == 2 or len(GSO.shape) == 3
            if len(GSO.shape) == 2:
                assert GSO.shape[0] == GSO.shape[1]
                GSOs[i] = torch.unsqueeze(GSO, axis=0)  # 1 x N x N
            else:
                assert GSO.shape[1] == GSO.shape[2]  # E x N x N
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        if nSelectedNodes is None:
            nSelectedNodes = [[GSOs[i].shape[1] for _ in range(len(nFilterTaps[i]))] for i in range(numGSOs)]
        else:
            for i in range(numGSOs):
                assert len(nSelectedNodes[i]) == len(nFilterTaps[i])
        # poolingSize also has to be a list of the same size
        for i in range(numGSOs):
            assert len(poolingSize[i]) == len(nFilterTaps[i])
        # Store the values (using the notation in the paper):
        self.L = [len(nTaps) for nTaps in nFilterTaps]  # Number of graph filtering layers
        self.F = [dims for dims in dimSignals]  # Features
        self.K = [nTaps for nTaps in nFilterTaps]  # Filter taps
        self.E = [GSO.shape[0] for GSO in GSOs]  # Number of edge features
        # For the incidence matrices
        self.B = []
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = alegnn.utils.graphTools.permIdentity
            # This is overridden if coarsening is selected, since the ordering
            # function is native to that pooling method.
        self.S = []
        self.order = []
        for i in range(numGSOs):
            newS, newOrder = self.permFunction(GSOs[i])
            self.S.append(newS)
            self.order.append(newOrder)
            if 'torch' not in repr(self.S[i].dtype):
                self.S[i] = torch.tensor(self.S[i])
            # Permute the incidence matrices to match the GSOs
            self.B.append(incidence_matrices[i][:, newOrder])
            if 'torch' not in repr(self.B[i].dtype):
                self.B[i] = torch.tensor(self.B[i])

        self.alpha = [poolSizes for poolSizes in poolingSize]
        self.N = [[GSOs[0].shape[1]] + nSelectedNodes[i]]  # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias  # Boolean
        # Store the rest of the variables
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.dimReadout = dimReadout
        self.targets = targets
        # And now, we're finally ready to create the architecture:
        # \\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        gfl = []  # Graph Filtering Layers
        offset = 0
        for i in range(numGSOs):
            # Add the pooling layer between GNNs and update offset, if another incidence matrix is provided
            if i < len(self.B):
                gfl.append(PoolCliqueToLine(self.B[i]))
                # Update the offset, since we're starting a new GNN
                if i > 0:
                    offset += 3*self.L[i-1] + 1
                else:
                    offset += 1
            for l in range(self.L[i]):
                # \\ Graph filtering stage:
                gfl.append(gml.GraphFilter(self.F[i][l], self.F[i][l + 1], self.K[i][l],
                                           self.E[i], self.bias))
                # There is a 3*l below here, because we have three elements per
                # layer: graph filter, nonlinearity and pooling, so after each layer
                # we're actually adding elements to the (sequential) list.
                gfl[3 * l + offset].addGSO(self.S[i])
                # \\ Nonlinearity
                gfl.append(self.sigma())
                # \\ Pooling
                gfl.append(self.rho(self.N[i][l], self.N[i][l + 1], self.alpha[i][l]))
                # Same as before, this is 3*l+2
                gfl[3 * l + 2 + offset].addGSO(self.S[i])
        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl)  # Graph Filtering Layers
        # \\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimReadout) > 0:  # Maybe we don't want to readout anything
            # The first layer has to connect whatever was left of the graph
            # filtering stage to create the number of features required by
            # the readout layer
            fc.append(nn.Linear(self.F[-1][-1], dimReadout[0], bias=self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss, or we add a softmax.)
            for l in range(len(dimReadout) - 1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimReadout[l], dimReadout[l + 1],
                                    bias=self.bias))
        # And we're done
        self.Readout = nn.Sequential(*fc)
        # so we finally have the architecture.
        # Lastly, for efficiency we pre-compute some terms for computing the integral Lipschitz constant
        self.construct_IL_terms()

    def changeGSO(self, GSOs, Bs, nSelectedNodes=[], poolingSize=[]):

        # We use this to change the GSO, using the same graph filters.

        # Check that the new GSO has the correct shape
        numGSOs = len(GSOs)
        for i in range(numGSOs):
            GSO = GSOs[i]
            assert len(GSO.shape) == 2 or len(GSO.shape) == 3
            if len(GSO.shape) == 2:
                assert GSO.shape[0] == GSO.shape[1]
                GSOs[i] = torch.unsqueeze(GSO, axis=0)  # 1 x N x N
            else:
                assert GSO.shape[1] == GSO.shape[2]  # E x N x N
            if i < numGSOs - 1:
                B = Bs[i]
                assert B.ndim == 2 and B.shape[0] == GSO.shape[0]  # N x M

        # Loop through all GSOs provided
        for i in range(numGSOs):
            # Get dataType and device of the current GSO, so when we replace it, it
            # is still located in the same type and the same device.
            dataType, device = getDataTypeAndDevice(self.S[i])
            # Reorder the new GSO
            self.S[i], self.order[i] = self.permFunction(GSOs[i])
            # Change data type and device as required
            self.S[i] = changeDataTypeAndDevice(self.S[i], dataType, device)
            if i < numGSOs - 1:
                dataType, device = getDataTypeAndDevice(self.B[i])
                self.B[i] = B[i][self.order, :]
                self.B[i] = changeDataTypeAndDevice(self.B[i], dataType, device)

        # Before making decisions, check if there is a new poolingSize list
        if len(poolingSize) > 0:
            # Check it has the right length
            assert len(poolingSize) == self.L
            # And update it
            self.alpha = poolingSize

        # Now, check if we have a new list of nodes
        if len(nSelectedNodes) > 0:
            # If we do, then we need to change the pooling functions to select
            # fewer nodes. This would allow using graphs of different size.
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
            offset = 0
            for i in range(numGSOs):
                for l in range(self.L[i]):
                    # For each layer, add the pooling function
                    self.GFL[3 * l + 2] = self.rho(self.N[i][l], self.N[i][l + 1],
                                                   self.alpha[i][l])
                offset += 3*self.L[i] + 1

        # And update the GSOs
        offset = 0
        for i in range(numGSOs):
            for l in range(self.L[i]):
                self.GFL[3 * l + offset].addGSO(self.S[i])  # Graph convolutional layer
                self.GFL[3 * l + 2 + offset].addGSO(self.S[i])
            offset += 3 * self.L[i] + 1
        # Lastly, for efficiency we pre-compute some terms for computing the integral Lipschitz constant
        self.construct_IL_terms()

    # Construct terms required to compute the integral Lipschitz constant ahead of time for efficiency
    def construct_IL_terms(self):
        self.IL_terms = []
        for layer in self.GFL:
            if isinstance(layer, gml.GraphFilter):
                lambda_max = torch.max(torch.abs(torch.linalg.eigvalsh(layer.S)))
                self.IL_terms.append( torch.Tensor([0] + [k * lambda_max ** k for k in range(1, layer.K)]) \
                                 .repeat(layer.F, layer.E, layer.G).reshape(layer.F, layer.E, layer.K, layer.G) )

    # Ensure the integral Lipschitz constant constraint is not violated.
    # Otherwise, the loss could be NaN (due to the log barrier penalties)
    def enforce_IL_condition(self, IL_constant):
        # Check if the IL terms have already been created, and compute them if not.
        try:
            self.IL_terms
        except NameError:
            self.construct_IL_terms()

        # Check the integral Lipschitz constants of all filtering layers, and shrink them if
        # the constraint has been violated.
        # We use no grad to make sure this does not impact the gradients to be computed later
        with torch.no_grad():
            layer_counter = 0
            for layer in self.GFL:
                if isinstance(layer, gml.GraphFilter):
                    C_vals = torch.abs(torch.sum(layer.weight * self.IL_terms[layer_counter], dim=2))
                    C_vals = C_vals.unsqueeze(dim=2).repeat_interleave(layer.K, dim=2)

                    # Get the indices of any filter coefficients that violate the condition
                    violated_inds = C_vals >= IL_constant

                    # Shrink these filter coefficients so that the integral Lipschitz constant will be at most 90% of
                    # the constrained value
                    layer.weight[violated_inds] = layer.weight[violated_inds] / C_vals[violated_inds] * IL_constant * 0.9
                    layer_counter += 1

    # Find the largest (or the mean) integral Lipschitz (IL) constant for all IL graph filtering layers.
    def compute_IL_constant(self, return_all=False):
        try:
            self.IL_terms
        except NameError:
            self.construct_IL_terms()

        C_tensor = torch.Tensor().to(self.IL_terms[0].device)
        layer_counter = 0
        for layer in self.GFL:
            if isinstance(layer, gml.GraphFilter):
                C_vals = torch.abs(torch.sum(layer.weight * self.IL_terms[layer_counter], dim=2))
                C_tensor = torch.cat((C_tensor, C_vals.flatten()))
                layer_counter += 1
        if return_all:
            return C_tensor
        else:
            return torch.max(C_tensor)

    def splitForward(self, x):

        # Now we compute the forward call
        assert x.ndim == 3
        assert x.shape[1] == self.F[0][0]
        assert x.shape[2] == self.B[0].shape[0]
        # Do not reorder here, as the incidence matrix will do this
        # Let's call the graph filtering layer
        yGFL = self.GFL(x)
        # Change the order, for the readout
        y = yGFL.permute(0, 2, 1)  # B x N[-1] x F[-1]
        # And, feed it into the Readout layer
        y = self.Readout(y)  # B x N[-1] x dimReadout[-1]
        # Reshape and return
        # return y.permute(0, 2, 1), yGFL
        if self.targets is not None:
            return y[:, self.targets, :], yGFL
        else:
            return y, yGFL
        # B x dimReadout[-1] x N[-1], B x dimFeatures[-1] x N[-1]

    def forward(self, x):

        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward function that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)

        return output

    # TODO: This is NOT updated and should not be used!!!
    def singleNodeForward(self, x, nodes):

        # x is of shape B x F[0] x N[-1]
        batchSize = x.shape[0]
        # nodes is either an int, or a list/np.array of ints of size B
        assert type(nodes) is int \
               or type(nodes) is list \
               or type(nodes) is np.ndarray

        # Let us start by building the selection matrix
        # This selection matrix has to be a matrix of shape
        #   B x N[-1] x 1
        # so that when multiplying with the output of the forward, we get a
        #   B x dimRedout[-1] x 1
        # and we just squeeze the last dimension

        # TODO: The big question here is if multiplying by a matrix is faster
        # than doing torch.index_select

        # Let's always work with numpy arrays to make it easier.
        if type(nodes) is int:
            # Change the node number to accommodate the new order
            nodes = self.order.index(nodes)
            # If it's int, make it a list and an array
            nodes = np.array([nodes], dtype=np.int)
            # And repeat for the number of batches
            nodes = np.tile(nodes, batchSize)
        if type(nodes) is list:
            newNodes = [self.order.index(n) for n in nodes]
            nodes = np.array(newNodes, dtype=np.int)
        elif type(nodes) is np.ndarray:
            newNodes = np.array([np.where(np.array(self.order) == n)[0][0] \
                                 for n in nodes])
            nodes = newNodes.astype(np.int)
        # Now, nodes is an np.int np.ndarray with shape batchSize

        # Build the selection matrix
        selectionMatrix = np.zeros([batchSize, self.N[-1], 1])
        selectionMatrix[np.arange(batchSize), nodes, 0] = 1.
        # And convert it to a tensor
        selectionMatrix = torch.tensor(selectionMatrix,
                                       dtype=x.dtype,
                                       device=x.device)

        # Now compute the output
        y = self.forward(x)
        # This output is of size B x dimReadout[-1] x N[-1]

        # Multiply the output
        y = torch.matmul(y, selectionMatrix)
        #   B x dimReadout[-1] x 1

        # Squeeze the last dimension and return
        return y.squeeze(2)

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the terms for the integral Lipschitz computations
        self.IL_terms = [s.to(device) for s in self.IL_terms]
        # Move the GSO
        offset = 1
        for i in range(len(self.S)):
            self.S[i] = self.S[i].to(device)
            # And all the other variables derived from it.
            for l in range(self.L[i]):
                self.GFL[3 * l + offset].addGSO(self.S[i])
                self.GFL[3 * l + 2 + offset].addGSO(self.S[i])
            offset += 3 * self.L[i] + 1