import numpy as np
import torch
from alegnn.utils.dataTools import _dataForClassification
from sklearn.metrics import f1_score
from scipy.special import softmax
from tqdm import tqdm


def generate_hypergraph_diffusion(sc, n_samples, n_sources, source_upper, timesteps):
    # get the number of nodes
    n = sc.pts.shape[0]

    # compute the gso for the hypergraph (weighted by involvement in multiple hyperedges)
    gso = np.zeros((n, n))
    for he in sc.simplices:
        for ind in range(len(he) - 1):
            for jnd in range(ind + 1, len(he)):
                gso[he[ind], he[jnd]] += 1
                gso[he[jnd], he[ind]] += 1

    # normalize gso
    # obtain eigenvalues
    eigenvalues, _ = np.linalg.eig(gso)

    # normalize by eigenvalue with largest absolute value
    gso = gso / np.max(np.abs(eigenvalues))

    # initialize the tensor used to store the samples
    # shape is n_samples x n x time x 1 features
    z = np.zeros((n_samples, n, timesteps, 1))

    for i in range(n_samples):
        # pick n_sources at random from n nodes
        sources = np.random.choice(n, n_sources, replace=False)

        # define z_0 for each sample
        z[i, sources, 0, 0] = np.random.uniform(0, 10, n_sources)

    # noise mean and variance
    mu = np.zeros(n)
    sigma = np.eye(n) * 1e-3

    for t in range(timesteps - 1):
        # generate noise
        noise = np.random.multivariate_normal(mu, sigma, n_samples)

        # generate z_t
        z[:, :, t + 1] = gso @ z[:, :, t] + np.expand_dims(noise, -1)

    # transpose dimensions so shape is n_samples x time x n x 1 feature
    z = z.transpose((0, 2, 1, 3))

    # squeeze feature dimension, as there is only 1 feature
    return z.squeeze()


class hypergraphSources(_dataForClassification):
    """
    SourceLocalization: Creates the dataset for a source localization problem

    Initialization:

    Input:
        G (class): Graph on which to diffuse the process, needs an attribute
            .N with the number of nodes (int) and attribute .W with the
            adjacency matrix (np.array)
        nTrain (int): number of training samples
        nValid (int): number of validation samples
        nTest (int): number of testing samples
        sourceNodes (list of int): list of indices of nodes to be used as
            sources of the diffusion process
        tMax (int): maximum diffusion time, if None, the maximum diffusion time
            is the size of the graph (default: None)
        dataType (dtype): datatype for the samples created (default: np.float64)
        device (device): if torch.Tensor datatype is selected, this is on what
            device the data is saved.

    Methods:

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label

    .expandDims(): Adds the feature dimension to the graph signals (i.e. for
        graph signals of shape nSamples x nNodes, turns them into shape
        nSamples x 1 x nNodes, so that they can be handled by general graph
        signal processing techniques that take into account a feature dimension
        by default)

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    errorRate = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): unnormalized probability of each label (shape:
                nDataPoints x nHyperedges)
            y (dtype.array): correct labels (2-D binary vector, shape:
                nDataPoints x nHyperedges)
            tol (float, default = 1e-9): numerical tolerance to consider two
                numbers to be equal
        Output:
            errorRate (float): proportion of incorrect labels

    """

    def __init__(self, H, nTrain, nValid, nTest, sourceEdges, tMax=None,
                 dataType=np.float64, device='cpu'):
        # Initialize parent
        super().__init__()
        # store attributes
        self.dataType = dataType
        self.device = device
        self.nTrain = nTrain
        self.nValid = nValid
        self.nTest = nTest
        self.sourceEdges = sourceEdges
        # If no tMax is specified, set it the maximum possible.
        if tMax is None:
            tMax = H.N
        # \\\ Generate the samples
        # total number of samples
        nTotal = nTrain + nValid + nTest
        # sample source nodes
        sampledSources = np.random.choice(sourceEdges, size=nTotal)
        # sample diffusion times
        sampledTimes = np.random.choice(np.arange(1, tMax), size=nTotal)

        # Construct the samples for each sampled source hyperedge. In each case,
        # we set the signal values of all nodes in the chosen hyperedge to 1,
        # and diffuse for tMax steps.
        signals_dict = {}
        labels_dict = {}
        for hedge_ind in sourceEdges:
            # Construct the initial node signal
            hedge = H.hyperedges[hedge_ind]
            x0 = np.array([1 if node_ind in hedge else 0 for node_ind in range(H.N)])

            # Diffuse the node signal for tMax steps
            signals_dict[hedge_ind] = H.diffuse(x0, tMax)

        # Relabel sources as 0,...,k-1
        relabeledSources = {}
        count = 0
        for source_ind in sourceEdges:
            relabeledSources[source_ind] = count
            count += 1

        # Now, we have the signals and the labels
        signals = np.expand_dims(np.array([signals_dict[sampledSources[i]][sampledTimes[i], :]
                                           for i in range(nTotal)]), axis=1)  # nTotal x N
        labels = torch.unsqueeze(torch.DoubleTensor([relabeledSources[sampledSources[i]] for i in range(nTotal)]), dim=1) # nTotal x 1

        # Split and save them
        self.samples['train']['signals'] = signals[0:nTrain, :, :]
        self.samples['train']['targets'] = labels[0:nTrain]
        self.samples['valid']['signals'] = signals[nTrain:nTrain + nValid, :, :]
        self.samples['valid']['targets'] = labels[nTrain:nTrain + nValid]
        self.samples['test']['signals'] = signals[nTrain + nValid:nTotal, :, :]
        self.samples['test']['targets'] = labels[nTrain + nValid:nTotal]
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)

    def evaluate(self, yHat, y, tol=1e-9):
        """
        Return the accuracy (ratio of yHat = y)
        """
        # N = y.shape[0]
        if 'torch' in repr(self.dataType):
            #   We compute the target label (hardmax)
            # yHat = torch.argmax(yHat, dim=1)
            #   And compute the error
            # totalErrors = torch.sum(torch.abs(yHat - y) > tol)
            # errorRate = totalErrors.type(self.dataType) / N
            # Take the labels from the GPU to the CPU for computing the evaluation metric, and make sure not to maintain
            # any unneeded gradients
            yHat = np.argmax(np.squeeze(yHat.detach().cpu().numpy()), axis=1)
            y = np.squeeze(y.detach().cpu().numpy())
            # Compute the F1 score for all classes at once (not treating classes differently)
            errorRate = f1_score(y, yHat, average='weighted')
        else:
            yHat = np.argmax(yHat, axis=1)
            y = np.array(y)
            #   We compute the target label (hardmax)
            # yHat = np.argmax(yHat, axis=1)
            #   And compute the error
            # totalErrors = np.sum(np.abs(yHat - y) > tol)
            # errorRate = totalErrors.astype(self.dataType) / N
            errorRate = f1_score(y, yHat, average='weighted')
        #   And from that, compute the accuracy
        return errorRate
