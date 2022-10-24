import numpy as np
import torch
from alegnn.utils.dataTools import _dataForClassification
from sklearn.metrics import f1_score
# from torchmetrics import F1Score
from scipy.special import softmax
from tqdm import tqdm
from Utils import plot_diffusions_hg


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
        sourceEdges (list of int): list of indices of hyperedges to be used as
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

    def __init__(self, G, nTrain, nValid, nTest, sourceEdges, tMax=None, SC=None, noiseParams=None,
                 doPlots=False, dataType=np.float64, device='cpu', num_folds=None):
        # Initialize parent
        super().__init__()
        # store attributes
        self.dataType = dataType
        self.device = device

        # Get data masks
        self.nTrain = nTrain
        self.nValid = snValid
        self.nTest = nTest
        # total number of samples
        self.nTotal = self.nTrain + self.nValid + self.nTest
        self.targets = sourceEdges
        self.rng = np.random.default_rng()
        self.metric = self.f1Score
        # If no tMax is specified, set it the maximum possible.
        if tMax is None:
            tMax = H.N
        # If we provided a number of folds, we want to do cross-validation.
        if num_folds is not None:
            self.cv_initialize_folds(num_folds)
        else:
            self.num_folds = None

        # \\\ Generate the samples
        # sample source nodes
        sampledSources = np.random.choice(self.targets, size=nTotal)
        # sample diffusion times
        sampledTimes = np.random.choice(np.arange(1, tMax), size=nTotal)

        # Construct the samples for each sampled source hyperedge. In each case,
        # we set the signal values of all nodes in the chosen hyperedge to 1,
        # and diffuse for tMax steps.
        signals_dict = {}
        labels_dict = {}
        for hedge_ind in self.targets:
            # Construct the initial node signal
            hedge = H.hyperedges[hedge_ind]
            x0 = np.array([1 if node_ind in hedge else 0 for node_ind in range(H.N)])
            if noiseParams is not None:
                mu, cov_multiplier = noiseParams
                noise = np.random.multivariate_normal(mu, np.eye(H.N) * cov_multiplier)
            else:
                noise = np.zeros(H.N)

            # Diffuse the node signal for tMax steps
            signals_dict[hedge_ind] = H.diffuse(x0 + noise, tMax)

        # Plot the diffusions
        if doPlots:
            plot_diffusions_hg(SC, H, self.targets, signals_dict=signals_dict, save_dir='../Learning/data/sourceLoc/')

        # Relabel targets as 0,...,k-1
        relabeledSources = {}
        count = 0
        for source_ind in self.targets:
            relabeledSources[source_ind] = count
            count += 1

        # Get the sampled signals
        sampled_signals = np.array([signals_dict[sampledSources[i]][sampledTimes[i], :] for i in range(nTotal)])

        # Generate measurement noise
        if noiseParams is not None:
            mu, cov_multiplier = noiseParams
            # covariance = np.mean(np.abs(sampled_signals), axis=1) * cov_multiplier
            noise = np.random.multivariate_normal(mu, np.eye(H.N)*cov_multiplier, size=nTotal)  # np.array([np.random.multivariate_normal(mu, np.eye(H.N) * covariance[i]) for i in range(nTotal)])
        else:
            noise = np.zeros(nTotal, H.N)

        # Now, we have the signals and the labels
        signals = np.expand_dims(sampled_signals + noise, axis=1)  # nTotal x 1 x N
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

    # Shuffles all samples between training, testing, and validation.
    def shuffle(self):
        signals = np.vstack((self.samples['train']['signals'], self.samples['valid']['signals'],
                             self.samples['test']['signals']))
        labels = np.vstack((self.samples['train']['targets'], self.samples['valid']['targets'],
                            self.samples['test']['targets']))

        shuffledIndices = np.arange(self.nTotal)
        rng = np.random.default_rng()
        self.rng.shuffle(shuffledIndices)
        signals = signals[shuffledIndices]
        labels = labels[shuffledIndices]

        # Save the shuffled samples
        self.samples['train']['signals'] = signals[0:self.nTrain, :, :]
        self.samples['train']['targets'] = labels[0:self.nTrain]
        self.samples['valid']['signals'] = signals[self.nTrain:self.nTrain + self.nValid, :, :]
        self.samples['valid']['targets'] = labels[self.nTrain:self.nTrain + self.nValid]
        self.samples['test']['signals'] = signals[self.nTrain + self.nValid:, :, :]
        self.samples['test']['targets'] = labels[self.nTrain + self.nValid:self.nTotal]

    # Sets the random seed
    def set_random_seed(self, seed):
        assert isinstance(seed, int)
        self.rng = np.random.default_rng(seed=seed)

    # Sets up the cross-validation folds
    def cv_initialize_folds(self, num_folds):
        if not isinstance(num_folds, int):
            num_folds = int(num_folds)

        # Set up the folds, adjusting the number of samples if necessary
        self.num_folds = num_folds
        fold_size = (self.nTrain + self.nValid) // num_folds
        self.nTrain = (num_folds - 1) * fold_size
        self.nValid = fold_size

        # Set up the folds
        shuffledIndices = np.arange(self.nTrain + self.nValid)
        self.rng.shuffle(shuffledIndices)
        self.folds = [shuffledIndices[k * fold_size:(k + 1) * fold_size] for k in range(num_folds)]

    # Sets the validation and training samples up, so that the validation samples are the kth fold and the
    # training samples are everything else. Does not change the test samples.
    def cv_set_fold(self, k):
        if self.num_folds is None:
            print("No folds have been specified. Run cv_initialize_folds() to set up cross-validation.")
            return
        assert 0 <= k < len(self.folds)

        signals = torch.vstack((self.samples['train']['signals'], self.samples['valid']['signals'],
                             self.samples['test']['signals']))
        labels = torch.vstack((self.samples['train']['targets'], self.samples['valid']['targets'],
                            self.samples['test']['targets']))

        validation_indices = self.folds[k]
        train_indices = np.stack(self.folds[:k] + self.folds[(k+1):]).flatten()

        self.samples['train']['signals'] = signals[train_indices, :, :]
        self.samples['train']['targets'] = labels[train_indices]
        self.samples['valid']['signals'] = signals[validation_indices, :, :]
        self.samples['valid']['targets'] = labels[validation_indices]


    def f1Score(self, yHat, y, average='weighted'):
        """
        Return the weighted F1 Score (f1 score of each class, weighted by class size)
        """
        assert average in ['weighted', 'macro', 'micro']
        yHat = torch.argmax(yHat, dim=1).squeeze()
        y = y.squeeze()
        numClasses = len(self.targets)

        if average == 'micro':
            TP_total = torch.zeros(1, device=self.device)
            FP_total = torch.zeros(1, device=self.device)
            FN_total = torch.zeros(1, device=self.device)
        else:
            f1score = torch.zeros(1, device=self.device)

        # Compute F1 score
        for i in range(numClasses):
            predictionMask = yHat == i
            targetMask = y == i
            TP = torch.sum(yHat[predictionMask] == y[predictionMask])
            FP = torch.sum(predictionMask) - TP
            FN = torch.sum(yHat[targetMask] != y[targetMask])
            if average == 'micro':
                TP_total += TP
                FP_total += FP
                FN_total += FN
            # If we're treating classes differently, compute F1 score for this class
            else:
                precision = torch.nan_to_num(TP / (TP + FP), nan=0)
                recall = torch.nan_to_num(TP / (TP + FN), nan=0)
                # Compute F1 score for this class, if it's nonzero
                if precision > 0 and recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    if average == 'weighted':
                        f1score += (f1 * torch.sum(targetMask) / y.shape[0])
                    elif average == 'macro':
                        f1score += f1 / numClasses
        # Compute overall Micro-F1 score
        if average =='micro':
            precision = TP_total / (TP_total + FP_total)
            recall = TP_total / (TP_total + FN_total)
            f1score = torch.nan_to_num(2 * precision * recall / (precision + recall), nan=0)

        return f1score

    # This is set by the architecture, so we return none
    def getTargets(self, inds):
        return None

    def evaluate(self, yHat, y, tol=1e-9):
        """
        Return the accuracy (ratio of yHat = y)
        """
        N = y.shape[0]
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
            # errorRate = self.f1Score(yHat, y, average='micro')
            errorRate = f1_score(y, yHat, average='weighted')
        else:
            # yHat = np.argmax(yHat, axis=1)
            # y = np.array(y)
            errorRate = f1_score(y, yHat, average='weighted')

            #   We compute the target label (hardmax)
            # yHat = np.argmax(yHat, axis=1)
            #   And compute the error
            # totalErrors = np.sum(np.abs(yHat - y) > tol)
            # errorRate = totalErrors.astype(self.dataType) / N
        #   And from that, compute the accuracy
        return errorRate

    def to(self, device):
        super().to(device)
        self.metric = self.metric.to(device)
