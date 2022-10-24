# For taking and processing data from the DHG hypergraph data repository:
# https://download.moon-lab.tech:28501/datasets/
import sys
sys.path.append('../Synthetic_Data_Generation')
sys.path.append('../graph-neural-networks')

import numpy as np
import torch
from alegnn.utils.dataTools import _dataForClassification
from sklearn.metrics import f1_score
import dhg
import pickle
from Hypergraphs import *


class dhgData(_dataForClassification):
    """
    dhgData: Creates the dataset from a dataset in the DHG package

    Initialization:

    Input:
        d (class): DHG data object to use for data generation
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

    def __init__(self, d, dataType=np.float64, device='cpu', num_folds=None):
        # Initialize parent
        super().__init__()
        # store attributes
        self.dataType = dataType
        self.device = device
        self.d = d

        # Get data masks, making sure there are no repetitions
        train_mask = d['train_mask']
        val_mask = torch.logical_and(d['val_mask'], torch.logical_not(train_mask))
        test_mask = torch.logical_and(d['test_mask'], torch.logical_not(torch.logical_or(train_mask, val_mask)))

        # Get indices
        self.indices = {'train': torch.nonzero(train_mask).squeeze(),
                        'valid': torch.nonzero(val_mask).squeeze(),
                        'test': torch.nonzero(test_mask).squeeze()}
        self.indices['all'] = torch.concat((self.indices['train'], self.indices['valid'], self.indices['test']))

        # Number of samples
        self.nTrain = len(self.indices['train'])
        self.nValid = len(self.indices['valid'])
        self.nTest = len(self.indices['test'])
        self.nTotal = self.nTrain + self.nValid + self.nTest
        self.rng = np.random.default_rng()
        self.metric = self.f1Score

        # Data properties
        self.N = d['num_vertices']
        if 'features' in d.content:
            self.F = d['dim_features']
        else:
            self.F = None

        # If we provided a number of folds, we want to do cross-validation.
        if num_folds is not None:
            self.cv_initialize_folds(num_folds)
        else:
            self.num_folds = None
            self.folds = None

        # Save signals (nTotal x dim_features x N) and labels (nTotal x 1)
        self.samples['train']['signals'] = self.make_signals(self.indices['train'])
        self.samples['train']['targets'] = self.make_labels(self.indices['train'])
        self.samples['valid']['signals'] = self.make_signals(self.indices['valid'])
        self.samples['valid']['targets'] = self.make_labels(self.indices['valid'])
        self.samples['test']['signals'] = self.make_signals(self.indices['test'])
        self.samples['test']['targets'] = self.make_labels(self.indices['test'])

        # For testing, if we want a fixed readout
        self.targets = None

        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)

    # Makes the signals from the specified indices
    def make_signals(self, inds_tensor):
        # If we have no features, generate uniform random signals
        if self.F is None:
            return torch.rand(len(inds_tensor), 1, self.N, device=self.device)
        # Build signals.
        else:
            # Use a list for indices so we aren't creating and concatenating tensors constantly
            signal_inds = []
            signal_values = torch.tensor([])
            nSamples = len(inds_tensor)

            for ind in range(nSamples):
                node_ind = inds_tensor[ind]
                nonzero_features = torch.nonzero(self.d['features'][node_ind, :]).flatten()
                signal_inds.extend([[ind, nzf.item(), node_ind] for nzf in nonzero_features])
                signal_values = torch.concat((signal_values, self.d['features'][node_ind, nonzero_features]))
            # Convert the list to a tensor
            signal_inds = torch.tensor(signal_inds).T

            return torch.sparse_coo_tensor(signal_inds, signal_values, (nSamples, self.F, self.N), dtype=self.dataType).coalesce()

    # Creates one-hot encoded labels
    def make_labels(self, inds):
        if type(inds) == list:
            length = len(inds)
        elif type(inds) == int:
            length = 1
        else:
            length = inds.shape[0]
        labels = torch.zeros((length, self.d['num_classes']), dtype=self.dataType)
        labels[range(length), self.d['labels'][inds]] = 1.0
        return labels

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

        # Get fold indices
        self.indices['valid'] = self.folds[k]
        self.indices['train'] = np.stack(self.folds[:k] + self.folds[(k+1):]).flatten()

        self.samples['train']['signals'] = self.make_signals(self.indices['train'])
        self.samples['train']['targets'] = self.make_labels(self.indices['train'])
        self.samples['valid']['signals'] = self.make_signals(self.indices['valid'])
        self.samples['valid']['targets'] = self.make_labels(self.indices['valid'])

    def f1Score(self, yHat, y, average='weighted'):
        """
        Return the weighted F1 Score (f1 score of each class, weighted by class size)
        """
        assert average in ['weighted', 'macro', 'micro']
        yHat = torch.argmax(yHat, dim=1).squeeze()
        y = y.squeeze()
        numClasses = len(self.sourceEdges)

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

    # Returns the targets (readout)
    def getTargets(self, samplesType, inds=None):
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
               or samplesType == 'test'

        if inds is None:
            targets = self.indices[samplesType]
        else:
            targets = self.indices[samplesType][inds]

        return np.stack((np.arange(targets.shape[0]), targets), 1)

    # We need to overwrite the parent class' method since we're using sparse signals
    def getSamples(self, samplesType, *args):
        # samplesType: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
               or samplesType == 'test'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there are no arguments, just return all the desired samples
        x = self.samples[samplesType]['signals']
        y = self.samples[samplesType]['targets']
        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                nSamples = x.shape[0]  # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size=args[0], replace=False)
                # Select the corresponding samples
                # xSelected = x[selectedIndices]
                xSelected = self.make_signals(self.indices[samplesType][selectedIndices])
                y = y[selectedIndices]
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                # xSelected = x[args[0]]
                xSelected = self.make_signals(self.indices[samplesType][args[0]])
                # And assign the labels
                y = y[args[0]]

            # If we only selected a single element, then the nDataPoints dim
            # has been left out. So if we have less dimensions, we have to
            # put it back
            if len(xSelected.shape) < len(x.shape):
                if 'torch' in self.dataType:
                    x = xSelected.unsqueeze(0)
                else:
                    x = np.expand_dims(xSelected, axis=0)
            else:
                x = xSelected

        return x, y

    def evaluate(self, yHat, y, tol=1e-9):
        """
        Return the accuracy (ratio of yHat = y)
        """
        N = y.shape[0]
        if 'torch' in repr(self.dataType):
            #   We compute the target label (hardmax)
            if yHat.ndim > 1:
                yHat = torch.argmax(yHat, dim=1)
            if y.ndim > 1:
                y = torch.argmax(y, dim=1)
            #   And compute the error
            totalErrors = torch.sum(torch.abs(yHat - y) > tol)
            errorRate = totalErrors.type(self.dataType) / N
            '''
            # Take the labels from the GPU to the CPU for computing the evaluation metric, and make sure not to maintain
            # any unneeded gradients
            yHat = np.squeeze(yHat.detach().cpu().numpy())
            if yHat.ndim > 1:
                yHat = np.argmax(yHat, axis=1)
            y = np.squeeze(y.detach().cpu().numpy())
            if y.ndim > 1:
                y = np.argmax(y, axis=1)
            # Compute the F1 score for all classes at once (not treating classes differently)
            # errorRate = self.f1Score(yHat, y, average='macro')
            errorRate = f1_score(y, yHat, average='macro')
            '''
        else:
            # We compute the target label (hardmax)
            if yHat.ndim > 1:
                yHat = np.argmax(yHat, axis=1)
            if y.ndim > 1:
                y = np.argmax(y, axis=1)
            # errorRate = f1_score(y, yHat, average='macro')

            #   And compute the error
            totalErrors = np.sum(np.abs(yHat - y) > tol)
            errorRate = totalErrors.astype(self.dataType) / N
        #   And from that, compute the accuracy
        return 1 - errorRate

    def to(self, device):
        super().to(device)
        # self.metric = self.metric.to(device)


if __name__ == '__main__':
    # Parameters
    useGPU = True
    do_matrices = False

    # Load dataset
    # name = 'dhgCora'
    # d = dhg.data.Cora()
    name = 'dhgCooking'
    d = dhg.data.Cooking200()

    #
    if do_matrices:
        H = Hypergraph(d['edge_list'])

        print('Generating GSOs...')
        L_c = H.clique_laplacian()
        L_l = H.line_laplacian()
        GSOs = [L_c, L_l, L_c]
        with open('../Learning/data/' + name + '/' + name + '_GSOs.pkl','wb') as f:
            pickle.dump(GSOs, f)
        del GSOs

        # Compute the incidence matrix
        print('Creating incidence matrix...')
        B = H.incidence_matrix()
        incidence_matrix = [B, B.T]
        with open('../Learning/data/' + name + '/' + name + '_incidence_matrices.pkl', 'wb') as f:
            pickle.dump(incidence_matrix, f)
        del incidence_matrix

    # Create data object
    dataParams = {'dataType': torch.float64, 'device': 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'}
    data = dhgData(d, **dataParams)
    with open('../Learning/data/' + name + '/' + name + '_data.pkl', 'wb') as f:
        pickle.dump(data, f)
