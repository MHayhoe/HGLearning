# 2022/04/12~
# Mikhail Hayhoe, mhayhoe@seas.upenn.edu
# Adapted from code by:
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
"""
Training Module

Trainer classes

Trainer: general trainer that just computes a loss over a training set and
    runs an evaluation on a validation test

"""

import torch
import numpy as np
import os
import pickle
import datetime

from sklearn.metrics import confusion_matrix


class sourceTrainer:
    """
    Trainer: general trainer that just computes a loss over a training set and
        runs an evaluation on a validation test

    Initialization:

        model (Modules.model class): model to train
        data (Utils.data class): needs to have a getSamples and an evaluate
            method
        nEpochs (int): number of epochs (passes over the dataset)
        batchSize (int): size of each minibatch

        Optional (keyword) arguments:

        validationInterval (int): interval of training (number of training
            steps) without running a validation stage.

        learningRateDecayRate (float): float that multiplies the latest learning
            rate used.
        learningRateDecayPeriod (int): how many training steps before
            multiplying the learning rate decay rate by the actual learning
            rate.
        > Obs.: Both of these have to be defined for the learningRateDecay
              scheduler to be activated.
        logger (Visualizer): save tensorboard logs.
        saveDir (string): path to the directory where to save relevant training
            variables.
        printInterval (int): how many training steps after which to print
            partial results (0 means do not print)
        graphNo (int): keep track of what graph realization this is
        realitizationNo (int): keep track of what data realization this is
        >> Alternatively, these last two keyword arguments can be used to keep
            track of different trainings of the same model

    Training:

        .train(): trains the model and returns trainVars dict with the keys
            'nEpochs': number of epochs (int)
            'nBatches': number of batches (int)
            'validationInterval': number of training steps in between
                validation steps (int)
            'batchSize': batch size of each training step (np.array)
            'batchIndex': indices for the start sample and end sample of each
                batch (np.array)
            'lossTrain': loss function on the training samples for each training
                step (np.array)
            'evalTrain': evaluation function on the training samples for each
                training step (np.array)
            'lossValid': loss function on the validation samples for each
                validation step (np.array)
            'evalValid': evaluation function on the validation samples for each
                validation step (np.array)
    """

    def __init__(self, model, data, nEpochs, batchSize, **kwargs):

        # \\\ Store model

        self.model = model
        self.data = data

        ####################################
        # ARGUMENTS (Store chosen options) #
        ####################################

        # Training Options:
        if 'doLogging' in kwargs.keys():
            doLogging = kwargs['doLogging']
        else:
            doLogging = False

        if 'doSaveVars' in kwargs.keys():
            doSaveVars = kwargs['doSaveVars']
        else:
            doSaveVars = True

        if 'printInterval' in kwargs.keys():
            printInterval = kwargs['printInterval']
            if printInterval > 0:
                doPrint = True
            else:
                doPrint = False
        else:
            doPrint = True
            printInterval = (data.nTrain // batchSize) // 5

        if 'learningRateDecayRate' in kwargs.keys() and \
                'learningRateDecayPeriod' in kwargs.keys():
            doLearningRateDecay = True
            learningRateDecayRate = kwargs['learningRateDecayRate']
            learningRateDecayPeriod = kwargs['learningRateDecayPeriod']
        else:
            doLearningRateDecay = False

        if 'validationInterval' in kwargs.keys():
            validationInterval = kwargs['validationInterval']
        else:
            validationInterval = data.nTrain // batchSize

        if 'earlyStoppingLag' in kwargs.keys():
            doEarlyStopping = True
            earlyStoppingLag = kwargs['earlyStoppingLag']
        else:
            doEarlyStopping = False
            earlyStoppingLag = 0

        if 'graphNo' in kwargs.keys():
            graphNo = kwargs['graphNo']
        else:
            graphNo = -1

        if 'realizationNo' in kwargs.keys():
            if 'graphNo' in kwargs.keys():
                realizationNo = kwargs['realizationNo']
            else:
                graphNo = kwargs['realizationNo']
                realizationNo = -1
        else:
            realizationNo = -1

        if doLogging:
            from alegnn.utils.visualTools import Visualizer
            logsTB = os.path.join(self.saveDir, self.name + '-logsTB')
            logger = Visualizer(logsTB, name='visualResults')
        else:
            logger = None

        # No training case:
        if nEpochs == 0:
            doSaveVars = False
            doLogging = False
            # If there's no training happening, there's nothing to report about
            # training losses and stuff.

        ###########################################
        # DATA INPUT (pick up on data parameters) #
        ###########################################

        nTrain = data.nTrain  # size of the training set

        # Number of batches: If the desired number of batches does not split the
        # dataset evenly, we reduce the size of the last batch (the number of
        # samples in the last batch).
        # The variable batchSize is a list of length nBatches (number of
        # batches), where each element of the list is a number indicating the
        # size of the corresponding batch.
        if nTrain < batchSize:
            nBatches = 1
            batchSize = [nTrain]
        elif nTrain % batchSize != 0:
            nBatches = np.ceil(nTrain / batchSize).astype(np.int64)
            batchSize = [batchSize] * nBatches
            # If the sum of all batches so far is not the total number of
            # graphs, start taking away samples from the last batch (remember
            # that we used ceiling, so we are overshooting with the estimated
            # number of batches)
            while sum(batchSize) != nTrain:
                batchSize[-1] -= 1
        # If they fit evenly, then just do so.
        else:
            nBatches = np.int(nTrain / batchSize)
            batchSize = [batchSize] * nBatches
        # batchIndex is used to determine the first and last element of each
        # batch.
        # If batchSize is, for example [20,20,20] meaning that there are three
        # batches of size 20 each, then cumsum will give [20,40,60] which
        # determines the last index of each batch: up to 20, from 20 to 40, and
        # from 40 to 60. We add the 0 at the beginning so that
        # batchIndex[b]:batchIndex[b+1] gives the right samples for batch b.
        batchIndex = np.cumsum(batchSize).tolist()
        batchIndex = [0] + batchIndex

        ###################
        # SAVE ATTRIBUTES #
        ###################

        self.trainingOptions = {}
        self.trainingOptions['doLogging'] = doLogging
        self.trainingOptions['logger'] = logger
        self.trainingOptions['doSaveVars'] = doSaveVars
        self.trainingOptions['doPrint'] = doPrint
        self.trainingOptions['printInterval'] = printInterval
        self.trainingOptions['doLearningRateDecay'] = doLearningRateDecay
        if doLearningRateDecay:
            self.trainingOptions['learningRateDecayRate'] = \
                learningRateDecayRate
            self.trainingOptions['learningRateDecayPeriod'] = \
                learningRateDecayPeriod
        self.trainingOptions['validationInterval'] = validationInterval
        self.trainingOptions['doEarlyStopping'] = doEarlyStopping
        self.trainingOptions['earlyStoppingLag'] = earlyStoppingLag
        self.trainingOptions['batchIndex'] = batchIndex
        self.trainingOptions['batchSize'] = batchSize
        self.trainingOptions['nEpochs'] = nEpochs
        self.trainingOptions['nBatches'] = nBatches
        self.trainingOptions['graphNo'] = graphNo
        self.trainingOptions['realizationNo'] = realizationNo

    def trainBatch(self, thisBatchIndices):

        # Get the samples
        xTrain, yTrain = self.data.getSamples('train', thisBatchIndices)
        xTrain = xTrain.to(self.model.device)
        yTrain = yTrain.to(self.model.device)

        # Start measuring time
        startTime = datetime.datetime.now()

        # Reset gradients
        self.model.archit.zero_grad()

        # Obtain the output of the GNN
        yHatTrain = self.model.archit(xTrain)

        # Compute loss
        lossValueTrain = self.model.loss(yHatTrain, yTrain)

        # Compute gradients
        lossValueTrain.backward()

        # Optimize
        self.model.optim.step()

        # Finish measuring time
        endTime = datetime.datetime.now()

        timeElapsed = abs(endTime - startTime).total_seconds()

        # Compute the accuracy
        #   Note: Using yHatTrain.data creates a new tensor with the
        #   same value, but detaches it from the gradient, so that no
        #   gradient operation is taken into account here.
        #   (Alternatively, we could use a with torch.no_grad():)
        costTrain = self.data.evaluate(yHatTrain.data, yTrain)

        return lossValueTrain.item(), costTrain.item(), timeElapsed

    def validationStep(self):

        # Validation:
        xValid, yValid = self.data.getSamples('valid')
        xValid = xValid.to(self.model.device)
        yValid = yValid.to(self.model.device)

        # Start measuring time
        startTime = datetime.datetime.now()

        # Under torch.no_grad() so that the computations carried out
        # to obtain the validation accuracy are not taken into
        # account to update the learnable parameters.
        with torch.no_grad():
            # Obtain the output of the GNN
            yHatValid = self.model.archit(xValid)

            # Compute loss
            lossValueValid = self.model.loss(yHatValid, yValid)

            # Finish measuring time
            endTime = datetime.datetime.now()

            timeElapsed = abs(endTime - startTime).total_seconds()

            # Compute accuracy:
            costValid = self.data.evaluate(yHatValid, yValid)

        return lossValueValid.item(), costValid.item(), timeElapsed

    def train(self):

        # Get back the training options
        assert 'trainingOptions' in dir(self)
        assert 'doLogging' in self.trainingOptions.keys()
        doLogging = self.trainingOptions['doLogging']
        assert 'logger' in self.trainingOptions.keys()
        logger = self.trainingOptions['logger']
        assert 'doSaveVars' in self.trainingOptions.keys()
        doSaveVars = self.trainingOptions['doSaveVars']
        assert 'doPrint' in self.trainingOptions.keys()
        doPrint = self.trainingOptions['doPrint']
        assert 'printInterval' in self.trainingOptions.keys()
        printInterval = self.trainingOptions['printInterval']
        assert 'doLearningRateDecay' in self.trainingOptions.keys()
        doLearningRateDecay = self.trainingOptions['doLearningRateDecay']
        if doLearningRateDecay:
            assert 'learningRateDecayRate' in self.trainingOptions.keys()
            learningRateDecayRate = self.trainingOptions['learningRateDecayRate']
            assert 'learningRateDecayPeriod' in self.trainingOptions.keys()
            learningRateDecayPeriod = self.trainingOptions['learningRateDecayPeriod']
        assert 'validationInterval' in self.trainingOptions.keys()
        validationInterval = self.trainingOptions['validationInterval']
        assert 'doEarlyStopping' in self.trainingOptions.keys()
        doEarlyStopping = self.trainingOptions['doEarlyStopping']
        assert 'earlyStoppingLag' in self.trainingOptions.keys()
        earlyStoppingLag = self.trainingOptions['earlyStoppingLag']
        assert 'batchIndex' in self.trainingOptions.keys()
        batchIndex = self.trainingOptions['batchIndex']
        assert 'batchSize' in self.trainingOptions.keys()
        batchSize = self.trainingOptions['batchSize']
        assert 'nEpochs' in self.trainingOptions.keys()
        nEpochs = self.trainingOptions['nEpochs']
        assert 'nBatches' in self.trainingOptions.keys()
        nBatches = self.trainingOptions['nBatches']
        assert 'graphNo' in self.trainingOptions.keys()
        graphNo = self.trainingOptions['graphNo']
        assert 'realizationNo' in self.trainingOptions.keys()
        realizationNo = self.trainingOptions['realizationNo']

        # Learning rate scheduler:
        if doLearningRateDecay:
            learningRateScheduler = torch.optim.lr_scheduler.StepLR(
                self.model.optim, learningRateDecayPeriod, learningRateDecayRate)
        else:
            lr_print = self.model.optim.param_groups[0]['lr']

        # Initialize counters (since we give the possibility of early stopping,
        # we had to drop the 'for' and use a 'while' instead):
        epoch = 0  # epoch counter
        lagCount = 0  # lag counter for early stopping

        # Store the training variables
        lossTrain = []
        costTrain = []
        lossValid = []
        costValid = []
        timeTrain = []
        timeValid = []

        while epoch < nEpochs \
                and (lagCount < earlyStoppingLag or (not doEarlyStopping)):
            # The condition will be zero (stop), whenever one of the items of
            # the 'and' is zero. Therefore, we want this to stop only for epoch
            # counting when we are NOT doing early stopping. This can be
            # achieved if the second element of the 'and' is always 1 (so that
            # the first element, the epoch counting, decides). In order to
            # force the second element to be one whenever there is not early
            # stopping, we have an or, and force it to one. So, when we are not
            # doing early stopping, the variable 'not doEarlyStopping' is 1,
            # and the result of the 'or' is 1 regardless of the lagCount. When
            # we do early stopping, then the variable 'not doEarlyStopping' is
            # 0, and the value 1 for the 'or' gate is determined by the lag
            # count.
            # ALTERNATIVELY, we could just keep 'and lagCount<earlyStoppingLag'
            # and be sure that lagCount can only be increased whenever
            # doEarlyStopping is True. But I somehow figured out that would be
            # harder to maintain (more parts of the code to check if we are
            # accidentally increasing lagCount).

            # Randomize dataset for each epoch
            randomPermutation = np.random.permutation(self.data.nTrain)
            # Convert a numpy.array of numpy.int into a list of actual int.
            idxEpoch = [int(i) for i in randomPermutation]

            # Learning decay
            if doLearningRateDecay:
                learningRateScheduler.step()

                if doPrint:
                    # All the optimization have the same learning rate, so just
                    # print one of them
                    lr_print = learningRateScheduler.get_last_lr()[-1]

            # Initialize counter
            batch = 0  # batch counter
            while batch < nBatches \
                    and (lagCount < earlyStoppingLag or (not doEarlyStopping)):

                # Extract the adequate batch
                thisBatchIndices = idxEpoch[batchIndex[batch]
                                            : batchIndex[batch + 1]]

                lossValueTrain, costValueTrain, timeElapsed = \
                    self.trainBatch(thisBatchIndices)

                # Logging values
                if doLogging:
                    lossTrainTB = lossValueTrain
                    costTrainTB = costValueTrain
                # Save values
                lossTrain += [lossValueTrain]
                costTrain += [costValueTrain]
                timeTrain += [timeElapsed]

                # Print:
                if doPrint:
                    if (epoch * nBatches + batch) % printInterval == 0:
                        print("\t(E: %2d, B: %3d, LR: %.8f) %6.4f / %7.4f - %6.4fs" % (
                            epoch + 1, batch + 1, lr_print, costValueTrain,
                            lossValueTrain, timeElapsed),
                              end=' ')
                        if graphNo > -1:
                            print("[%d" % graphNo, end='')
                            if realizationNo > -1:
                                print("/%d" % realizationNo,
                                      end='')
                            print("]", end='')
                        print("")

                # \\\\\\\
                # \\\ TB LOGGING (for each batch)
                # \\\\\\\

                if doLogging:
                    logger.scalar_summary(mode='Training',
                                          epoch=epoch * nBatches + batch,
                                          **{'lossTrain': lossTrainTB,
                                             'costTrain': costTrainTB})

                # \\\\\\\
                # \\\ VALIDATION
                # \\\\\\\

                if (epoch * nBatches + batch) % validationInterval == 0:

                    lossValueValid, costValueValid, timeElapsed = \
                        self.validationStep()

                    # Logging values
                    if doLogging:
                        lossValidTB = lossValueValid
                        costValidTB = costValueValid
                    # Save values
                    lossValid += [lossValueValid]
                    costValid += [costValueValid]
                    timeValid += [timeElapsed]

                    # Print:
                    if doPrint:
                        print("\t(E: %2d, B: %3d, LR: %.8f) %6.4f / %7.4f - %6.4fs" % (
                            epoch + 1, batch + 1, lr_print, costValueTrain,
                            lossValueTrain, timeElapsed),
                              end=' ')
                        print("[VALIDATION", end='')
                        if graphNo > -1:
                            print(".%d" % graphNo, end='')
                            if realizationNo > -1:
                                print("/%d" % realizationNo, end='')
                        print(" (%s)]" % self.model.name)

                    if doLogging:
                        logger.scalar_summary(mode='Validation',
                                              epoch=epoch * nBatches + batch,
                                              **{'lossValid': lossValidTB,
                                                 'costValid': costValidTB})

                    # No previous best option, so let's record the first trial
                    # as the best option
                    if epoch == 0 and batch == 0:
                        bestScore = costValueValid
                        bestEpoch, bestBatch = epoch, batch
                        # Save this model as the best (so far)
                        self.model.save(label='Best')
                        # Start the counter
                        if doEarlyStopping:
                            initialBest = True
                    else:
                        thisValidScore = costValueValid
                        if thisValidScore > bestScore:
                            bestScore = thisValidScore
                            bestEpoch, bestBatch = epoch, batch
                            if doPrint:
                                print("\t=> New best achieved: %.4f" % \
                                      (bestScore))
                            self.model.save(label='Best')
                            # Now that we have found a best that is not the
                            # initial one, we can start counting the lag (if
                            # needed)
                            initialBest = False
                            # If we achieved a new best, then we need to reset
                            # the lag count.
                            if doEarlyStopping:
                                lagCount = 0
                        # If we didn't achieve a new best, increase the lag
                        # count.
                        # Unless it was the initial best, in which case we
                        # haven't found any best yet, so we shouldn't be doing
                        # the early stopping count.
                        elif doEarlyStopping and not initialBest:
                            lagCount += 1

                # \\\\\\\
                # \\\ END OF BATCH:
                # \\\\\\\

                # \\\ Increase batch count:
                batch += 1

            # \\\\\\\
            # \\\ END OF EPOCH:
            # \\\\\\\

            # \\\ Increase epoch count:
            epoch += 1

        # \\\ Save models:
        self.model.save(label='Last')

        #################
        # TRAINING OVER #
        #################

        # We convert the lists into np.arrays
        lossTrain = np.array(lossTrain)
        costTrain = np.array(costTrain)
        lossValid = np.array(lossValid)
        costValid = np.array(costValid)
        costValidBest = np.max(costValid)
        # And we would like to save all the relevant information from
        # training
        trainVars = {'nEpochs': nEpochs,
                     'nBatches': nBatches,
                     'validationInterval': validationInterval,
                     'batchSize': np.array(batchSize),
                     'batchIndex': np.array(batchIndex),
                     'lossTrain': lossTrain,
                     'costTrain': costTrain,
                     'lossValid': lossValid,
                     'costValid': costValid,
                     'costValidBest': costValidBest
                     }

        if doSaveVars:
            saveDirVars = os.path.join(self.model.saveDir, 'trainVars')
            if not os.path.exists(saveDirVars):
                os.makedirs(saveDirVars)
            pathToFile = os.path.join(saveDirVars,
                                      self.model.name + 'trainVars.pkl')
            with open(pathToFile, 'wb') as trainVarsFile:
                pickle.dump(trainVars, trainVarsFile)

        # Now, if we didn't do any training (i.e. nEpochs = 0), then the last is
        # also the best.
        if nEpochs == 0:
            self.model.save(label='Best')
            self.model.save(label='Last')
            if doPrint:
                print("WARNING: No training. Best and Last models are the same.")

        # After training is done, reload best model before proceeding to
        # evaluation:
        self.model.load(label='Best')

        # \\\ Print out best:
        if doPrint and nEpochs > 0:
            print("=> Best validation achieved (E: %d, B: %d): %.4f" % (
                bestEpoch + 1, bestBatch + 1, bestScore))

        return trainVars


"""
evaluation.py Evaluation Module

Methods for evaluating the models.

evaluate: evaluate a model
evaluateSingleNode: evaluate a model that has a single node forward
evaluateFlocking: evaluate a model using the flocking cost
"""


def sourceEvaluate(model, data, **kwargs):
    """
    evaluate: evaluate a model using classification error

    Input:
        model (model class): class from Modules.model
        data (data class): a data class from the Utils.dataTools; it needs to
            have a getSamples method and an evaluate method.
        doPrint (optional, bool): if True prints results

    Output:
        evalVars (dict): 'errorBest' contains the error rate for the best
            model, and 'errorLast' contains the error rate for the last model
    """

    # Get the device we're working on
    device = model.device

    if 'doSaveVars' in kwargs.keys():
        doSaveVars = kwargs['doSaveVars']
    else:
        doSaveVars = True

    ########
    # DATA #
    ########
    # \\\ If CUDA is selected, empty cache:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xTest, yTest = data.getSamples('test')
    xTest = xTest.to(device)
    yTest = yTest.to(device)

    ##############
    # BEST MODEL #
    ##############

    model.load(label='Best')

    with torch.no_grad():
        # Process the samples
        yHatTest = model.archit(xTest)
        # yHatTest is of shape
        #   testSize x numberOfClasses
        # We compute the error
        costBest = data.evaluate(yHatTest, yTest)

        # Confusion matrices
        yHat_np = np.argmax(np.squeeze(yHatTest.detach().cpu().numpy()), axis=1)
        y_np = np.squeeze(yTest.detach().cpu().numpy())
        confusionMatrixBest = confusion_matrix(y_np, yHat_np)

    ##############
    # LAST MODEL #
    ##############

    model.load(label='Last')

    with torch.no_grad():
        # Process the samples
        yHatTest = model.archit(xTest)
        # yHatTest is of shape
        #   testSize x numberOfClasses
        # We compute the error
        costLast = data.evaluate(yHatTest, yTest)

        # Confusion matrices
        yHat_np = np.argmax(np.squeeze(yHatTest.detach().cpu().numpy()), axis=1)
        y_np = np.squeeze(yTest.detach().cpu().numpy())
        confusionMatrixLast = confusion_matrix(y_np, yHat_np)

    # Save the evaluation of the best and last models
    evalVars = {}
    evalVars['costBest'] = costBest.item()
    evalVars['costLast'] = costLast.item()
    evalVars['confusionMatrixBest'] = confusionMatrixBest
    evalVars['confusionMatrixLast'] = confusionMatrixLast

    if doSaveVars:
        saveDirVars = os.path.join(model.saveDir, 'evalVars')
        if not os.path.exists(saveDirVars):
            os.makedirs(saveDirVars)
        pathToFile = os.path.join(saveDirVars, model.name + 'evalVars.pkl')
        with open(pathToFile, 'wb') as evalVarsFile:
            pickle.dump(evalVars, evalVarsFile)

    return evalVars
