import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy.io
import gc
import pickle
import itertools
from sklearn.metrics import confusion_matrix
import sys
from keras.backend.tensorflow_backend import set_session
import deeplift
from deeplift.conversion import keras_conversion as kc
from deeplift.blobs import NonlinearMxtsMode
from deeplift.util import compile_func

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

sys.path.insert(0, './preparation')

# Keras imports
import keras
from keras.models import Model
from keras.layers import Input, Conv1D, Conv2D, Dense, Flatten, Dropout, MaxPooling1D, MaxPooling2D, Activation, \
    BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import Callback, warnings


###################################################################
### Callback method for reducing learning rate during training  ###
###################################################################
class AdvancedLearnignRateScheduler(Callback):

    def __init__(self, monitor='val_loss', patience=0, verbose=0, mode='auto', decayRatio=0.1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.decayRatio = decayRatio

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('Mode %s is unknown, '
                          'fallback to auto mode.'
                          % (self.mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        current_lr = K.get_value(self.model.optimizer.lr)
        print("\nLearning rate:", current_lr)
        if current is None:
            warnings.warn('AdvancedLearnignRateScheduler'
                          ' requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('\nEpoch %05d: reducing learning rate' % (epoch))
                    assert hasattr(self.model.optimizer, 'lr'), \
                        'Optimizer must have a "lr" attribute.'
                    current_lr = K.get_value(self.model.optimizer.lr)
                    new_lr = current_lr * self.decayRatio
                    K.set_value(self.model.optimizer.lr, new_lr)
                    self.wait = 0
            self.wait += 1


###########################################
## Function to plot confusion matrices  ##
#########################################
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#     cm = np.around(cm, decimals=3)
#     print(cm)
#
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.savefig('confusion.eps', format='eps', dpi=1000)


#####################################
## Model definition              ##
## ResNet based on Rajpurkar    ##
##################################
# Add CNN layers left branch (higher frequencies)
# Parameters from paper
INPUT_FEAT = 1
OUTPUT_CLASS = 6  # output classes

convfilt = 64
encoder_confilt = 64 # encoder filters' num
encoder_poolstr = 2
convstr = 1
ksize = 4
poolsize = 2
poolstr = 2
drop = 0.5
loop = 4
def ResNet_model(WINDOW_SIZE):

    k = 1  # increment every 4th residual block
    p = True  # pool toggle every other residual block (end with 2^8)
    # Modelling with Functional API
    # input1 = Input(shape=(None,1), name='input')
    input1 = Input(shape=(90, 3), name='input')

    ## encoder def
    enx = Conv1D(filters=encoder_confilt, kernel_size=ksize, activation='relu', padding='same')(input1)
    enx = Activation('relu')(enx)
    enx = Conv1D(filters=encoder_confilt, kernel_size=encoder_poolstr, strides=encoder_poolstr, padding='same')(enx)

    enx = Conv1D(filters=encoder_confilt, kernel_size=ksize, activation='relu', padding='same')(enx)
    enx = Activation('relu')(enx)
    encoded = Conv1D(filters=encoder_confilt, kernel_size=encoder_poolstr, strides=encoder_poolstr, padding='same')(enx)
    # encoded = MaxPooling1D((2), strides=encoder_poolstr, padding='same')(enx)
    ## encoder end


    ## First convolutional block (conv,BN, relu)
    x = Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(encoded)
    x = Activation('relu')(x)
    x = Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=encoder_confilt, kernel_size=encoder_poolstr, strides=poolstr, padding='same')(x)
    x = Dropout(0.5)(x)

    x = Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=encoder_confilt, kernel_size=encoder_poolstr, strides=poolstr, padding='same')(x)
    x = Dropout(0.5)(x)

    # x = Conv1D(filters=convfilt * 2,
    #            kernel_size=ksize,
    #            padding='same',
    #            strides=convstr,
    #            kernel_initializer='he_normal')(x)
    # x = Activation('relu')(x)
    # x = Conv1D(filters=convfilt * 2,
    #            kernel_size=ksize,
    #            padding='same',
    #            strides=convstr,
    #            kernel_initializer='he_normal')(x)
    # x = Activation('relu')(x)
    # x = Conv1D(filters=encoder_confilt, kernel_size=encoder_poolstr, strides=poolstr, padding='same')(x)
    # x = Dropout(0.5)(x)

    # x = Conv1D(filters=convfilt * 2,
    #            kernel_size=ksize,
    #            padding='same',
    #            strides=convstr,
    #            kernel_initializer='he_normal')(x)
    # x = Activation('relu')(x)
    # x = Conv1D(filters=convfilt * 2,
    #            kernel_size=ksize,
    #            padding='same',
    #            strides=convstr,
    #            kernel_initializer='he_normal')(x)
    # x = Activation('relu')(x)
    # x = Conv1D(filters=encoder_confilt, kernel_size=encoder_poolstr, strides=poolstr, padding='same')(x)
    # x = Dropout(0.5)(x)
    # ## Second convolutional block (conv, BN, relu, dropout, conv) with residual net
    # # Left branch (convolutions)
    # x1 = Conv1D(filters=convfilt,
    #             kernel_size=ksize,
    #             padding='same',
    #             strides=convstr,
    #             kernel_initializer='he_normal')(x)
    # x1 = BatchNormalization()(x1)
    # x1 = Activation('relu')(x1)
    # x1 = Dropout(drop)(x1)
    # x1 = Conv1D(filters=convfilt,
    #             kernel_size=ksize,
    #             padding='same',
    #             strides=convstr,
    #             kernel_initializer='he_normal')(x1)
    # x1 = MaxPooling1D(pool_size=poolsize,
    #                   strides=poolstr, padding='same')(x1)
    # # Right branch, shortcut branch pooling
    # x2 = MaxPooling1D(pool_size=poolsize,
    #                   strides=poolstr, padding='same')(x)
    # # Merge both branches
    # x = keras.layers.add([x1, x2])
    # del x1, x2

    # ## Main loop
    # p = not p
    # for l in range(loop):
    #
    #     if (l % 4 == 0) and (l > 0):  # increment k on every fourth residual block
    #         k *= 2
    #         # k += 1
    #         # increase depth by 1x1 Convolution case dimension shall change
    #         xshort = Conv1D(filters=int(convfilt / k), kernel_size=1, padding='same')(x)
    #     else:
    #         xshort = x
    #         # Left branch (convolutions)
    #     # notice the ordering of the operations has changed
    #     x1 = BatchNormalization()(x)
    #     x1 = Activation('relu')(x1)
    #     x1 = Dropout(drop)(x1)
    #     x1 = Conv1D(filters=int(convfilt / k),
    #                 kernel_size=ksize,
    #                 padding='same',
    #                 strides=convstr,
    #                 kernel_initializer='he_normal')(x1)
    #     x1 = BatchNormalization()(x1)
    #     x1 = Activation('relu')(x1)
    #     x1 = Dropout(drop)(x1)
    #     x1 = Conv1D(filters=int(convfilt / k),
    #                 kernel_size=ksize,
    #                 padding='same',
    #                 strides=convstr,
    #                 kernel_initializer='he_normal')(x1)
    #     if p:
    #         x1 = MaxPooling1D(pool_size=poolsize, strides=poolstr, padding='same')(x1)
    #
    #         # Right branch: shortcut connection
    #     if p:
    #         x2 = MaxPooling1D(pool_size=poolsize, strides=poolstr, padding='same')(xshort)
    #     else:
    #         x2 = xshort  # pool or identity
    #     # Merging branches
    #     x = keras.layers.add([x1, x2])
    #     # change parameters
    #     p = not p  # toggle pooling

    # Final bit
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = keras.layers.Average(x)
    x = Flatten()(x)
    # x = Dense(256)(x)
    # x = Dropout(0.5)(x)
    x = Dense(OUTPUT_CLASS)(x)
    # x = Dense(1000)(x)
    out = Activation('softmax')(x)
    model = Model(inputs=input1, outputs=out)
    adam = keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model.summary()
    # sequential_model_to_ascii_printout(model)
    # plot_model(model, to_file='feature_singleRaw_model.png')
    return model


###########################################################
## Function to perform K-fold Crossvalidation on model  ##
##########################################################
def model_eval(X, y, Xval, yval):
    batch = 32
    epochs = 100
    rep = 1  # K fold procedure can be repeated multiple times
    Kfold = 1
    Ntrain = 8528  # number of recordings on training set
    Nsamp = int(Ntrain / Kfold)  # number of recordings to take as validation

    # Need to add dimension for training
    # X = np.expand_dims(X, axis=2)
    # Xval = np.expand_dims(Xval, axis=2)
    classes = [1, 2, 3, 4, 5, 6]
    Nclass = len(classes)
    cvconfusion = np.zeros((Nclass, Nclass, Kfold * rep))
    cvscores = []
    counter = 0
    # repetitions of cross validation





    for r in range(rep):
        print("Rep %d" % (r + 1))
        # cross validation loop
        for k in range(Kfold):
            print("Cross-validation run %d" % (k + 1))
            # Callbacks definition
            callbacks = [
                # Early stopping definition
                EarlyStopping(monitor='val_loss', patience=3, verbose=1),
                # Decrease learning rate by 0.1 factor
                AdvancedLearnignRateScheduler(monitor='val_loss', patience=1, verbose=1, mode='auto', decayRatio=0.1),
                # Saving best model
                ModelCheckpoint('HARWISDM_notNormalized-filters{}-poolingstr{}_resnet-filters{}-ksize{}-poolingstr{}-loopnum{}.hdf5'
                                .format(encoder_confilt, encoder_poolstr, convfilt, ksize, poolstr, loop), monitor='val_loss',
                                save_best_only=True, verbose=1),
            ]
            # Load model
            model = ResNet_model(WINDOW_SIZE)
            model.summary()

            # Loading Encoding Weights
            # weights_matrix = scipy.io.loadmat('recon_filters{}_kernel{}_poolstride{}_withoutDrop.mat'.format(encoder_confilt, ksize, poolstr))
            # weights_matrix = scipy.io.loadmat('./recon_filters64_kernel32_pooling2convstr4_withoutBatchNorm.mat')
            # weights_matrix = weights_matrix['weights_matrix'][0]
            # for i in range(7):
            #     layer = model.layers[i]
            #     #if len(weights_matrix[i]) != 0:
            #     if len(weights_matrix[i]) == 1:
            #         weights = []
            #         weight1 = weights_matrix[i][0][0]
            #         # weights.append(np.reshape(weight1, (weight1.shape[0], weight1.shape[2])))
            #         weights.append(weight1)
            #         weight2 = weights_matrix[i][0][1]
            #         weights.append(np.reshape(weight2, (weight2.shape[1])))
            #         layer.set_weights(weights)
            #     else:
            #         layer.set_weights(weights_matrix[i])


            model.fit(X, y,
                      validation_data=(Xval, yval),
                      epochs=epochs, batch_size=batch, callbacks=callbacks)

            # Evaluate best trained model
            model.load_weights('HARWISDM_notNormalized-filters{}-poolingstr{}_resnet-filters{}-ksize{}-poolingstr{}-loopnum{}.hdf5'
                                .format(encoder_confilt, encoder_poolstr, convfilt, ksize, poolstr, loop))
            ypred = model.predict(Xval)
            ypred = np.argmax(ypred, axis=1)
            ytrue = np.argmax(yval, axis=1)
            cvconfusion[:, :, counter] = confusion_matrix(ytrue, ypred)
            F1 = np.zeros((6, 1))
            for i in range(6):
                F1[i] = 2 * cvconfusion[i, i, counter] / (
                np.sum(cvconfusion[i, :, counter]) + np.sum(cvconfusion[:, i, counter]))
                print("F1 measure for {} rhythm: {:1.4f}".format(classes[i], F1[i, 0]))
            cvscores.append(np.mean(F1) * 100)
            print("Overall F1 measure: {:1.4f}".format(np.mean(F1)))
            K.clear_session()
            gc.collect()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            K.set_session(sess)
            counter += 1
    # Saving cross validation results
    scipy.io.savemat('xval_HARWISDM_notNormalized-filters{}-poolingstr{}_resnet-filters{}-ksize{}-poolingstr{}-loopnum{}.mat'
                     .format(encoder_confilt, encoder_poolstr, convfilt, ksize, poolstr, loop), mdict={'cvconfusion': cvconfusion.tolist()})
    return model


###########################
## Function to load data ##
###########################
def loaddata(WINDOW_SIZE):
    '''
        Load training/test data into workspace

        This function assumes you have downloaded and padded/truncated the
        training set into a local file named "trainingset.mat". This file should
        contain the following structures:
            - trainset: NxM matrix of N ECG segments with length M
            - traintarget: Nx4 matrix of coded labels where each column contains
            one in case it matches ['A', 'N', 'O', '~'].

    '''
    print("Loading data training set")
    matfile = scipy.io.loadmat('../../data/preprocessed_data/CinC/balanced_window60_1.mat')
    matfile2 = scipy.io.loadmat('../../data/preprocessed_data/CinC/balanced_window60_2.mat')
    X1 = matfile['trainset']
    y1 = matfile['traintarget']
    X2 = matfile2['trainset']
    y2 = matfile2['traintarget']
    X = np.concatenate([X1, X2])
    y = np.concatenate([y1, y2])

    load_fn5 = '../../data/preprocessed_data/CinC/balanced_window60_test.mat'
    load_data5 = scipy.io.loadmat(load_fn5)
    final_testset = load_data5['final_testset']
    final_testtarget = load_data5['final_testtarget']

    load_fn_val = '../../data/preprocessed_data/CinC/balanced_window60_val.mat'
    load_data_val = scipy.io.loadmat(load_fn_val)
    val_set = load_data_val['val_set']
    val_target = load_data_val['val_target']

    # Merging datasets
    # Case other sets are available, load them then concatenate
    # y = np.concatenate((traintarget,augtarget),axis=0)
    # X = np.concatenate((trainset,augset),axis=0)

    # X =  X[:,0:WINDOW_SIZE]
    return (X, y), (val_set, val_target)
    # return (X[0:300], y[0:300]),(val_set[0:300], val_target[0:300])

from har_utilities import *
def load_HAR_data():
    X_train, labels_train, list_ch_train = read_data(data_path="./data/", split="train")  # train
    X_test, labels_test, list_ch_test = read_data(data_path="./data/", split="test")  # test
    X = X_train
    y = labels_train
    val_set = X_test
    val_target = labels_test
    return (X, trans2adhoc(y)), (val_set, trans2adhoc(val_target))
def trans2adhoc(y):
    ori_traintarget = np.zeros((len(y), 6))
    classes = [1, 2, 3, 4, 5, 6]
    for row in range(len(y)):
        ori_traintarget[row, classes.index(y[row])] = 1
    return ori_traintarget

def load_HAR_WISDM():
    # processedData = open('./data/processed_WISDM_HAR.pkl', 'rb')
    processedData = open('./data/processed_WISDM_HAR_100000_notNormalized.pkl', 'rb')
    # processedData = open('../processedData/processedData.pkl', 'rb')

    processedData = pickle.load(processedData)
    # train_x = processedData[0]
    train_x, train_y, test_x, test_y = processedData[0], processedData[1], processedData[2], processedData[3]
    train_x = train_x.reshape(len(train_x), 90, 3)
    test_x = test_x.reshape(len(test_x), 90, 3)
    return (train_x, train_y), (test_x, test_y)
#####################
# Main function   ##
###################


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
seed = 7
np.random.seed(seed)

# Parameters
FS = 300
WINDOW_SIZE = 60 * FS  # padding window for CNN
Is_Training = 0

# Loading data
# (X_train, y_train), (Xval, yval) = loaddata(WINDOW_SIZE)
(X_train, y_train), (Xval, yval) = load_HAR_WISDM()

if Is_Training == 1:
    model = model_eval(X_train, y_train, Xval, yval)
else:
    XvalCsv = np.array([range(90 * 3)])
    # Xval2save = np.append(Xval2save, Xval[833:843], axis=0)
    Xval2save = Xval
    Xval2save = Xval2save.reshape(Xval2save.shape[0], Xval2save.shape[1]*Xval2save.shape[2])
    XvalCsv = np.append(XvalCsv, Xval2save, axis=0)
    np.savetxt('raw_all_HARWISDM_notNormalized_Xval.csv', XvalCsv, delimiter=',')
    np.savetxt('raw_all_HARWISDM_notNormalized_yval.csv', yval, delimiter=',')

    model = ResNet_model(WINDOW_SIZE)
    # model = model_eval(X_train, y_train, Xval, yval)
    model.summary()
    model.load_weights('HARWISDM_notNormalized-filters{}-poolingstr{}_resnet-filters{}-ksize{}-poolingstr{}-loopnum{}.hdf5'
                       .format(encoder_confilt, encoder_poolstr, convfilt, ksize, poolstr, loop))
    ypred = model.predict(Xval)
    # result2save = np.array([0, yval[0], ypred[0]])
    # for i in range(len(ypred)):
    #     result2save = np.append(result2save, [i, yval[i], ypred[i]], axis=0)
    # for i in range(len(ypred)):
    #     print(i, yval[i], ypred[i])
# Outputing results of cross validation
matfile = scipy.io.loadmat('xval_HARWISDM_notNormalized-filters{}-poolingstr{}_resnet-filters{}-ksize{}-poolingstr{}-loopnum{}.mat'
                           .format(encoder_confilt, encoder_poolstr, convfilt, ksize, poolstr, loop))
cv = matfile['cvconfusion']
F1mean = np.zeros(cv.shape[2])
for j in range(cv.shape[2]):
    classes = [1, 2, 3, 4, 5, 6]
    F1 = np.zeros((6, 1))
    for i in range(6):
        F1[i] = 2 * cv[i, i, j] / (np.sum(cv[i, :, j]) + np.sum(cv[:, i, j]))
        print("F1 measure for {} rhythm: {:1.4f}".format(classes[i], F1[i, 0]))
    F1mean[j] = np.mean(F1[0:6])
    print("mean F1 measure for: {:1.4f}".format(F1mean[j]))
print("Overall F1 : {:1.4f}".format(np.mean(F1mean)))
# Plotting confusion matrix
cvsum = np.sum(cv, axis=2)
for i in range(6):
    F1[i] = 2 * cvsum[i, i] / (np.sum(cvsum[i, :]) + np.sum(cvsum[:, i]))
    print("F1 measure for {} rhythm: {:1.4f}".format(classes[i], F1[i, 0]))
F1mean = np.mean(F1[0:6])
print("mean F1 measure for: {:1.4f}".format(F1mean))
# plot_confusion_matrix(cvsum, classes, normalize=True, title='Confusion matrix')
if Is_Training == 0:
    deeplift_model = kc.convert_functional_model(model=model, nonlinear_mxts_mode=NonlinearMxtsMode.RevealCancel)
# # deeplift_prediction_func = compile_func([deeplift_model.get_layers()[0].get_activation_vars()],
# #                                        deeplift_model.get_layers()[-1].get_activation_vars())
# # converted_model_predictions = deeplift.util.run_function_in_batches(
# #                                 input_data_list=[Xval],
# #                                 func=deeplift_prediction_func,
# #                                 batch_size=32,
# #                                 progress_update=None)
# # deeplift_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=0, target_layer_idx=-2)
    deeplift_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_name='input', pre_activation_target_layer_name='dense_1')
# scores = np.array(deeplift_contribs_func(task_idx=0,
#                                          input_data_list=[Xval],
#                                          batch_size=10,
#                                          progress_update=1000))
# print(scores)
    ypred = model.predict(Xval)
    np.savetxt('raw_all_HARWISDM_notNormalized_ypred.csv', ypred, delimiter=',')
    final_scores = []
    for task_idx in range(6):
        print("\tComputing scores for task: " + str(task_idx))
        scores = np.array(deeplift_contribs_func(task_idx=task_idx,
                                             input_data_list=[Xval],
                                             batch_size=10,
                                             progress_update=1000))
        # scores = np.squeeze(scores)
    # for s in scores:
    #     sco = np.append(sco, s, axis=0)
        scores = scores.reshape(scores.shape[0], scores.shape[1] * scores.shape[2])
        sco = np.array([range(90 * 3)])
        sco = np.append(sco, scores, axis=0)
        np.savetxt('raw_all_HARWISDM_notNormalized_final_scores_' + str(task_idx) + '.csv', sco, delimiter=',')
        # final_scores.append(scores)
    # ypred = np.array(ypred)
    # final_scores = np.array(final_scores)
    # np.savetxt('final_scores.csv', final_scores, delimiter=',')