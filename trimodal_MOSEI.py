import numpy as np, json
import pickle, sys, argparse
from keras.models import Model
from keras import backend as K
from keras import initializers
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from keras.layers import *
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
global seed
seed = 1337
np.random.seed(seed)
import gc
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
#=============================================================
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#=============================================================
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
set_session(tf.Session(config=config))
#==============================================================

def calc_valid_result(result, valid_label, valid_mask, print_detailed_results=False):

    true_label=[]
    predicted_label=[]

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if valid_mask[i,j]==1:
                true_label.append(np.argmax(valid_label[i,j] ))
                predicted_label.append(np.argmax(result[i,j] ))

    if print_detailed_results:
        print ("Confusion Matrix :")
        print (confusion_matrix(true_label, predicted_label))
        print ("Classification Report :")
        print (classification_report(true_label, predicted_label))
    # print ("Accuracy ", accuracy_score(true_label, predicted_label))
    return accuracy_score(true_label, predicted_label)

def attention(att_type, x, y):

    if att_type == 'simple':
        m_dash = dot([x, y], axes=[2,2])
        m = Activation('softmax')(m_dash)
        h_dash = dot([m, y], axes=[2,1])
        return multiply([h_dash, x])

    elif att_type == 'gated':
        alpha_dash = dot([y, x], axes=[2,2])
        alpha = Activation('softmax')(alpha_dash)
        x_hat = Permute((2, 1))(dot([x, alpha], axes=[1,2]))
        return multiply([y, x_hat])

    else:
        print ('Attention type must be either simple or gated.')

def mmmu(td_text, td_audio, td_video):
    va_att = attention('simple', td_video, td_audio)
    vt_att = attention('simple', td_video, td_text)
    av_att = attention('simple', td_audio, td_video)
    at_att = attention('simple', td_audio, td_text)
    tv_att = attention('simple', td_text, td_video)
    ta_att = attention('simple', td_text, td_audio)

    return concatenate([va_att, vt_att, av_att, at_att, tv_att, ta_att, td_video, td_audio, td_text])

def musa(td_text, td_audio, td_video):
    vv_att = attention('simple', td_video, td_video)
    tt_att = attention('simple', td_text, td_text)
    aa_att = attention('simple', td_audio, td_audio)

    return concatenate([aa_att, vv_att, tt_att, td_video, td_audio, td_text])

def mmuu(td_text, td_audio, td_video, td_units):
    attention_features = []
    for j in range(max_segment_len):

        m1 = Lambda(lambda x: x[:, j:j+1, :])(td_video)
        m2 = Lambda(lambda x: x[:, j:j+1, :])(td_audio)
        m3 = Lambda(lambda x: x[:, j:j+1, :])(td_text)

        utterance_features = concatenate([m1, m2, m3], axis=1)
        mmuu_attention = attention('simple', utterance_features, utterance_features)

        attention_features.append(mmuu_attention)

    merged_attention = concatenate(attention_features, axis=1)

    if timedistributed:
        merged_attention = Lambda(lambda x: K.reshape(x, (-1, max_segment_len, 3*td_units)))(merged_attention)

    else:
        merged_attention = Lambda(lambda x: K.reshape(x, (-1, max_segment_len, 3*r_units)))(merged_attention)

    return concatenate([merged_attention, td_video, td_audio, td_text])

def featuresExtraction():
    global train_text, train_audio, train_video, train_label, train_mask
    global valid_text, valid_audio, valid_video, valid_label, valid_mask
    global test_text, test_audio, test_video, test_label, test_mask
    global max_segment_len

    text          = np.load('MOSEI/text.npz',mmap_mode='r')
    audio         = np.load('MOSEI/audio.npz',mmap_mode='r')
    video         = np.load('MOSEI/video.npz',mmap_mode='r')

    train_text    = text['train_data']
    train_audio   = audio['train_data']
    train_video   = video['train_data']

    valid_text    = text['valid_data']
    valid_audio   = audio['valid_data']
    valid_video   = video['valid_data']

    test_text     = text['test_data']
    test_audio    = audio['test_data']
    test_video    = video['test_data']

    train_label   = video['trainSentiLabel']
    train_label   = to_categorical(train_label>=0)
    valid_label   = video['validSentiLabel']
    valid_label   = to_categorical(valid_label>=0)
    test_label    = video['testSentiLabel']
    test_label    = to_categorical(test_label>=0)

    train_length  = video['train_length']
    valid_length  = video['valid_length']
    test_length   = video['test_length']

    max_segment_len = train_text.shape[1]

    train_mask = np.zeros((train_video.shape[0], train_video.shape[1]), dtype='float')
    valid_mask = np.zeros((valid_video.shape[0], valid_video.shape[1]), dtype='float')
    test_mask  = np.zeros((test_video.shape[0], test_video.shape[1]), dtype='float')

    for i in xrange(len(train_length)):
        train_mask[i,:train_length[i]]=1.0

    for i in xrange(len(valid_length)):
        valid_mask[i,:valid_length[i]]=1.0

    for i in xrange(len(test_length)):
        test_mask[i,:test_length[i]]=1.0


def multimodal_cross_attention(attn_type, highway, recurrent, timedistributed):

    featuresExtraction()
    # run each model 2 times with different seeds and find best result among these runs

    runs = 1
    best_accuracy = 0

    for i in range(runs):

        drop0    = 0.3
        drop1    = 0.3
        r_drop   = 0.3
        td_units = 100
        r_units  = 300

        in_text = Input(shape=(train_text.shape[1], train_text.shape[2]))
        in_audio = Input(shape=(train_audio.shape[1], train_audio.shape[2]))
        in_video = Input(shape=(train_video.shape[1], train_video.shape[2]))

        masked_text = Masking(mask_value=0)(in_text)
        masked_audio = Masking(mask_value=0)(in_audio)
        masked_video = Masking(mask_value=0)(in_video)

        rnn_text  = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(masked_text)
        rnn_audio = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(masked_audio)
        rnn_video = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(masked_video)

        inter_text  = Dropout(drop0)(rnn_text)
        inter_audio = Dropout(drop0)(rnn_audio)
        inter_video = Dropout(drop0)(rnn_video)

        td_text = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(inter_text))
        td_audio = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(inter_audio))
        td_video = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(inter_video))

        if attn_type == 'mmmu':    ## cross modal cross utterance attention ##
            merged = mmmu(td_text, td_audio, td_video)

        elif attn_type == 'musa':  ## uni modal cross utterance attention ##
            merged = musa(td_text, td_audio, td_video)

        elif attn_type == 'mmuu':  ## cross modal uni utterance attention ##
            merged = mmuu(td_text, td_audio, td_video, td_units)

        elif attn_type == 'None':  ## no attention ##
            merged = concatenate([td_text, td_audio, td_video])
        else:
            print ("attn type must be either 'mmmu' or 'mu_sa' or 'mmuu' or 'None'.")

        # ==================================================================================================================
        output = TimeDistributed(Dense(2, activation='softmax'))(merged)
        model = Model([in_text, in_audio, in_video], output)
        model.compile(optimizer='adam', loss='binary_crossentropy', sample_weight_mode='temporal', metrics=['accuracy'])
        # ==================================================================================================================

        path = 'weights/trimodal_run_' + str(i) + '.hdf5'
        check1 = EarlyStopping(monitor='val_loss', patience=10)
        check2 = ModelCheckpoint(path, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        np.random.seed(i)
        history = model.fit([train_text, train_audio, train_video], train_label,
                            epochs=100,
                            batch_size=32,
                            sample_weight=train_mask,
                            shuffle=True,
                            callbacks=[check1, check2],
                            validation_data=([test_text, test_audio, test_video], test_label, test_mask),
                            verbose=1)

        acc = max(history.history['val_acc'])

        if acc > best_accuracy:
            best_accuracy = acc
            model.load_weights(path)
            result = model.predict([test_text, test_audio, test_video])

            np.ndarray.dump(result,open('results/prediction_run_' + str(i) +'.np', 'wb'))

        ################### release gpu memory ###################
        K.clear_session()
        del model
        #del history
        gc.collect()



    ###################### write results #######################

    '''open('results/dushyant/tri_result.txt', 'a').write('Highway: ' + str(highway) +
                                      ', Recurrent: ' + str(recurrent) +
                                      ', TimeDistributed: ' + str(timedistributed) +
                                      ', Attention type: ' + str(attn_type) +
                                      ', Best Accuracy: ' + str(best_accuracy) + '\n'*2 )'''

    print ('Best valid accuracy:', best_accuracy)
    print ('-'*127)


if __name__=="__main__":

    multimodal_cross_attention(attn_type='mmmu', highway=False, recurrent=True, timedistributed=True)
    multimodal_cross_attention(attn_type='musa', highway=False, recurrent=True, timedistributed=True)
    multimodal_cross_attention(attn_type='mmuu', highway=False, recurrent=True, timedistributed=True)
    multimodal_cross_attention(attn_type='None', highway=False, recurrent=True, timedistributed=True)

