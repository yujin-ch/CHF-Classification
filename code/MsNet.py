import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from keras.layers import *
from keras import backend as K
from keras.models import *
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
from keras import initializers
from contextlib import redirect_stdout
import random
from CBAM import CBAM_block
import itertools


def seed_tensorflow(seed=40):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  

seed_tensorflow(45)

def plot_confusion_matrix(cm, classes, title='',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=15)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], fontsize=15,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)


def swish(x):
    """Swish activation function.
    # Arguments
        x: Input tensor.
    # Returns
        The Swish activation: `x * sigmoid(x)`.
    """
    if K.backend() == 'tensorflow':
        try:
            # The native TF implementation has a more
            # memory-efficient gradient implementation
            return K.tf.nn.swish(x)
        except AttributeError:
            pass
    return x * K.sigmoid(x)


def se_block(input_feature, ratio=16):
    init = input_feature
    filters = K.int_shape(init)[-1]

    se_shape = (1, filters)
    se = GlobalAveragePooling1D()(init)

    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', use_bias=True)(se)
    se = Dense(filters, activation='sigmoid', use_bias=True)(se)
    x = multiply([init, se])

    return x


def Conv_bn_relu(num_filters,
                 kernel_size,
                 norm="layer",
                 ac="relu",
                 strides=1,
                 padding='same'):
    def layer(input_tensor):
        x = Conv1D(num_filters, kernel_size,
                   padding=padding, kernel_initializer='he_normal',
                   strides=strides)(input_tensor)
        if norm == "batch":
            x = BatchNormalization()(x)
        if norm == "layer":
            x = LayerNormalization()(x)
        if ac == "relu":
            x = Activation("relu")(x)
        if ac == "swish":
            x = Activation(swish)(x)
        return x

    return layer


def ms_net(num_filters, slice_num, ratio, kernel_size, norm="layer", ac="relu", short_cut="True", only_att="False"):
    def layer(input_tensor):
        identity = input_tensor
        z = Conv_bn_relu(num_filters=num_filters, kernel_size=3, ac=ac, norm=norm)(input_tensor)
        z = Conv_bn_relu(num_filters=(num_filters // slice_num), kernel_size=1, ac=ac, norm=norm)(z)
        for i in range(1, slice_num):
            y = Conv_bn_relu(num_filters=num_filters, kernel_size=(2 * i + 3), ac=ac, norm=norm)(input_tensor)
            y = Conv_bn_relu(num_filters=(num_filters // slice_num), kernel_size=1, ac=ac, norm=norm)(y)
            z = concatenate([z, y])

        if only_att == "True":
            z = se_block(z, ratio)
        else:
            z = CBAM_block(z, ratio, kernel_size=kernel_size)

        if short_cut == "True":
            z = concatenate([z, identity])
        print(input_tensor.shape)
        print('z:',z.shape)
        return z

    return layer


def MsNet(two_layer, norm_type, ac_type, rate1, rate2, rate3, rate4, rate5,
          ratio, kernel_size, norm, ac, short_cut, only_att):
    inputs = Input((100, 49))
    x = Dropout(rate=0.05)(inputs)
    x = ms_net(num_filters=128, slice_num=4, ratio=ratio, kernel_size=kernel_size, norm=norm, ac=ac,
               short_cut=short_cut, only_att=only_att)(inputs)
    x = Dropout(rate=rate1)(x)

    if two_layer == "True":
        x = ms_net(num_filters=128, slice_num=4, ratio=ratio, kernel_size=kernel_size, norm=norm, ac=ac,
                   short_cut=short_cut, only_att=only_att)(x)
        x = Dropout(rate=rate1)(x)

    x = Conv1D(filters=128, kernel_size=11, strides=4, padding="same",
               kernel_initializer=initializers.RandomNormal(stddev=0.01))(x)
    if norm_type == "local":
        x = tf.expand_dims(x, 2)
        x = tf.nn.local_response_normalization(x)
        x = tf.squeeze(x, 2)
    elif norm_type == "layer":
        x = LayerNormalization()(x)
    elif norm_type == "batch":
        x = BatchNormalization()(x)
    if ac_type == "relu":
        x = Activation("relu")(x)
    else:
        x = Activation(swish)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(rate=rate2)(x)
    #
    x = Conv1D(filters=256, kernel_size=5, strides=1, padding="same",
               kernel_initializer=initializers.RandomNormal(stddev=0.01))(x)
    if norm_type == "local":
        x = tf.expand_dims(x, 2)
        x = tf.nn.local_response_normalization(x)
        x = tf.squeeze(x, 2)
    elif norm_type == "layer":
        x = LayerNormalization()(x)
    elif norm_type == "batch":
        x = BatchNormalization()(x)
    if ac_type == "relu":
        x = Activation("relu")(x)
    else:
        x = Activation(swish)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(rate=rate3)(x)
    #
    x = Conv1D(filters=384, kernel_size=3, padding="same",
               kernel_initializer=initializers.RandomNormal(stddev=0.01))(x)
    # x = Dropout(rate=0.1)(x)
    # x = LayerNormalization()(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=384, kernel_size=3, padding="same",
               kernel_initializer=initializers.RandomNormal(stddev=0.01))(x)
    # x = Dropout(rate=0.1)(x)
    # x = LayerNormalization()(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=256, kernel_size=3, padding="same",
               kernel_initializer=initializers.RandomNormal(stddev=0.01))(x)
    # x = Dropout(rate=0.1)(x)
    # x = LayerNormalization()(x)
    x = BatchNormalization()(x)
    if ac_type == "relu":
        x = Activation("relu")(x)
    else:
        x = Activation(swish)(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Dropout(rate=rate4)(x)

    x = Flatten()(x)
    x = Dense(units=1024)(x)
    if ac_type == "relu":
        x = Activation("tanh")(x)
    else:
        x = Activation(swish)(x)
    x = Dropout(rate=rate5)(x)
    # x = Dense(2, activation="softmax")(x)
    x = Dense(2)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    return model


def AlexNet(model_name=""):
    inputs = Input((100, 49), name="input")

    x = Conv1D(filters=128, kernel_size=11, strides=4, padding="same",
               kernel_initializer=initializers.RandomNormal(stddev=0.01))(inputs)
    x = tf.expand_dims(x, 2)
    x = tf.nn.local_response_normalization(x)
    x = tf.squeeze(x, 2)
    x = Activation(activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=256, kernel_size=5, strides=1, padding="same",
               kernel_initializer=initializers.RandomNormal(stddev=0.01))(x)
    x = tf.expand_dims(x, 2)
    x = tf.nn.local_response_normalization(x)
    x = tf.squeeze(x, 2)
    x = Activation(activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=384, kernel_size=3, padding="same",
               kernel_initializer=initializers.RandomNormal(stddev=0.01))(x)
    x = Conv1D(filters=384, kernel_size=3, padding="same",
               kernel_initializer=initializers.RandomNormal(stddev=0.01))(x)
    x = Conv1D(filters=256, kernel_size=3, padding="same",
               kernel_initializer=initializers.RandomNormal(stddev=0.01))(x)
    x = Activation(activation="relu")(x)
    x = MaxPooling1D(pool_size=3)(x)

    x = Flatten()(x)
    x = Dense(units=1024, activation="tanh")(x)
    x = Dropout(rate=0.3)(x)
    x = Dense(units=2)(x)
    x = Softmax()(x)
    model = Model(inputs, x, name=model_name)
    return model


def training(model_name=""):
    flod_idx = 0
    for i in range(1):
        beta_1 = 0.9
        beta_2 = 0.99

        optim_name = "Adam"
        learning_rate = 0.0008
        decay = learning_rate / 50

        two_layer = "True"
        norm_type = "batch"
        ac_type = "swish"
        rate1 = 0.5
        rate2 = 0.5
        rate3 = 0.5
        rate4 = 0.5
        rate5 = 0.5
        ratio = 32

        norm = "layer"
        ac = "swish"
        short_cut = "False"
        only_att = "True"
        if only_att == "False":
            kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7, 9, 11, 13])
        else:
            kernel_size = 3

        ##### Change to AlexNet to run the AlexNet Architecture
        model = MsNet(two_layer, norm_type, ac_type, rate1, rate2, rate3, rate4, rate5,
                      ratio, kernel_size, norm, ac, short_cut, only_att)
        # model = AlexNet()


        model.summary()
        if optim_name == "Adam":
            optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optim_name == "SGD":
            momentum = 0.8
            optim = tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum)

        model.compile(optimizer=optim,
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        flod_idx = flod_idx + 1
        data_dir = r'C:\Users\eugen\Documents\Deep-Learning\final project\model 2\TCN_Data\TCN_Data\data\data'
        x_train = np.load(data_dir + '\\traindata.npy')
        y_train = np.load(data_dir + '\\trainlabel.npy')
        x_test = np.load(data_dir + '\\testdata.npy')
        y_test = np.load(data_dir + '\\testlabel.npy')
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)
        

        x_train = x_train.reshape(-1, 100 * 49)
        x_test = x_test.reshape(-1, 100 * 49)
        scale1 = preprocessing.StandardScaler().fit(x_train)
        x_train = scale1.transform(x_train)
        x_test = scale1.transform(x_test)
        # x_train = x_train.reshape(-1, 49, 100)
        # x_test = x_test.reshape(-1, 49, 100)
        x_train = x_train.reshape(-1, 100, 49)
        x_test = x_test.reshape(-1, 100, 49)
        # x_train = ((x_train - x_train.min()) / (x_train.max() - x_train.min()))
        # x_test = ((x_test - x_test.min()) / (x_test.max() - x_test.min()))

        count = 0
        for num in y_train:
            if num == 1:
                count += 1
        print('Negative Count:',count)

        count = 0
        for num in y_train:
            if num == 0:
                count += 1
        print('Positive Count:', count)

        print(x_train.shape, x_test.shape)
        print(y_train.shape, y_test.shape)
        print(sum(y_train) / len(y_train))
        print(sum(y_test) / len(y_test))
        print('Y:',y_train.shape)
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
        print('Y:', y_train.shape)

        # x_train_input = x_train[:, 10:15, :].reshape(-1, 100, 5)
        # x_test_input = x_test[:, 10:15, :].reshape(-1, 100, 5)
        # print(x_train_input.shape)
        # print(x_test_input.shape)

        monitor = "val_accuracy"
        filepath = r"./results/"
        model_file = filepath + "model.h5"
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        checkpoint = ModelCheckpoint(model_file, monitor=monitor, verbose=0, save_best_only=True,
                                     mode='max',
                                     save_freq='epoch')
        callbacks = [
            checkpoint,
            tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=10, mode='max'),
        ]

        print('-------------------------' + 'flod-' + str(flod_idx) + ' start' + '-------------------------')
        history = model.fit(
            x=x_train, # x_train_input not x_train
            y=y_train,
            batch_size=256,
            epochs=90,
            validation_data=(x_test, #x_test_input
                             y_test),
            callbacks=callbacks,
            shuffle=True
        )
        model_loaded = load_model(model_file)
        loss, accuracy = model_loaded.evaluate(x_test, y_test)  #x_test_input
        y_score = model_loaded.predict(x_test) #x_test_input
        print("Test loss: ", loss)
        print("Accuracy: ", y_score)
        predict = np.argmax(y_score, axis=1)
        fpr1, tpr1, thresholds_keras1 = roc_curve(y_test[:, 1], y_score[:, 1])
        auc1 = auc(fpr1, tpr1)
        print("auc:", auc1)
        tn, fp, fn, tp = confusion_matrix(y_test[:, 1], predict).ravel()
        acc = (tn + tp) / (tn + fp + fn + tp)
        sen = tp / (tp + fn)
        pre = tp / (tp + fp)
        spe = tn / (tn + fp)
        f1 = 2 * pre * sen / (pre + sen)
        print("acc:%6f" % acc, "sen:%6f" % sen, "pre:%6f" % pre, "spe:%6f" % spe, "f1_score:%6f" % f1, "auc:%6f" % auc1)
        print("tn:%d, fp:%d, fn:%d, tp:%d" % (tn, fp, fn, tp))

        with open(filepath + "metrics.txt", 'w') as f:
            f.write("acc:" + str(acc) + "\n")
            f.write("sen:" + str(sen) + "\n")
            f.write("pre:" + str(pre) + "\n")
            f.write("spe:" + str(spe) + "\n")
            f.write("f1_score:" + str(f1) + "\n")
            f.write("auc:" + str(auc1) + "\n")
            f.write("TN:" + str(tn) + "\t")
            f.write("FP:" + str(fp) + "\n")
            f.write("FN:" + str(fn) + "\t")
            f.write("TP:" + str(tp))

        with open("./results/model_summary.txt", 'a') as f:
            with redirect_stdout(f):
                model.summary()
        confusion_matrix1 = confusion_matrix(y_test[:, 1], predict)
        # Plot non-normalized confusion matrix
        class_names = ["N", "SAS"]
        fig1 = plt.figure(figsize=(3, 4))
        plot_confusion_matrix(confusion_matrix1, title="Confusion Matrix"
                              , classes=class_names)
        # plt.show()
        fig1.savefig(filepath + 'Confusion Matrix', dpi=600, format='jpg')

        # Plot ROC
        fig2 = plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.plot(fpr1, tpr1, label='ROC AUC:{:.4f}'.format(auc1))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(loc=4)
        # plt.show()
        fig2.savefig(filepath + 'ROC', dpi=600, format='jpg')

        fig3 = plt.figure()
        plt.title("Accuracy")
        plt.plot(history.history['accuracy'], label="train acc")
        plt.plot(history.history['val_accuracy'], label="validation acc")
        plt.legend()
        plt.ylabel("categorical_accuracy")
        plt.xlabel("epochs")
        # plt.show()
        fig3.savefig(filepath + 'ACC', dpi=600, format='jpg')

        fig4 = plt.figure()
        plt.title("Loss")
        plt.plot(history.history['loss'], label="train loss")
        plt.plot(history.history['val_loss'], label="validation loss")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("epochs")
        # plt.show()
        fig4.savefig(filepath + 'Loss', dpi=600, format='jpg')


training(model_name="MsNet")
print("end")