import os
import sys
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

import ModelKeras
from ModelKeras import ATCNet_
from preprocessingkeras import get_data

def draw_learning_curves(history, sub):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy - subject: ' + str(sub))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.close()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss - subject: ' + str(sub))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.close()

def draw_confusion_matrix(cf_matrix, sub, results_path, classes_labels):
    display_labels = classes_labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=display_labels)
    disp.plot()
    disp.ax_.set_xticklabels(display_labels, rotation=12)
    plt.title('Confusion Matrix of Subject: ' + sub)
    plt.savefig(results_path + '/subject_' + sub + '.png')
    plt.show()

def draw_performance_barChart(num_sub, metric, label):
    fig, ax = plt.subplots()
    x = list(range(1, num_sub+1))
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title('Model '+ label + ' per subject')
    ax.set_ylim([0,1])

def train(dataset_conf, train_conf, results_path):
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)

    in_exp = time.time()
    best_models = open(results_path + "/best models.txt", "w")
    log_write = open(results_path + "/log.txt", "w")

    dataset = dataset_conf.get('name')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')

    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves')
    n_train = train_conf.get('n_train')
    model_name = train_conf.get('model')
    from_logits = train_conf.get('from_logits')

    acc = np.zeros((n_sub, n_train))
    kappa = np.zeros((n_sub, n_train))

    for sub in range(n_sub):
        print('\nTraining on subject ', sub+1)
        log_write.write('\nTraining on subject ' + str(sub+1) + '\n')
        BestSubjAcc = 0
        bestTrainingHistory = []

        X_train, _, y_train_onehot, _, _, _ = get_data(
            data_path, sub, dataset, LOSO=LOSO, isStandard=isStandard)

        X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(X_train, y_train_onehot, test_size=0.2, random_state=42)

        for train in range(n_train):
            tf.random.set_seed(train+1)
            np.random.seed(train+1)

            in_run = time.time()

            filepath = os.path.join(results_path, 'saved models', f'run-{train+1}')
            os.makedirs(filepath, exist_ok=True)
            weight_path = os.path.join(filepath, f'subject-{sub+1}.weights.h5')

            model = getModel(model_name, dataset_conf, from_logits)
            model.compile(loss=CategoricalCrossentropy(from_logits=from_logits), optimizer=Adam(learning_rate=lr), metrics=['accuracy'])

            callbacks = [
                ModelCheckpoint(
                    filepath=weight_path,
                    monitor='val_loss',
                    verbose=0,
                    save_best_only=True,
                    save_weights_only=True,
                    mode='min'
                ),
                ReduceLROnPlateau(monitor="val_loss", factor=0.90, patience=20, verbose=0, min_lr=0.0001),
            ]

            history = model.fit(X_train, y_train_onehot, validation_data=(X_val, y_val_onehot),
                                epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)

            # ✅ Load the best weights
            model.load_weights(weight_path)
            model.save(os.path.join(results_path, f"subject-{sub+1}.keras"))  # ✅ Save full model

            # ✅ Save full model in Keras v3 format
            keras_model_path = os.path.join(results_path, 'keras_models', f'run-{train+1}')
            os.makedirs(keras_model_path, exist_ok=True)
            model.save(os.path.join(keras_model_path, f'subject-{sub+1}.keras'))

            y_pred = model.predict(X_val)

            if from_logits:
                y_pred = tf.nn.softmax(y_pred).numpy().argmax(axis=-1)
            else:
                y_pred = y_pred.argmax(axis=-1)

            labels = y_val_onehot.argmax(axis=-1)
            acc[sub, train] = accuracy_score(labels, y_pred)
            kappa[sub, train] = cohen_kappa_score(labels, y_pred)

            out_run = time.time()
            info = f'Subject: {sub+1}   seed {train+1}   time: {(out_run-in_run)/60:.1f} m   '
            info += f'valid_acc: {acc[sub, train]:.4f}   valid_loss: {min(history.history["val_loss"]):.3f}'
            print(info)
            log_write.write(info + '\n')

            if BestSubjAcc < acc[sub, train]:
                BestSubjAcc = acc[sub, train]
                bestTrainingHistory = history

        best_run = np.argmax(acc[sub, :])
        filepath = f'/saved models/run-{best_run+1}/subject-{sub+1}.h5\n'
        best_models.write(filepath)

        if LearnCurves:
            draw_learning_curves(bestTrainingHistory, sub+1)

    best_models.close()
    log_write.close()
    
#%% Evaluation 
def test(model, dataset_conf, results_path, allRuns = True):
    # Open the  "Log" file to write the evaluation results 
    log_write = open(results_path + "/log.txt", "a")
    
    # Get dataset paramters
    dataset = dataset_conf.get('name')
    n_classes = dataset_conf.get('n_classes')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    classes_labels = dataset_conf.get('cl_labels')
     
    # Test the performance based on several runs (seeds)
    runs = os.listdir(results_path) 
    # Initialize variables
    acc = np.zeros((n_sub, len(runs)))
    kappa = np.zeros((n_sub, len(runs)))
    cf_matrix = np.zeros([n_sub, len(runs), n_classes, n_classes])

    # Iteration over subjects 
    # for sub in range(n_sub-1, n_sub): # (num_sub): for all subjects, (i-1,i): for the ith subject.
    inference_time = 0 #  inference_time: classification time for one trial
    for sub in range(n_sub): # (num_sub): for all subjects, (i-1,i): for the ith subject.
        # Load data
        _, _, _, X_test, _, y_test_onehot = get_data(data_path, sub, dataset, LOSO = LOSO, isStandard = isStandard)     

        # Iteration over runs (seeds) 
        for seed in range(len(runs)): 
            # Load the model of the seed.
            model.load_weights('{}/saved models/{}/subject-{}.h5'.format(results_path, runs[seed], sub+1))
            
            inference_time = time.time()
            # Predict MI task
            y_pred = model.predict(X_test).argmax(axis=-1)
            inference_time = (time.time() - inference_time)/X_test.shape[0]
            # Calculate accuracy and K-score          
            labels = y_test_onehot.argmax(axis=-1)
            acc[sub, seed]  = accuracy_score(labels, y_pred)
            kappa[sub, seed] = cohen_kappa_score(labels, y_pred)
            # Calculate and draw confusion matrix
            cf_matrix[sub, seed, :, :] = confusion_matrix(labels, y_pred, normalize='true')
            # draw_confusion_matrix(cf_matrix[sub, seed, :, :], str(sub+1), results_path, classes_labels)
        
    # Print & write the average performance measures for all subjects     
    head1 = head2 = '                  '
    for sub in range(n_sub): 
        head1 = head1 + 'sub_{}   '.format(sub+1)
        head2 = head2 + '-----   '
    head1 = head1 + '  average'
    head2 = head2 + '  -------'
    info = '\n' + head1 +'\n'+ head2
    info = '\n---------------------------------\nTest performance (acc & k-score):\n'
    info = info + '---------------------------------\n' + head1 +'\n'+ head2
    for run in range(len(runs)): 
        info = info + '\nSeed {}: '.format(run+1)
        info_acc = '(acc %)   '
        info_k = '        (k-sco)   '
        for sub in range(n_sub): 
            info_acc = info_acc + '{:.2f}   '.format(acc[sub, run]*100)
            info_k = info_k + '{:.3f}   '.format(kappa[sub, run])
        info_acc = info_acc + '  {:.2f}   '.format(np.average(acc[:, run])*100)
        info_k = info_k + '  {:.3f}   '.format(np.average(kappa[:, run]))
        info = info + info_acc + '\n' + info_k
    info = info + '\n----------------------------------\nAverage - all seeds (acc %): '
    info = info + '{:.2f}\n                    (k-sco): '.format(np.average(acc)*100)
    info = info + '{:.3f}\n\nInference time: {:.2f}'.format(np.average(kappa), inference_time * 1000)
    info = info + ' ms per trial\n----------------------------------\n'
    print(info)
    log_write.write(info+'\n')
         
    # Draw a performance bar chart for all subjects 
    draw_performance_barChart(n_sub, acc.mean(1), 'Accuracy')
    draw_performance_barChart(n_sub, kappa.mean(1), 'k-score')
    # Draw confusion matrix for all subjects (average)
    draw_confusion_matrix(cf_matrix.mean((0,1)), 'All', results_path, classes_labels)
    # Close opened file    
    log_write.close() 
    
    
#%%
def getModel(model_name, dataset_conf, from_logits = False):
    
    n_classes = dataset_conf.get('n_classes')
    n_channels = dataset_conf.get('n_channels')
    in_samples = dataset_conf.get('in_samples')

def getModel(model_name, dataset_conf, from_logits=False):
    
    n_classes = dataset_conf.get('n_classes')
    n_channels = dataset_conf.get('n_channels')
    in_samples = dataset_conf.get('in_samples')

    # Select the model
    if(model_name == 'ATCNet'):
        # Train using the proposed ATCNet model: https://ieeexplore.ieee.org/document/9852687
        model = ModelKeras.ATCNet_(
            n_classes=n_classes, 
            in_chans=n_channels, 
            in_samples=in_samples, 
            n_windows=5, 
            attention='mha',  # Options: None, 'mha','mhla', 'cbam', 'se'
            eegn_F1=16,
            eegn_D=2, 
            eegn_kernelSize=64,
            eegn_poolSize=7,
            eegn_dropout=0.3,
            tcn_depth=2, 
            tcn_kernelSize=4,
            tcn_filters=32,
            tcn_dropout=0.3, 
            tcn_activation='elu',
        )
    elif(model_name == 'TCNet_Fusion'):
        model = ModelKeras.TCNet_Fusion(n_classes=n_classes, Chans=n_channels, Samples=in_samples)
    elif(model_name == 'EEGTCNet'):
        model = ModelKeras.EEGTCNet(n_classes=n_classes, Chans=n_channels, Samples=in_samples)
    elif(model_name == 'EEGNet'):
        model = ModelKeras.EEGNet_classifier(n_classes=n_classes, Chans=n_channels, Samples=in_samples)
    elif(model_name == 'EEGNeX'):
        model = ModelKeras.EEGNeX_8_32(n_timesteps=in_samples, n_features=n_channels, n_outputs=n_classes)
    elif(model_name == 'DeepConvNet'):
        model = ModelKeras.DeepConvNet(nb_classes=n_classes, Chans=n_channels, Samples=in_samples)
    elif(model_name == 'ShallowConvNet'):
        model = ModelKeras.ShallowConvNet(nb_classes=n_classes, Chans=n_channels, Samples=in_samples)
    elif(model_name == 'MBEEG_SENet'):
        model = ModelKeras.MBEEG_SENet(nb_classes=n_classes, Chans=n_channels, Samples=in_samples)
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model

#%%
def run():
    # Define dataset parameters
    dataset = 'BCI2a' # Options: 'BCI2a','HGD', 'CS2R'
    
    if dataset == 'BCI2a': 
        in_samples = 1125
        n_channels = 14
        n_sub = 9
        n_classes = 4
        classes_labels = ['Left hand', 'Right hand','Foot','Tongue']
        data_path = os.path.expanduser('C:/Users/hp/ATC-NET2/BCI2a')

    elif dataset == 'HGD': 
        in_samples = 1125
        n_channels = 44
        n_sub = 14
        n_classes = 4
        classes_labels = ['Right Hand', 'Left Hand','Rest','Feet']     
        data_path = os.path.expanduser('~') + '/mne_data/MNE-schirrmeister2017-data/robintibor/high-gamma-dataset/raw/master/data/'
    elif dataset == 'CS2R': 
        in_samples = 1125
        # in_samples = 576
        n_channels = 32
        n_sub = 18
        n_classes = 3
        # classes_labels = ['Fingers', 'Wrist','Elbow','Rest']     
        classes_labels = ['Fingers', 'Wrist','Elbow']     
        # classes_labels = ['Fingers', 'Elbow']     
        data_path = os.path.expanduser('~') + '/CS2R MI EEG dataset/all/EDF - Cleaned - phase one (remove extra runs)/two sessions/'
    else:
        raise Exception("'{}' dataset is not supported yet!".format(dataset))
        
    # Create a folder to store the results of the experiment
    results_path = "C:/Users/hp/ATC-NET2/Results"

    if not  os.path.exists(results_path):
      os.makedirs(results_path)   # Create a new directory if it does not exist 
      
    # Set dataset paramters 
    dataset_conf = { 'name': dataset, 'n_classes': n_classes, 'cl_labels': classes_labels,
                    'n_sub': n_sub, 'n_channels': n_channels, 'in_samples': in_samples,
                    'data_path': data_path, 'isStandard': True, 'LOSO': False}
    # Set training hyperparamters
    train_conf = { 'batch_size': 64, 'epochs': 300, 'patience': 100, 'lr': 0.001,'n_train': 9,
                  'LearnCurves': True, 'from_logits': False, 'model':'ATCNet'}
           
    # Train the model
    train(dataset_conf, train_conf, results_path)

    # Evaluate the model based on the weights saved in the '/results' folder
    model = getModel(train_conf.get('model'), dataset_conf)
    test(model, dataset_conf, results_path)    

#%%
if __name__ == "__main__":
    run()
    


    


