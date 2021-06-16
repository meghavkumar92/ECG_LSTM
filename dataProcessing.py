import wfdb
import os
import numpy as np
import wfdb
from wfdb import processing
import glob
from matplotlib import pyplot as plt
import pandas as pd
import torch
#from LSTM_model import RecurrentAutoencoder
import LSTM_model as lstmModel

def data_preprocess(qrs_idx_list,signals):
    lh = 200; rh = 500;
    sig = np.zeros(lh+rh);
    counter = [];
    invalCnter = [];
    sig_samples = [];
    sig_sample_targets = [];
    cnt = 0;
    icnt = 0;
    N = 2000;


    for i, idx_lst in enumerate(qrs_idx_list):
        #print(idx_lst)
        if (idx_lst is None):
            icnt += 1;
            continue;            
        for idx in idx_lst:
            if idx < lh or idx > len(signals) - (rh+1):
                icnt += 1;
                continue;
            #print(idx)
            sigval = np.mean(signals,axis=1);#[i*N:(i+1)*N];
            pkid = i*N+idx
            #print(i,N,sigval);
            sample = (sigval[pkid-lh:pkid+rh] - np.mean(sigval[pkid-lh:pkid+rh]));
            sig_samples.append(sample);            
            sig += sample;
            cnt += 1;
            #print(len(sig))
            #plt.plot(sample); plt.pause(.1)
        
    counter.append(cnt);
    invalCnter.append(icnt);
    print(counter,invalCnter)
    print("data_preprocess method end!");
    ret_samp = pd.DataFrame(sig_samples);
    print(ret_samp.shape)
    return ret_samp

def data_peakidx(signals):
    N = 2000;
    L = len(signals);
    qrs_list = [[None]]*L;
    qrs_flat_list = [];
    min_bpm = 20    
    max_bpm = 230

    search_radius = int(1000 * 60 / max_bpm)
    qlist = [];

    for i in range(0,len(signals),N):
        #print(i)
        qrs_inds = processing.qrs.gqrs_detect(sig=np.mean(signals,axis=1)[i:i+N], fs=1000)
        #print(qrs_inds)

        if not len(qrs_inds):
            corrected_peak_inds = [-1];
            qlist.append(corrected_peak_inds)
            qrs_flat_list = qrs_flat_list + list(corrected_peak_inds)
            continue

        corrected_peak_inds = processing.peaks.correct_peaks(np.mean(signals,axis=1)[i:i+N], peak_inds=qrs_inds, search_radius=search_radius, smooth_window_size=150)

        qlist.append(corrected_peak_inds)
        qrs_flat_list = qrs_flat_list + list(corrected_peak_inds)
    qrs_list = qlist.copy()
    print(len(qrs_list))
    print("data_peakidx method end!");
    ret_sig = data_preprocess(qrs_list,signals)
    print(ret_sig.shape)

    return ret_sig


def read_file():
    d_signal = []
    d_label = []
    sig_labels = {'Reason for admission: Myocardial infarction' : 100,
    'Reason for admission: Cardiomyopathy': 200,
    'Reason for admission: Heart failure (NYHA 2)': 201,
    'Reason for admission: Heart failure (NYHA 3)' : 202,
    'Reason for admission: Heart failure (NYHA 4)': 203,
    'Reason for admission: Bundle branch block': 300, 
    'Reason for admission: Dysrhythmia': 400,
    'Reason for admission: Hypertrophy' : 500,
    'Reason for admission: Valvular heart disease': 600,
    'Reason for admission: Myocarditis': 700,
    'Reason for admission: Unstable angina': 800,
    'Reason for admission: Stable angina': 801,
    'Reason for admission: Palpitation' : 802,
    'Reason for admission: n/a': 803,
    'Reason for admission: Healthy control': 10}
    getSigLabel = lambda kystr : int(sig_labels[kystr]/100)
    print("Read File method")
    for file_name in glob.iglob('uploads/*', recursive=True):
        print(file_name.split('.')[0])
        signals,fields = wfdb.rdsamp(file_name.split('.')[0])
        for i in fields['comments']:
            #print(i)
            if i == 'Reason for admission: Myocardial infarction' or i == 'Reason for admission: Cardiomyopathy' or i == 'Reason for admission: Heart failure (NYHA 2)' or i == 'Reason for admission: Heart failure (NYHA 3)' or i == 'Reason for admission: Heart failure (NYHA 4)' or i == 'Reason for admission: Bundle branch block' or i == 'Reason for admission: Dysrhythmia' or i == 'Reason for admission: Hypertrophy' or  i == 'Reason for admission: Valvular heart disease' or i == 'Reason for admission: Myocarditis' or i == 'Reason for admission: Unstable angina' or i == 'Reason for admission: Stable angina' or i == 'Reason for admission: Palpitation' or i == 'Reason for admission: n/a' or  i == 'Reason for admission: Healthy control':
                d_label.append(getSigLabel(i));
        
        print(signals.shape)
        print(d_label)
        d_signal.append(signals);
    print("read_file method end!");
    ret_ecg_sig = data_peakidx(d_signal[0])
    print(ret_ecg_sig.shape)

    return ret_ecg_sig




def classify_ecg(sig_pred_losses):
    #sig_predictions, sig_pred_losses = lstmModel.predict(model, sig_df)
    THRESHOLD = .025
    pred_mask = (sig_pred_losses <= THRESHOLD)
    pred_correct = sum(l <= THRESHOLD for l in sig_pred_losses)
    pred_perc = pred_correct/len(sig_pred_losses);
    print(f'Correct predictions: {pred_correct}/{len(sig_pred_losses)}',pred_perc)
    return pred_mask, pred_perc


def plot_prediction(data, pred_data, pred_losses, title, ax):

    ax.plot(data[0], label='true')
    ax.plot(pred_data, label='reconstructed')
    ax.set_title(f'{title} (loss: {np.around(pred_losses, 2)})')
    ax.legend()

def make_plot(dataloader, sig_preds, sig_pred_losses):
    fig, axs = plt.subplots(nrows=2,
            ncols=3,
            sharey=True,
            sharex=True,
            figsize=(22, 8)
            )
    axs = axs.flatten()
    for i, data in enumerate(dataloader):
        plot_prediction(data, sig_preds[i], sig_pred_losses[i], title='Test', ax=axs[i])
        plt.pause(.10)
        if i == 5:
            break;

    plt.ion()
    fig.tight_layout();
    plt.show()
    plt.savefig('ECG_reconstruction.png', dpi=300)

if __name__ == "__main__":
    print('In data processing main')

    #""" 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = lstmModel.load_model(device=device);

    ecg_samples_df = read_file()
    
    dataloader, seq_len, n_features = lstmModel.get_dataloader(ecg_samples_df);

    sig_preds, sig_pred_losses = lstmModel.get_predictions(model,dataloader,device);

    pred_mask, pred_perc = classify_ecg(sig_pred_losses)

    make_plot(dataloader,sig_preds,sig_pred_losses) # just for debug and testing

    if( pred_perc < .7):
        print("This is not a normal ECG");
    else:
        print("This is a normal ECG"); 
   # """

    
    