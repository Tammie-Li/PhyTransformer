import numpy as np 
import mne
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn

transfer = [1, 5, 3, 8, 10, 12, 14, 23, 21, 19, 17, 26, 28, 30, 32, 38, 38, 36, 34, 42, 43, 45, 47, 47, 54, 
            50, 57, 56, 56, 49, 42, 35, 0, 2, 6, 4, 0, 7, 9, 11, 13, 15, 24, 22, 20, 18, 16, 25, 27, 29,
            31, 33, 41, 39, 37, 35, 44, 44, 46, 48, 48, 55, 51, 58]

self_tar_data = self_tar_data[transfer]
self_no_tar_data = self_no_tar_data[transfer]

borikang_montage = mne.channels.make_standard_montage('biosemi64')


borikang_montage = mne.channels.make_standard_montage('biosemi64')
biosemi_montage = mne.channels.make_standard_montage('biosemi256')
ch_types = 'eeg'
sfreq = 256
info_boruikang = mne.create_info(borikang_montage.ch_names, sfreq, ch_types)
info_biosemi = mne.create_info(biosemi_montage.ch_names, sfreq, ch_types)

self_conducted = np.mean(np.concatenate((self_tar_data, self_no_tar_data), axis=1), axis=1).reshape(64, 1)
# self_conducted = self_conducted[transfer]


public_no_tar_data[-8:, :] = 0
public_target_MNE_epoch_data = mne.EvokedArray(public_tar_data, info_biosemi)
public_no_tar_MNE_epoch_data = mne.EvokedArray(public_no_tar_data, info_biosemi)

public = np.mean(np.concatenate((public_tar_data, public_no_tar_data), axis=1), axis=1).reshape(256, 1)
public_MNE_epoch_data = mne.EvokedArray(public, info_biosemi)


self_target_MNE_epoch_data = mne.EvokedArray(self_tar_data, info_boruikang)
self_no_tar_MNE_epoch_data = mne.EvokedArray(self_no_tar_data, info_boruikang)
self_conducted = mne.EvokedArray(self_conducted, info_boruikang)

self_target_MNE_epoch_data.set_montage(borikang_montage)
fig = plt.figure()
im, cm = mne.viz.plot_topomap(self_target_MNE_epoch_data.data[:, 0]/150, self_target_MNE_epoch_data.info, show=False)
fig.colorbar(im)
fig.show()
fig.savefig("a.svg")
