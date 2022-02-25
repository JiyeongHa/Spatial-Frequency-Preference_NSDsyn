import sys
sys.path.append('../../')
import os
import itertools
import nibabel as nib
import numpy as np
import pandas as pd
import h5py
import re
import itertools
import pandas as pd
import seaborn as sns
import matplotlib as mpl
#mpl.use('module://backend_interagg')
from matplotlib import pyplot as plt
import torch
import binning_eccen

def log_norm_pdf(x, a, mode, sigma):
    """the pdf of the log normal distribution, with a scale factor
    """
    # note that mode here is the actual mode, for us, the peak spatial frequency. this differs from
    # the 2d version we have, where we we have np.log2(x)+np.log2(p), so that p is the inverse of
    # the preferred period, the ivnerse of the mode / the peak spatial frequency.
    pdf = a * torch.exp(-(torch.log2(x)-torch.log2(mode))**2/(2*sigma**2))

    return pdf


def np_log_norm_pdf(x, a, mode, sigma):
    """the pdf of the log normal distribution, with a scale factor
    """
    # note that mode here is the actual mode, for us, the peak spatial frequency. this differs from
    # the 2d version we have, where we we have np.log2(x)+np.log2(p), so that p is the inverse of
    # the preferred period, the ivnerse of the mode / the peak spatial frequency.
    pdf = a * np.exp(-(np.log2(x)-np.log2(mode))**2/(2*sigma**2))
    return pdf

#-------------------
# df should be loaded before 1D_model_fitting part.

#def _create_output_df(input_df=all_subj_df, )
#torch.autograd.set_detect_anomaly(True)

def pytorch_1D_model_fitting(input_df, subj_list=np.arange(1,9),
                             vroi_list=None, initial_val = [1, 1, 5], epoch=5000, alpha=0.05):

    if subj_list is not None:
        subj_list = [str(i).zfill(2) for i in subj_list]
        subj_list = [f"subj{s}" for s in subj_list]
    elif subj_list is None:
        subj_list = input_df.subj.unique()
    if vroi_list is None:
        vroi_list = binning_eccen._sort_vroinames(input_df.vroinames)

    # only leave subjs and vrois specified in the df
    input_df = input_df[input_df.vroinames.isin(vroi_list) & input_df.subj.isin(subj_list)]
    eccrois_list = binning_eccen._sort_vroinames(input_df.eccrois)

    # Create output df
    output_df = pd.DataFrame({})
    for sn in subj_list:
        for cur_roi, cur_ecc in itertools.product(vroi_list, eccrois_list):
            tmp_single_df = input_df.query('(subj == @sn) & (vroinames == @cur_roi) & (eccrois == @cur_ecc)')
            x = torch.from_numpy(tmp_single_df.local_sf.values) #.to(torch.float32)
            y = torch.from_numpy(tmp_single_df.beta.values) #.to(torch.float32)
            #### Torch part ###
            # set initial parameters
            amp = torch.tensor([initial_val[0]], dtype=torch.float32, requires_grad=True)
            mode = torch.tensor([initial_val[1]], dtype=torch.float32, requires_grad=True)
            sigma = torch.tensor([initial_val[2]], dtype=torch.float32, requires_grad=True)
            # select an optimizer
            optimizer = torch.optim.Adam([amp, mode, sigma], lr=alpha)
            # set a loss function - mean squared error
            criterion = torch.nn.MSELoss()
            losses = torch.empty(epoch)
            for epoch_count in range(epoch):
                optimizer.zero_grad()
                loss = criterion(log_norm_pdf(x, amp, mode, sigma), y)
                loss.backward()
                optimizer.step()
                losses[epoch_count] = loss.item()
                if (epoch_count % 500) == 0:
                    print('epoch ' + str(epoch_count) + ' loss ' + str(round(loss.item(), 3)))
                if epoch_count == epoch-1:
                    print('*** ' + sn + ' ' + cur_roi + ' ecc ' + str(cur_ecc) + ' finished ***')
                    print(f'amplitude {round(amp.item(),2)}, mode {round(mode.item(),2)}, sigma {round(sigma.item())}\n')
            tmp_output_df = pd.DataFrame({'subj': sn,
                             'vroinames': cur_roi,
                             'cur_ecc': cur_ecc,
                             'amp': amp.detach().numpy(),
                             'mode': mode.detach().numpy(),
                             'sigma': sigma.detach().numpy()})
            output_df = pd.concat([output_df, tmp_output_df], ignore_index=True, axis=0)
    return output_df

#
#
#
# # plot loss gradient
# plt.plot(losses.detach().numpy())
# plt.show()
# plt.semilogy(losses.detach().numpy())
# plt.show()
#
# # plot final 3 parameters
# plt.plot(x.numpy(), log_norm_pdf(x, amp, mode, sigma).detach().numpy())
# plt.show()
