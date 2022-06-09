import sys

sys.path.append('../../')
import os
import numpy as np
import itertools
import pandas as pd
import torch
import sfp_nsd_utils as utils
from timeit import default_timer as timer


def torch_log_norm_pdf(x, slope, mode, sigma):
    """the pdf of the log normal distribution, with a scale factor
    """
    # note that mode here is the actual mode, for us, the peak spatial frequency. this differs from
    # the 2d version we have, where we we have np.log2(x)+np.log2(p), so that p is the inverse of
    # the preferred period, the ivnerse of the mode / the peak spatial frequency.
    pdf = slope * torch.exp(-(torch.log2(x) - torch.log2(mode)) ** 2 / (2 * sigma ** 2))

    return pdf


def np_log_norm_pdf(x, slope, mode, sigma):
    """the pdf of the log normal distribution, with a scale factor
    """
    # note that mode here is the actual mode, for us, the peak spatial frequency. this differs from
    # the 2d version we have, where we we have np.log2(x)+np.log2(p), so that p is the inverse of
    # the preferred period, the ivnerse of the mode / the peak spatial frequency.
    pdf = slope * np.exp(-(np.log2(x) - np.log2(mode)) ** 2 / (2 * sigma ** 2))
    return pdf


def _track_loss_df(values, create_loss_df=True, loss_df=None,
                   keys=["subj", "vroinames", "eccrois", "alpha", "epoch", "start_loss", "final_loss"]):
    if create_loss_df:
        loss_df = utils.create_empty_df(col_list=keys)
    else:
        if loss_df is None:
            raise Exception("Dataframe for saving loss log is not defined!")

    loss_df = loss_df.append(dict(zip(keys, values)), ignore_index=True)
    return loss_df


def _set_initial_params(init_list):
    if init_list == "random":
        init_list = np.random.random(3)

    slope = torch.tensor([init_list[0]], dtype=torch.float32, requires_grad=True)
    mode = torch.tensor([init_list[1]], dtype=torch.float32, requires_grad=True)
    sigma = torch.tensor([init_list[2]], dtype=torch.float32, requires_grad=True)
    return slope, mode, sigma


def _df_column_to_torch(df, column):
    torch_column_val = torch.from_numpy(df[column].values)
    return torch_column_val


def fit_1D_model(df, sn, stim_class, varea, ecc_bins="bins", n_print=1000,
                 initial_val="random", epoch=5000, lr=1e-3):
    subj = utils.sub_number_to_string(sn)
    eroi_list = utils.sort_a_df_column(df[ecc_bins])

    # Initialize output df
    loss_history_df = {}
    model_history_df = {}

    # start fitting process
    cur_df = df.query('(subj == @subj) & (names == @stim_class) & (vroinames == @varea)')
    print(f'##{subj}-{stim_class}-{varea} start!##')
    start = timer()
    for cur_ecc in eroi_list:
        tmp_df = cur_df[cur_df[ecc_bins] == cur_ecc]
        x = _df_column_to_torch(tmp_df, "local_sf")
        y = _df_column_to_torch(tmp_df, "avg_betas")
        # set initial parameters
        slope, mode, sigma = _set_initial_params(initial_val)
        # select an optimizer
        optimizer = torch.optim.Adam([slope, mode, sigma], lr=lr)
        # set a loss function - mean squared error
        criterion = torch.nn.MSELoss()

        loss_history = []
        model_history = []
        for t in range(epoch):
            optimizer.zero_grad()
            loss = criterion(torch_log_norm_pdf(x, slope, mode, sigma), y)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            model_history.append([slope.item(), mode.item(), sigma.item()])
            if (t + 1) % n_print == 0:
                print(f'Loss at epoch {str(t + 1)}: {str(round(loss.item(), 3))}')

        print(f'###Eccentricity bin {str(eroi_list.index(cur_ecc) + 1)} out of {len(eroi_list)} finished!###')
        print(
            f'Final parameters: slope {slope.item()}, mode {mode.item()}, sigma {sigma.item()}\n')
        model_history_df[cur_ecc] = pd.DataFrame(model_history, columns=['slope', 'mode', 'sigma'])
        loss_history_df[cur_ecc] = pd.DataFrame(loss_history, columns=['loss'])
    end = timer()
    elapsed_time = end - start
    print(f'###{subj}-{stim_class}-{varea} finished!###')
    print(f'Elapsed time: {np.round(end - start, 2)} sec')
    model_history_df = pd.concat(model_history_df).reset_index().rename(
        columns={'level_0': "ecc_bins", 'level_1': "epoch"})
    loss_history_df = pd.concat(loss_history_df).reset_index().rename(
        columns={'level_0': "ecc_bins", 'level_1': "epoch"})
    loss_history_df['lr'] = lr
    return model_history_df, loss_history_df, elapsed_time


def run_1D_model(df, sn_list, stim_class_list, varea_list, ecc_bins="bins", n_print=1000,
                 initial_val="random", epoch=5000, lr=1e-3):
    model_history_df = {}
    loss_history_df = {}
    for sn in sn_list:

        subj = utils.sub_number_to_string(sn)
        m_m_df = {}
        l_l_df = {}
        for stim_class in stim_class_list:
            m_df = {}
            l_df = {}
            for varea in varea_list:
                m_df[varea], l_df[varea], e_time = fit_1D_model(df, sn=sn, stim_class=stim_class,
                                                                varea=varea, ecc_bins=ecc_bins, n_print=n_print,
                                                                initial_val=initial_val, epoch=epoch, lr=lr)

            m_m_df[stim_class] = pd.concat(m_df)
            l_l_df[stim_class] = pd.concat(l_df)
        model_history_df[subj] = pd.concat(m_m_df)
        loss_history_df[subj] = pd.concat(l_l_df)
    col_replaced = {'level_0': 'subj', 'level_1': 'names', 'level_2': 'vroinames'}
    model_history_df = pd.concat(model_history_df).reset_index().drop(columns='level_3').rename(columns=col_replaced)
    loss_history_df = pd.concat(loss_history_df).reset_index().drop(columns='level_3').rename(columns=col_replaced)

    return model_history_df, loss_history_df


def fit_1D_model_all_subj(input_df, subj_list=None,
                          vroi_list=None, eroi_list=None, initial_val=[1, 1, 1], epoch=5000, alpha=0.025,
                          save_output_df=False,
                          output_df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/first_level_analysis',
                          output_df_name='1D_model_results.csv',
                          save_loss_df=False,
                          loss_df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/first_level_analysis/loss_track',
                          loss_df_name='1D_model_loss.csv'):
    if subj_list is None:
        subj_list = utils.remove_subj_strings(input_df['subj'])
    if vroi_list is None:
        vroi_list = utils.sort_a_df_column(input_df['vroinames'])
    if eroi_list is None:
        eroi_list = utils.sort_a_df_column(input_df['eccrois'])

    # Initialize output df
    output_cols = ['subj', 'vroinames', 'eccrois', 'slope', 'mode', 'sigma']
    output_df = utils.create_empty_df(output_cols)

    loss_cols = ["subj", "vroinames", "eccrois", "alpha", "n_epoch", "start_loss", "final_loss"]
    loss_df = utils.create_empty_df(col_list=loss_cols)

    for sn in subj_list:
        output_single_df, loss_single_df = fit_1D_model(df=input_df, sn=sn, vroi_list=vroi_list, eroi_list=eroi_list,
                                                        initial_val=initial_val, epoch=epoch, lr=alpha,
                                                        save_output_df=save_output_df, output_df_dir=output_df_dir,
                                                        output_df_name=output_df_name, save_loss_df=save_loss_df,
                                                        loss_df_dir=loss_df_dir, loss_df_name=loss_df_name)
        output_df = pd.concat([output_df, output_single_df], ignore_index=True)
        loss_df = pd.concat([loss_df, loss_single_df], ignore_index=True)

    return output_df, loss_df
