import sys
sys.path.append('../../')
import os
import numpy as np
import itertools
import pandas as pd
import torch
import sfp_nsd_utils as utils

def torch_log_norm_pdf(x, a, mode, sigma):
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
def _track_loss_df(values,
                   create_loss_df=True, loss_df=None,
                   keys=["subj", "vroinames", "eccrois", "alpha", "n_epoch", "start_loss", "final_loss"]):
    if create_loss_df:
        loss_df = utils.create_empty_df(col_list=keys)
    else:
        if loss_df is None:
            raise Exception("Dataframe for saving loss log is not defined!")

    loss_df = loss_df.append(dict(zip(keys, values)), ignore_index=True)
    return loss_df

def _set_initial_params(init_list):
    amp = torch.tensor([init_list[0]], dtype=torch.float32, requires_grad=True)
    mode = torch.tensor([init_list[1]], dtype=torch.float32, requires_grad=True)
    sigma = torch.tensor([init_list[2]], dtype=torch.float32, requires_grad=True)
    return amp, mode, sigma

def _df_column_to_torch(df, column):
    torch_column_val = torch.from_numpy(df[column].values)
    return torch_column_val

def fit_1D_model(input_df, subj,
                 vroi_list=None, eroi_list=None,
                 initial_val=[1, 1, 1], epoch=5000, alpha=0.025,
                 save_output_df=False,
                 output_df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/first_level_analysis',
                 output_df_name='1D_model_results.csv',
                 save_loss_df=False,
                 loss_df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/first_level_analysis/loss_track',
                 loss_df_name='1D_model_loss.csv'):

    sn = utils.sub_number_to_string(subj)
    if vroi_list is None:
        vroi_list = utils.sort_a_df_column(input_df['vroinames'])
    if eroi_list is None:
        eccrois_list = utils.sort_a_df_column(input_df['eccrois'])

    # Initialize output df
    output_cols = ['subj', 'vroinames', 'eccrois', 'amp', 'mode', 'sigma']
    output_single_df = utils.create_empty_df(output_cols)

    loss_cols = ["subj", "vroinames", "eccrois", "alpha", "n_epoch", "start_loss", "final_loss"]
    loss_single_df = utils.create_empty_df(col_list=loss_cols)

    # start fitting process
    for cur_roi, cur_ecc in itertools.product(vroi_list, eccrois_list):
        tmp_single_df = input_df.query('(subj == @sn) & (vroinames == @cur_roi) & (eccrois == @cur_ecc)')
        for axis, col in zip(["x", "y"], ["local_sf", "avg_betas"]):
            globals()[axis] = _df_column_to_torch(tmp_single_df, col)
        # set initial parameters
        amp, mode, sigma = _set_initial_params(initial_val)
        # select an optimizer
        optimizer = torch.optim.Adam([amp, mode, sigma], lr=alpha)
        # set a loss function - mean squared error
        criterion = torch.nn.MSELoss()
        losses = torch.empty(epoch)
        for epoch_count in range(epoch):
            optimizer.zero_grad()
            loss = criterion(torch_log_norm_pdf(x, amp, mode, sigma), y)
            loss.backward()
            optimizer.step()
            losses[epoch_count] = loss.item().detach().numpy()
            if epoch_count+1 % 500 == 0:
                print('epoch ' + str(epoch_count+1) + ' loss ' + str(round(loss.item(), 3)))
            if epoch_count == epoch - 1:
                print('*** ' + sn + ' ' + cur_roi + ' ecc ' + str(cur_ecc) + ' finished ***')
                print(f'amplitude {round(amp.item(), 2)}, mode {round(mode.item(), 2)}, sigma {round(sigma.item())}\n')

        loss_values = [sn, cur_roi, cur_ecc, alpha, epoch, losses[0], losses[-1]]
        loss_single_df = _track_loss_df(values=loss_values, create_loss_df=False, loss_df=loss_single_df)
        output_values = [sn, cur_roi, cur_ecc, amp.detach().numpy(), mode.detach().numpy(), sigma.detach().numpy()]
        tmp_df = pd.DataFrame(dict(zip(output_cols, output_values)))
        output_single_df = pd.concat([output_single_df, tmp_df], ignore_index=True, axis=0)

    if save_output_df:
        output_df_filename = f'{sn}_lr-{alpha}_ep-{epoch}_{output_df_name}'
        utils.save_df_to_csv(output_single_df, output_df_dir, output_df_filename)
    if save_loss_df:
        loss_df_filename = f'{sn}_lr-{alpha}_ep-{epoch}_{loss_df_name}'
        utils.save_df_to_csv(loss_single_df, loss_df_dir, loss_df_filename)

    return output_single_df, loss_single_df



def pytorch_1D_model_fitting(input_df, subj_list=None,
                             vroi_list=None, initial_val=[1, 1, 1], epoch=5000, alpha=0.025,
                             save_output_df=False,
                             output_df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/first_level_analysis',
                             output_df_name='1D_model_results.csv',
                             save_loss_df=False,
                             loss_df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/first_level_analysis/loss_track',
                             loss_df_name='1D_model_loss.csv'):

    if subj_list is not None:
        subj_list = [f"subj{str(i).zfill(2)}" for i in np.arange(1, 9)]
    elif subj_list is None:
        subj_list = input_df.subj.unique()
    if vroi_list is None:
        vroi_list = utils.sort_a_df_column(input_df['vroinames'])

    # only leave subjs and vrois specified in the df
    input_df = input_df[input_df.vroinames.isin(vroi_list) & input_df.subj.isin(subj_list)]
    eccrois_list = utils.sort_a_df_column(input_df['eccrois'])

    # Create output df
    output_df = pd.DataFrame({})
    loss_df = utils.create_empty_df(["subj", "vroinames", "eccrois", "alpha", "n_epoch", "start_loss", "final_loss"])
    for sn in subj_list:
        output_single_df = pd.DataFrame({})
        for cur_roi, cur_ecc in itertools.product(vroi_list, eccrois_list):
            tmp_single_df = input_df.query('(subj == @sn) & (vroinames == @cur_roi) & (eccrois == @cur_ecc)')
            x = torch.from_numpy(tmp_single_df.local_sf.values) #.to(torch.float32)
            y = torch.from_numpy(tmp_single_df.avg_betas.values) #.to(torch.float32)
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
                loss = criterion(torch_log_norm_pdf(x, amp, mode, sigma), y)
                loss.backward()
                optimizer.step()
                losses[epoch_count] = loss.item()
                if epoch_count % 500 == 0 or epoch_count == (epoch-1):
                    print('epoch ' + str(epoch_count) + ' loss ' + str(round(loss.item(), 3)))
                if epoch_count == epoch-1:
                    print('*** ' + sn + ' ' + cur_roi + ' ecc ' + str(cur_ecc) + ' finished ***')
                    print(f'amplitude {round(amp.item(),2)}, mode {round(mode.item(),2)}, sigma {round(sigma.item())}\n')
            values = [sn, cur_roi, cur_ecc, alpha, epoch, losses[0].detach().numpy(), losses[-1].detach().numpy()]
            loss_df = _track_loss_df(values=values, create_loss_df=False, loss_df=loss_df)
            tmp_output_df = pd.DataFrame({'subj': sn,
                             'vroinames': cur_roi,
                             'eccrois': cur_ecc,
                             'amp': amp.detach().numpy(),
                             'mode': mode.detach().numpy(),
                             'sigma': sigma.detach().numpy()})
            output_single_df = pd.concat([output_single_df, tmp_output_df], ignore_index=True, axis=0)
            output_df = pd.concat([output_df, tmp_output_df], ignore_index=True, axis=0)
        if save_output_df:
            if not os.path.exists(output_df_dir):
                os.makedirs(output_df_dir)
            output_path = os.path.join(output_df_dir, f'{sn}_lr-{alpha}_ep-{epoch}_{output_df_name}')
            output_single_df.to_csv(output_path, index=False)
        if save_loss_df:
            if not os.path.exists(loss_df_dir):
                os.makedirs(loss_df_dir)
            loss_output_path = os.path.join(loss_df_dir, f'{sn}_lr-{alpha}_ep-{epoch}_{loss_df_name}')
            loss_df.to_csv(loss_output_path, index=False)
    return output_df, loss_df
