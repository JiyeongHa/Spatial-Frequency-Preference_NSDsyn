#!/usr/bin/env python3
"""
Cross-validation script for 2D spatial frequency model evaluation.
Tests 4 random classes out of 28 for one subject.
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import random
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import sfp_nsdsyn.utils as utils
from sfp_nsdsyn import two_dimensional_model as model
from sfp_nsdsyn.visualization.plot_2D_model_results import weighted_mean, _change_params_to_math_symbols
import warnings
import matplotlib as mpl
warnings.filterwarnings("ignore")



rc = {'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.edgecolor': 'black',
    'font.family': 'Helvetica',
    'axes.linewidth': 1,
    'axes.labelpad': 3,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'ytick.major.pad': 5,
    'xtick.major.pad': 5,
    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'lines.linewidth': 1,
    'font.size': 11,
    'axes.titlesize': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.title_fontsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 12,
    'figure.dpi': 72 * 3,
    'savefig.dpi': 72 * 4
    }
mpl.rcParams.update(rc)

def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def create_train_test_split(class_idx, n_folds = 7, n_test_classes = 4, random_state=42):
    """
    Create train/test split based on class indices
    """
    # Get all unique class indices
    all_classes = sorted(class_idx)
    shuffled_classes = np.random.permutation(all_classes)
    test_classes_list, train_classes_list = [], []
    for fold in range(n_folds):
        test = shuffled_classes[fold*n_test_classes:(fold+1)*n_test_classes]
        test_classes_list.append(test)
        train_classes_list.append([c for c in all_classes if c not in test])
    
    return train_classes_list, test_classes_list

def split_train_test_dataset(df, train_classes, test_classes):
    """
    Create train/test split based on class indices
    """
    train_df = df.query('class_idx in @train_classes')
    test_df = df.query('class_idx in @test_classes')

    train_dataset = model.SpatialFrequencyDataset(train_df, beta_col='betas')
    test_dataset = model.SpatialFrequencyDataset(test_df, beta_col='betas')
    return train_dataset, test_dataset

def evaluate_model(my_model, dataset):
    """
    Evaluate model on test dataset
    """
    my_model.eval()
    with torch.no_grad():
        predictions = my_model.forward(
            theta_l=dataset.ori, 
            theta_v=dataset.angle, 
            r_v=dataset.eccen, 
            w_l=dataset.sf
        )
        # 
        losses = model.loss_fn(dataset.sigma_v_squared, predictions, dataset.target)
        mean_loss = torch.mean(losses)
        
    return mean_loss.item(), losses.detach().numpy()

def cross_validation_evaluate_model(my_model, dataset):

    """
    Evaluate model on test dataset
    """
    my_model.eval()
    with torch.no_grad():
        predictions = my_model.forward(theta_l=dataset.ori, 
                                       theta_v=dataset.angle, 
                                       r_v=dataset.eccen, 
                                       w_l=dataset.sf)
        target = dataset.target

    return target, predictions


def run_cross_validation(df, sfp_model,
                         n_folds=7, n_test_classes=4, 
                         learning_rate=1e-4, max_epoch=1000, 
                         print_every=100, 
                         random_state=42):
    
    train_classes_list, test_classes_list = create_train_test_split(df.class_idx.unique(), 
                                                                    n_folds = n_folds, 
                                                                    n_test_classes = n_test_classes, 
                                                                    random_state=random_state)

    # Store results
    cv_results = {'fold': [], 'test_classes': []}
    all_targets = []
    all_predictions = []
    all_sigma_v_squared = []
    model_params = pd.DataFrame({})

    # Run cross-validation
    for fold in range(n_folds):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*50}")
        train_dset, test_dset = split_train_test_dataset(df, 
                                                        train_classes_list[fold], 
                                                        test_classes_list[fold])
        sfp_model.train()
        loss_history, model_history, _ = model.fit_model(sfp_model, 
                                                         train_dset,
                                                         learning_rate=learning_rate,
                                                         max_epoch=max_epoch,
                                                         print_every=print_every,
                                                         save_path=None,
                                                         loss_all_voxels=False)

        # Get predictions on testing data 
        sfp_model.eval()
        predictions = sfp_model.forward(theta_l=test_dset.ori, 
                                       theta_v=test_dset.angle, 
                                       r_v=test_dset.eccen, 
                                       w_l=test_dset.sf)

        # Get final model parameters
        final_params = {}
        for name, param in sfp_model.named_parameters():
            if param.requires_grad:
                final_params[name] = param.detach().numpy().item()
        
        # Store results
        cv_results['fold'].append(fold)
        cv_results['test_classes'].append(test_classes_list[fold])

        # Store test losses per voxel
        all_targets.append(test_dset.target)
        all_predictions.append(predictions)
        all_sigma_v_squared.append(test_dset.sigma_v_squared)

        # Store model parameters
        tmp = pd.DataFrame(final_params, index=[0])
        tmp['fold'] = fold
        model_params = pd.concat([model_params, tmp], axis=0)
        model_params.reset_index(drop=True, inplace=True)
        
    
    # Calculate test loss 
    all_targets = torch.cat(all_targets, dim=1)
    all_predictions = torch.cat(all_predictions, dim=1)
    all_sigma_v_squared = torch.cat(all_sigma_v_squared, dim=1)
    losses = model.loss_fn(all_sigma_v_squared, all_predictions, all_targets)
    print(f"Test loss: {losses.mean():.4f}")

    # Merge results other than losses
    cv_results = pd.DataFrame(cv_results)
    cv_results = pd.merge(cv_results, model_params, on='fold')

    return cv_results, losses



def analyze_cv_results(cv_results):
    """
    Analyze cross-validation results
    
    Parameters:
    -----------
    cv_results : dict
        Cross-validation results
    
    Returns:
    --------
    analysis : dict
        Analysis results
    """
    # Convert to DataFrame for easier analysis
    if not isinstance(cv_results, pd.DataFrame):
        results_df = pd.DataFrame({
            'fold': cv_results['fold'],
            'train_loss': cv_results['train_loss'],
            'test_loss': cv_results['test_loss']
        })

    
    # Calculate statistics
    analysis = {
        'mean_train_loss': np.round(np.mean(cv_results['train_loss']),3),
        'std_train_loss': np.round(np.std(cv_results['train_loss']),3),
        'mean_test_loss': np.round(np.mean(cv_results['test_loss']),3),
        'std_test_loss': np.round(np.std(cv_results['test_loss']),3),
        'mean_generalization_error': np.round(np.mean(np.array(cv_results['test_loss']) - np.array(cv_results['train_loss'])),3),
        'results_df': results_df
    }
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    print(f"Mean training loss: {analysis['mean_train_loss']:.4f} ± {analysis['std_train_loss']:.4f}")
    print(f"Mean test loss: {analysis['mean_test_loss']:.4f} ± {analysis['std_test_loss']:.4f}")
    print(f"Mean generalization error: {analysis['mean_generalization_error']:.4f}")
    
    return analysis

def plot_cv_results_group(loss_df, save_path=None):
    """
    Plot cross-validation results
    """
    from matplotlib import gridspec
    sns.set_context("notebook")
    sample_df = loss_df.melt(id_vars=['sub'], 
                            var_name='loss_type', 
                            value_name='loss')
    fig = plt.figure(figsize=(8, 13), dpi=300)
    # Create GridSpec with custom width ratios for top row
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 2], height_ratios=[1, 1, 1])

    # Create axes using the GridSpec
    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0])  # larger
    axes[0, 1] = fig.add_subplot(gs[0, 1])  # smaller
    axes[1, 0] = fig.add_subplot(gs[1, 0:2])  # same size
    axes[1, 1] = fig.add_subplot(gs[2, 0:2])  # same size
    # Plot 1: Train vs Test loss across folds & subjects
    sns.barplot(ax=axes[0, 0], 
                data=sample_df, 
                x='loss_type', y='loss', hue='loss_type', errorbar=('ci', 68))

    axes[0, 1].set_xlabel('Fold')
    fig.text(0.45, 1.01, 'Train vs Test Loss', ha='center', va='top', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    # Plot 3: Train loss for each subject and fold
    sns.lineplot(ax=axes[1, 0], data=all_df, x='fold', y='train_loss', hue='sub', hue_order=all_df['sub'].unique())
    axes[1, 0].set_xlabel(' ')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Train Loss for Each Subject and Fold', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    # Plot 4: Test loss for each subject and fold
    sns.lineplot(ax=axes[1, 1], data=all_df, x='fold', y='test_loss', hue='sub', hue_order=all_df['sub'].unique())
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].set_title('Test Loss for Each Subject and Fold', fontweight='bold')
    for ax in axes.flat:
        ax.set_ylabel('Loss')

    plt.tight_layout()
    if save_path:
        utils.save_fig(save_path)
    plt.show()

def plot_cv_results(cv_results, analysis, save_path=None):
    """
    Plot cross-validation results
    
    Parameters:
    -----------
    cv_results : dict
        Cross-validation results
    analysis : dict
        Analysis results
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Train vs Test loss across folds
    axes[0, 0].plot(cv_results['fold'], cv_results['train_loss'], 'b-o', label='Train Loss')
    axes[0, 0].plot(cv_results['fold'], cv_results['test_loss'], 'r-o', label='Test Loss')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Train vs Test Loss Across Folds')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss distribution
    axes[0, 1].hist(cv_results['train_loss'], alpha=0.7, label='Train Loss', bins=10)
    axes[0, 1].hist(cv_results['test_loss'], alpha=0.7, label='Test Loss', bins=10)
    axes[0, 1].set_xlabel('Loss')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Loss Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Generalization error
    gen_error = np.array(cv_results['test_loss']) - np.array(cv_results['train_loss'])
    axes[1, 0].plot(cv_results['fold'], gen_error, 'g-o')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('Generalization Error (Test - Train)')
    axes[1, 0].set_title('Generalization Error Across Folds')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Box plot of losses
    loss_data = [cv_results['train_loss'], cv_results['test_loss']]
    axes[1, 1].boxplot(loss_data, labels=['Train', 'Test'])
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Loss Distribution Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def save_cv_results(cv_results, analysis, output_path):
    """
    Save cross-validation results to files
    
    Parameters:
    -----------
    cv_results : dict
        Cross-validation results
    analysis : dict
        Analysis results
    output_path : str
        Base path for saving results
    """
    # Save detailed results
    results_df = pd.DataFrame({
        'fold': cv_results['fold'],
        'test_classes': [str(classes) for classes in cv_results['test_classes']],
        'train_loss': cv_results['train_loss'],
        'test_loss': cv_results['test_loss']
    })
    
    # Add model parameters
    param_names = list(cv_results['model_params'][0].keys())
    for param in param_names:
        results_df[f'param_{param}'] = [params[param] for params in cv_results['model_params']]
    
    # Save to CSV
    csv_path = output_path.replace('.h5', '.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Detailed results saved to: {csv_path}")
    
    # Save to HDF
    results_df.to_hdf(output_path, key='cv_results', mode='w')
    
    # Save analysis summary
    analysis_df = pd.DataFrame([{
        'metric': 'mean_train_loss',
        'value': analysis['mean_train_loss'],
        'std': analysis['std_train_loss']
    }, {
        'metric': 'mean_test_loss',
        'value': analysis['mean_test_loss'],
        'std': analysis['std_test_loss']
    }, {
        'metric': 'mean_generalization_error',
        'value': analysis['mean_generalization_error'],
        'std': np.std(np.array(cv_results['test_loss']) - np.array(cv_results['train_loss']))
    }])
    
    analysis_path = output_path.replace('.h5', '_analysis.csv')
    analysis_df.to_csv(analysis_path, index=False)
    print(f"Analysis summary saved to: {analysis_path}")

def normalize_loss_across_model(loss_df, add_mean=True, match_broderick=True):
    """
    Normalize loss values for each subject for model comparisonby subtracting subject-specific means
    and optionally adding back the global mean or scaling to match Broderick et al.

    Parameters
    ----------
    loss_df : pd.DataFrame
        DataFrame containing loss values with columns 'subj', 'loss'
    add_mean : bool, optional
        If True, adds back the mean loss across all subjects after normalization.
        Default is True.
    match_broderick : bool, optional
        If True, scales the normalized loss values by a factor of 8 to match
        Broderick et al. Default is True.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns:
        - normalized_loss: Loss values normalized 
    """
    subj_mean_df = loss_df.groupby('subj').mean().rename(columns={'loss': 'mean_loss'})
    mean_across_all = loss_df['loss'].mean()
    loss_df = pd.merge(loss_df, subj_mean_df, on='subj', how='right')
    loss_df['normalized_loss'] = loss_df['loss'] - loss_df['mean_loss']
    if add_mean:
        loss_df['normalized_loss'] = loss_df['normalized_loss'] + mean_across_all
    if match_broderick:
        loss_df['normalized_loss'] = loss_df['normalized_loss']*8
    return loss_df

def plot_model_comparison(loss_df, 
                          x='model_type', 
                          y='normalized_loss', 
                          hue='subj', 
                          ylim=None,
                          xlim=None,
                          ax=None,
                          save_path=None,
                          rc=None,
                          orient='h',
                          **kwargs):
    """
    Plot model comparison
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=72*3)
    sns.pointplot(data=loss_df,
                  ax=ax, 
                  x=x,
                  y=y, 
                  hue=hue,
                  orient=orient,
                  **kwargs)
    ax.set_xlabel('Normalized loss')

    #ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    #ax.set_title('7-fold cross-validation results')
    if hue is None or hue is x or hue is y:
        ax.get_legend().remove()
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

    if ylim is not None:
        ax.set(ylim=ylim)
    if xlim is not None:
        ax.set(xlim=xlim)
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    if save_path:
        utils.save_fig(save_path)

def show_model_type(data=None, ax=None):
    ax.tick_params(axis='x', bottom=False, top=False)
    ax.tick_params(axis='y', left=False, right=False)
    # Move y-tick labels closer to the axis by adjusting labelpad
    ax.yaxis.set_tick_params(pad=0)
    ax.xaxis.set_tick_params(pad=0)
    sns.set_context("notebook")
    if data is None:
        # Create data for heatmap
        x = np.linspace(0, 1, 9)  # 9 points along x-axis
        y = np.linspace(1, 7, 7)  # 7 points along y-axis
        X, Y = np.meshgrid(x, y)  # Create 2D grid
        data =np.tile(y[:, np.newaxis], (1, len(x)))  # Create data that only depends on y value
        data[5, 5:7] = np.nan
        data[4, 7:] = np.nan
        data[3, 5:] = np.nan
        data[2, 3:] = np.nan
        data[1, [1, 3, 4, 5, 6, 7, 8]] = np.nan
        data[0, [2, 3, 4, 5, 6, 7, 8]] = np.nan
    tab10_palette_7 = sns.color_palette("tab10", 7)

            
    custom_palette = sns.color_palette([
        (0.50, 0.50, 0.50),
        (0.80, 0.73, 0.47),
        (0.22, 0.42, 0.69),
        (0.80, 0.47, 0.65),
        (0.80, 0.20, 0.20),
        (0.59, 0.45, 0.34),
        (0.42, 0.25, 0.63)
    ])


    # Convert 0 values to NaN so they show as white
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=72*3)
    sns.heatmap(data, 
                cmap=custom_palette,
                xticklabels=range(1,10),
                yticklabels=range(1,8),
                square=False,
                cbar=False,
                ax=ax)
    # Add black grid lines
    for i in range(data.shape[0] + 1):
        ax.axhline(i, color='w', alpha=0.5, lw=1)
    for i in range(data.shape[1] + 1):
        ax.axvline(i, color='w', alpha=0.5, lw=1)
    # Decrease the space between ticklabels and axis
    ax.tick_params(axis='x', pad=0)
    ax.set_xticklabels([r"$\sigma$", r"$m$", r"$b$", r"$p_1$", r"$p_2$", r"$p_3$", r"$p_4$", r"$A_1$", r"$A_2$"], ha='center')
    ax.set_yticklabels([f'Model {i}' for i in range(1,len(data)+1)], rotation=0)
    ax.tick_params(axis='x', labeltop=False, labelbottom=True)


def plot_precision_weighted_avg_parameter(df, params, hue, hue_order, ax, ylim=None, yticks=None, pal=None, **kwargs):
    sns.set_theme("paper", style='ticks', rc=rc)

    tmp = group_params(df, params, [1]*len(params))
    tmp = tmp.query('params in @params')
    tmp['value_and_weights'] = tmp.apply(lambda row: row.value + row.precision * 1j, axis=1)
    tmp['params'] = _change_params_to_math_symbols(tmp['params'])
    g = sns.pointplot(data=tmp,
                      x='params', y='value_and_weights',
                      hue=hue, hue_order=hue_order,
                      palette=pal, linestyles='',
                      estimator=weighted_mean, errorbar=("ci", 68),
                      dodge=0.23,
                      ax=ax, **kwargs)
    g.set(ylabel='Parameter estimates', xlabel=None)
    if 'p_' in params[0] or 'A_' in params[0] or 'sigma' in params[0]:
        g.tick_params(axis='x', labelsize=rc['axes.labelsize'], pad=5)
    else:
        g.tick_params(axis='x', pad=5)
    if ylim is not None:
        g.set(ylim=ylim)
    if yticks is not None:
        g.set(yticks=yticks)

    ticks = [t.get_text() for t in g.get_xticklabels()]
    if any('p_' in s for s in ticks) or any('A_' in s for s in ticks):
        g.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.9, zorder=0)

    return g
def _replace_param_names_with_latex(params_list):
    """
    Replace entries in params_list with LaTeX-formatted names for plotting.
    """
    new_list = {
        'sigma': r"$\sigma$",
        'slope': r"$Slope$" "\n" r"$m$",
        'intercept': r"$Intercept$" "\n" r"$b$",
        'p_1': r"$p_1$",
        'p_2': r"$p_2$",
        'p_3': r"$p_3$",
        'p_4': r"$p_4$",
        'A_1': r"$A_1$",
        'A_2': r"$A_2$"
    }
    return [[new_list.get(param, param) for param in sublist] for sublist in params_list]

def plot_model_comparison_params(model_df,                                 params_list,
                                 hue='sub',
                                 fig=None, axes=None, 
                                 save_path=None, 
                                 weighted_average=False, 
                                 ylim=None, yticks=None,
                                 **kwargs):
    """
    Plot model parameters
    """ 
    # Get columns that are not in params_list
    flat_params_list = [param for sublist in params_list for param in sublist]
    non_param_columns = [col for col in model_df.columns if col not in flat_params_list]
    model_long_df = model_df.melt(id_vars=non_param_columns,
                                  var_name='param', 
                                  value_name='value')
    if weighted_average:
        model_long_df['value_and_weights'] = model_long_df.apply(lambda row: row.value + row.precision * 1j, axis=1)
        kwargs['estimator'] = weighted_mean
        y = 'value_and_weights'
    else:
        y = 'value'
    model_long_df['param'] = _change_params_to_math_symbols(model_long_df['param'])
    params_list = _replace_param_names_with_latex(params_list)
    if axes is None:
        fig, axes = plt.subplots(1,len(params_list), figsize=(7, 5), 
                                 gridspec_kw={'width_ratios': [1,2,1.5,1.5,1.5]})
    for i, ax in enumerate(axes.flatten()):
        tmp_param = params_list[i]
        tmp = model_long_df.query(f'param in @tmp_param')
        
        sns.pointplot(ax=ax, data=tmp, linestyles='',
                      x='param', y=y, scale=1, 
                      errorbar=('ci', 68),
                      order=params_list[i], 
                      hue=hue, dodge=0.5,
                      **kwargs)
        
        ax.set_xlabel('')
        ax.get_legend().remove()
        if i >= 2:
            ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.9, zorder=0)
        if i == 1:    
            ax.margins(x=0.05)
        if i == 0:
            ax.set_ylabel('Parameter estimates')
        else:
            ax.set_ylabel('')

    if ylim is not None:
        for i,ax in enumerate(axes.flatten()):
            ax.set(ylim=ylim[i])
    if yticks is not None:
        for i,ax in enumerate(axes.flatten()):
            ax.set(yticks=yticks[i])
    plt.tight_layout()
    
    ax.legend(loc='center left', title='Model type', bbox_to_anchor=(1.02, 0.7), frameon=False)
    plt.subplots_adjust(right=0.9)
    if save_path:
        utils.save_fig(save_path)
