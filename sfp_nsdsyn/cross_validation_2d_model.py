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
import warnings
warnings.filterwarnings("ignore")

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


def run_cross_validation(df, sfp_model,
                         n_folds=7, n_test_classes=4, 
                         learning_rate=1e-4, max_epoch=1000, 
                         print_every=100, save_path=None,
                         loss_all_voxels=False, random_state=42):
    
    train_classes_list, test_classes_list = create_train_test_split(df.class_idx.unique(), 
                                                                    n_folds = n_folds, 
                                                                    n_test_classes = n_test_classes, 
                                                                    random_state=random_state)

    # Store results
    cv_results = {
        'fold': [],
        'test_classes': [],
        'train_loss': [],
        'test_loss': [],
        'train_losses_per_voxel': [],
        'test_losses_per_voxel': [],
        'model_params': [],
        'loss_history': [],
        'model_history': []
        }
    
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
                                                         loss_all_voxels=loss_all_voxels)
        # Evaluate on training/testing data
        train_loss, train_losses_per_voxel = evaluate_model(sfp_model, train_dset)
        test_loss, test_losses_per_voxel = evaluate_model(sfp_model, test_dset)
        
        # Get final model parameters
        final_params = {}
        for name, param in sfp_model.named_parameters():
            if param.requires_grad:
                final_params[name] = param.detach().numpy().item()
        
        # Store results
        cv_results['fold'].append(fold)
        cv_results['test_classes'].append(test_classes_list[fold])
        cv_results['train_loss'].append(train_loss)
        cv_results['test_loss'].append(test_loss)
        cv_results['train_losses_per_voxel'].append(train_losses_per_voxel)
        cv_results['test_losses_per_voxel'].append(test_losses_per_voxel)
        cv_results['model_params'].append(final_params)
        cv_results['loss_history'].append(loss_history)
        cv_results['model_history'].append(model_history)
        
        print(f"Train loss: {train_loss:.4f}")
        print(f"Test loss: {test_loss:.4f}")
    
    return cv_results



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

def plot_cv_results_group(all_df, save_path=None):
    """
    Plot cross-validation results
    """
    from matplotlib import gridspec
    sns.set_context("notebook")
    sample_df = all_df.melt(id_vars=['sub','fold','model_params'], 
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
                data=sample_df, x='loss_type', y='loss', hue='loss_type')
    # Plot 2: Train vs Test loss across subjects for each fold
    sns.barplot(ax=axes[0, 1], 
                data=sample_df, 
                x='fold', y='loss', 
                hue='loss_type', 
                errorbar=('ci', 68))
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


def plot_model_params(model_df, params_list, ax=None, hue='sub', save_path=None):
    """
    Plot model parameters
    """    
    model_long_df = model_df.melt(id_vars=['sub','fold'],
                                  var_name='param_name', 
                                  value_name='value')
    
    if ax is None:
        fig, axes = plt.subplots(1,len(params_list), figsize=(9, 3), 
                                 gridspec_kw={'width_ratios': [1,2,1.5,1.5,1.5]})
    for i, ax in enumerate(axes.flatten()):
        tmp_param = params_list[i]
        tmp = model_long_df.query(f'param_name in @tmp_param')
        sns.pointplot(ax=ax, data=tmp, linestyles='',
                    x='param_name', y='value', 
                    order=params_list[i], hue=hue,
                    palette=sns.color_palette("Set2"), dodge=True)
        ax.set_title(params_list[i])
        ax.set_xlabel('')
        ax.get_legend().remove()
    plt.tight_layout()
    
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.7), frameon=False)
    plt.subplots_adjust(right=0.9)
    if save_path:
        utils.save_fig(save_path)

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

def main():
    """Main function to run cross-validation"""
    # Configuration
    output_dir = '/Volumes/server/Projects/sfp_nsd/derivatives'
    dset = 'nsdsyn'
    subj = 'subj01'
    roi = 'V1'
    vs = 'pRFsize'
    
    # Cross-validation parameters
    n_folds = 10
    n_test_classes = 4
    learning_rate = 0.001
    max_epochs = 1000
    print_every = 100
    random_state = 42
    
    # Output paths
    output_path = os.path.join(output_dir, 'cv_results', 
                              f'cv_results_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.h5')
    plot_path = os.path.join(output_dir, 'cv_results', 
                            f'cv_plot_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.png')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Loading data...")
    df = load_data(output_dir, dset, subj, roi, vs)
    print(f"Loaded data with {len(df)} samples")
    print(f"Number of unique classes: {df['class_idx'].nunique()}")
    print(f"Number of unique voxels: {df['voxel'].nunique()}")
    
    print("\nRunning cross-validation...")
    cv_results = run_cross_validation(
        df, 
        n_folds=n_folds,
        n_test_classes=n_test_classes,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        print_every=print_every,
        random_state=random_state
    )
    
    print("\nAnalyzing results...")
    analysis = analyze_cv_results(cv_results)
    
    print("\nPlotting results...")
    plot_cv_results(cv_results, analysis, save_path=plot_path)
    
    print("\nSaving results...")
    save_cv_results(cv_results, analysis, output_path)
    
    print("\nCross-validation completed successfully!")

if __name__ == "__main__":
    main() 