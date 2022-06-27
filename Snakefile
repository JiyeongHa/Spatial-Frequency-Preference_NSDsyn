import numpy as np
configfile:
    "config.json"
sub_list = [f"subj{str(i).zfill(2)}" for i in np.arange(1,9)]

rule load_subjects_dataframes:
    """ Load each subject's dataframes that contain beta, pRF properties for each stimulus type. """
    input:
        '{sn}.txt'
    output:
        '{sn}_df.csv'
    script:
        "binning_eccen.py"

rule combine_all_subject_dataframes:
    input:
        expand('{sn}_df.csv', sn=['subj01', 'subj05'])
    output:
        csv = 'all_subj_df.csv',
        fig = 'all_subj.png',
    run:
        import pandas as pd
        df = []
        for f in input:
            tmp = pd.read_csv(f)
            df.append(tmp)
        df = pd.concat(df)
        df.to_csv(output.csv)
        fig = ...
        fig.savefig(output.fig)

rule first_level_analysis:
    """ Run first level analysis for each subject """
    input:
        "all_subj_df" # extensions
    output:
        "output_df"
    script:
        "first_level_analysis.py"

rule merge_fitting_result_to_dataframe:
    """ Merge final parameters earned from the first level analysis to the dataframes """
    input:
        "all_subj_df",
        "output_df",
    output:
        "merged_df"
    script:
        "merge_dataframes.py"

rule plot_1D_model_fitting_results:
    """ Plot the first level analysis result averaged across subjects with actual data points """
    input:
        "merged_df",
    output:
        "1D_model_results.svg"
    script:
        "plot_1D_model_results.py"

rule linear_regression_for_preferred_period:
    """ Perform a linear regression to explain preferred period as a function of eccentricity """
    input:
        "merged_df"
    output:
        "merged_df_updated"
    script: # I hope I can use sub-functions here # run 497 lines
        "linear_regresson.py"

rule plot_preferred_period_vs_eccentricity:
    """ Plot preferred period for each eccentricity along with a linear fit"""
    input:
        "merged_df_updated"
    output:
        "preferred_period_results.svg"
    script: # I hope I can use sub-functions here
        "plot_preferred_period.py"


