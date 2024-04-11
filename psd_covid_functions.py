import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import probplot, boxcox
from scipy.stats import zscore
from scipy.stats import pearsonr
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def unistats(df):
    
    output_df = pd.DataFrame(columns=['Mode', 'Mean', 'Min', 
                                      '25%', 'Median', '75%', 'Max', 
                                      'Std Dev', 'Skew', 'Kurt', 'Outliers (IQR)', 'Outliers (z-score)'])
    
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            
            # calculate z-scores
            z_scores = zscore(df[col].dropna())
            outliers_z_score = (abs(z_scores) > 3).sum()

            std_dev = df[col].std()
            mean = df[col].mean()
            cv = std_dev / mean  # Coefficient of Variation
            
            output_df.loc[col] = [df[col].mode().values[0], df[col].mean(), 
                                  df[col].min(), df[col].quantile(0.25), 
                                  df[col].median(), df[col].quantile(0.75),
                                  df[col].max(), cv, 
                                  df[col].skew(), df[col].kurt(), outliers, outliers_z_score]
        else:
            output_df.loc[col] = [df[col].mode().values[0], '-', '-', '-', '-', 
                                  '-', '-', '-', '-', '-', '-', '-']
    
    return output_df


def calculate_correlation(df, var1, var2):
    # Filter DataFrame based on 'NICU Stay'
    df_without_nicu = df[df['NICU Stay'] == 'Without NICU stay']
    df_with_nicu = df[df['NICU Stay'] == 'With NICU stay']

    # Calculate correlation and p-value for data without NICU stay
    correlation_without, p_value_without = pearsonr(df_without_nicu[var1], df_without_nicu[var2])
    correlation_without = round(correlation_without, 5)
    p_value_without = round(p_value_without, 5)
    print(f'{var1} and {var2}, without NICU: correlation: {correlation_without}, p-value: {p_value_without}')

    # Calculate correlation and p-value for data with NICU stay
    correlation_with, p_value_with = pearsonr(df_with_nicu[var1], df_with_nicu[var2])
    correlation_with = round(correlation_with, 5)
    p_value_with = round(p_value_with, 5)
    print(f'{var1} and {var2}, with NICU: correlation: {correlation_with}, p-value: {p_value_with}')

def perform_tukey_test(df, continuous_var, group_var):
    # Conduct Tukey HSD test
    tukey_result = pairwise_tukeyhsd(df[continuous_var], df[group_var])

    # Convert Tukey HSD result to DataFrame
    tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])

    # Filter results for p-adj < 0.05
    significant_results = tukey_df[tukey_df['p-adj'] < 0.05]

    print(f'Significant Tukey HSD test results for {continuous_var}:\n{significant_results}\n')


