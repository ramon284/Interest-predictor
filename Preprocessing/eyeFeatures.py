import pandas as pd
import numpy as np

class eyePreprocessor:
    
    def get_EAR(self, df):
        epsilon = 1e-2
        df['eyeWidth'] = df['x39'] - df['x36']
        df['eyeHeight'] = df[['y40', 'y41']].max(axis=1) - df[['y37', 'y38']].min(axis=1)
        df['EAR'] = round(df['eyeWidth'] / (df['eyeHeight'] + epsilon), 3)        
        mean_EAR = df.groupby('Person')['EAR'].mean()
        mean_EAR_df = mean_EAR.reset_index().rename(columns={'EAR': 'mean_EAR'})
        df = df.merge(mean_EAR_df, on='Person')

        
        return df
    
    def pupilFeatures(self, df):
        mean_pupil_ratio = df.groupby('Person')['pupil_ratio'].mean()
        # Create a DataFrame with the mean pupil_ratio for each person
        mean_pupil_ratio_df = mean_pupil_ratio.reset_index().rename(columns={'pupil_ratio': 'mean_pupil_ratio'})
        # Merge the mean_pupil_ratio_df with the original DataFrame on 'Person'
        df = df.merge(mean_pupil_ratio_df, on='Person')
        # Create a new 'looking_away' column based on the deviation from the mean_pupil_ratio
        df['looking_away'] = abs(df['pupil_ratio'] - df['mean_pupil_ratio']) > 0.10
        return df

