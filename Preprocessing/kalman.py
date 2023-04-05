import numpy as np
import pandas as pd
from pykalman import KalmanFilter

class filterKalman:
    def __init__(self, df, selected_columns, n_landmarks=68):
        self.df = df
        self.n_landmarks = n_landmarks
        self.n_dim_state = self.n_landmarks * 2
        self.n_dim_obs = self.n_landmarks * 2
        self.landmark_columns = selected_columns

    def runFilter(self):
        for person in self.df['Person'].unique():
            print('person: ',person)
            temp_df = self.df[self.df['Person'] == person].copy()
            initial_state_mean = np.zeros(self.n_dim_state)
            initial_state_covariance = np.eye(self.n_dim_state) * 1e4

            transition_matrices = np.eye(self.n_dim_state)
            transition_covariance = np.eye(self.n_dim_state) * 1e-1

            observation_matrices = np.eye(self.n_dim_obs)
            observation_covariance = np.eye(self.n_dim_obs) * 1e-1

            kf = KalmanFilter(
                transition_matrices=transition_matrices,
                observation_matrices=observation_matrices,
                initial_state_mean=initial_state_mean,
                initial_state_covariance=initial_state_covariance,
                transition_covariance=transition_covariance,
                observation_covariance=observation_covariance,
            )

            observations = temp_df[self.landmark_columns].values

            # Apply the Kalman filter
            filtered_state_means, filtered_state_covariances = kf.filter(observations)

            temp_df[self.landmark_columns] = filtered_state_means
            self.df.update(temp_df)

        self.df.to_csv('kalmanned.csv', index=False)
        return self.df