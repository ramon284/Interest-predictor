import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation

class preprocessorPredictions:
    
    def preprocess(self, df, humanCheck = False):
        listIntToBool = ['AU7', 'AU20', 'looking_away']
        for col in listIntToBool:
            df[col] = df[col].astype('bool')
            
        listBB = ['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight']
        listLandmarks = ['x' + str(i) for i in range(68)] + ['y' + str(i) for i in range(68)]
        listTime = ['Timestamp']
        
        listAUDiff = ['AU1_diff','AU2_diff','AU4_diff','AU5_diff','AU6_diff','AU7_diff','AU9_diff','AU10_diff','AU11_diff','AU12_diff',
                    'AU14_diff','AU15_diff','AU17_diff','AU20_diff','AU23_diff','AU24_diff','AU25_diff','AU26_diff','AU28_diff','AU43_diff']
        listEmoDiff = ['disgust_diff','fear_diff','happiness_diff', 'anger_diff', 'sadness_diff','surprise_diff','neutral_diff','emotion_mirroring_diff']
        
        listEyes = ['eyeWidth', 'eyeHeight', 'mean_EAR', 'mean_pupil_ratio']
        
        listAU = ['AU1','AU2','AU4','AU5','AU6','AU7','AU9','AU10','AU11',
                'AU12','AU14','AU15','AU17','AU20','AU23','AU24','AU25','AU26','AU28','AU43']
        
        listFaceDirection = ['Pitch', 'Yaw', 'Roll']
        
        featureTest = ['AU5', 'AU10', 'AU15', 'AU26', 'AU28', 'looking_away', 'AU1_diff', 'AU2_diff', 'AU4_diff', 'AU5_diff', 'AU6_diff', 'AU7_diff', 'AU9_diff', 'AU10_diff', 
                    'AU11_diff', 'AU12_diff', 'AU14_diff', 'AU15_diff', 'AU17_diff', 'AU20_diff', 'AU23_diff', 'AU24_diff', 'AU25_diff', 'AU26_diff', 'AU28_diff', 
                    'AU43_diff', 'anger_diff', 'disgust_diff', 'fear_diff', 'happiness_diff', 'surprise_diff', 'neutral_diff']
        ##      + listEmoDiff 
        listOther = ['emotion_mirroring_diff']
        
        totalcolumns = listBB + listLandmarks + listEyes + featureTest + listOther# + listFaceDirection
            
        df = df.drop(columns = totalcolumns)
        df = df.drop(columns=['pupil_direction', 'Person'])
        return df


    def generate_chunk_summary(self, chunk, global_statistics={}, fullList=[], meanSDList=[], humanCheck = False):
        chunk_df = pd.DataFrame(chunk)
        chunk_summary = chunk_df.mean().to_dict()  # calculate mean for all columns\n",
        for col in fullList:  # calculate quartiles and standard deviation for columns in fullList\n",
            chunk_summary.update({
                f'{col}_mean': chunk_df[col].mean(),
                f'{col}_Q1': chunk_df[col].quantile(0.25),
                f'{col}_Q2': chunk_df[col].median(),
                f'{col}_Q3': chunk_df[col].quantile(0.75),
                f'{col}_SD': chunk_df[col].std(),
                f'{col}_MAD': median_abs_deviation(chunk_df[col])
            })
        for col in meanSDList:  # calculate quartiles and standard deviation for columns in fullList\n",
            chunk_summary.update({
                f'{col}_mean': chunk_df[col].mean(),
                f'{col}_SD': chunk_df[col].std(),
            })    

        return chunk_summary


    def create_chunks(self, df, chunk_size, global_statistics={}, humanCheck = False):
        fullList = ['EAR']
        fewList = [] # 'Pitch', 'Yaw', 'Roll'
        df_new = pd.DataFrame()
        current_chunk = []
        start_of_probe = -1
        summaries = []

        for _, row in df.iterrows():
            if start_of_probe < 0 or (row['Frame'] - start_of_probe) >= 500:
                # Finalize the current chunk, if there is one
                if current_chunk:
                    chunk_summary = self.generate_chunk_summary(current_chunk, global_statistics, fullList, fewList, humanCheck)
                    summaries.append(chunk_summary)
                
                # Start a new probe and a new chunk
                start_of_probe = row['Frame']
                current_chunk = [row]
            elif len(current_chunk) == chunk_size:
                # We have a complete chunk, so finalize it
                chunk_summary = self.generate_chunk_summary(current_chunk, global_statistics, fullList, fewList, humanCheck)
                summaries.append(chunk_summary)
                current_chunk = [row]
            else:
                # Add the current row to the current chunk
                current_chunk.append(row)

        # Finalize the last chunk, if there is one
        if current_chunk:
            chunk_summary = self.generate_chunk_summary(current_chunk, global_statistics, fullList, fewList, humanCheck)
            summaries.append(chunk_summary)
        
        df_new = pd.DataFrame(summaries)
        return df_new

    def preprocess_and_chunk(self, filename, chunk_size, cols_full=[], humanCheck = False):
        df = pd.read_csv(filename)
        fullDataframe = []
        persons = df['Person'].unique()
        for person in persons:
            tempdf = df[df['Person'] == person].copy()  # 
            
            columns_of_interest = [] # Add more columns if needed
            global_statistics = {
                'mean': {col: tempdf[col].mean() for col in columns_of_interest},
                'sd': {col: tempdf[col].std() for col in columns_of_interest},
                'mad': {col: tempdf[col].mad() for col in columns_of_interest},
                'quartiles': {col: tempdf[col].quantile([0.25, 0.5, 0.75]).to_dict() for col in columns_of_interest}
            }    
            cols_full = []
            for col in cols_full:
                mean_full = tempdf[col].mean()
                tempdf[f'{col}_mean_full'] = mean_full
                tempdf[f'{col}_mean_diff'] = tempdf[col] - mean_full

            tempdf['Yaw_normalized'] = (tempdf['Yaw'] + 90) / 180   
            tempdf['combined_gaze'] = (tempdf['Yaw_normalized'] + tempdf['pupil_ratio']) / 2
            tempdf = tempdf.drop(columns='Yaw_normalized')

            tempdf['head_movement'] = tempdf[['Pitch', 'Yaw', 'Roll']].diff().abs().sum(axis=1)
            # df['Yaw'] = df['Yaw'].abs()
            # df['Roll'] = df['Roll'].abs()

            mean_value = tempdf['EAR'].iloc[:1000].mean()
            tempdf['calibrated_EAR'] = mean_value
            combined_gaze = tempdf['combined_gaze'].iloc[:1000].mean()
            tempdf['calibrated_combined_gaze'] = combined_gaze
            tempdf['gaze_deviation_from_start'] = abs(tempdf['combined_gaze'] - tempdf['calibrated_combined_gaze'])
            #df['pupil_deviation'] = abs(df['mean_pupil_ratio'] - df['pupil_ratio'])
            tempdf['EAR_deviation'] = abs(tempdf['EAR'] - tempdf['calibrated_EAR'])
            
            drop_columns = [f'{col}_mean_full' for col in cols_full]
            tempdf = tempdf.drop(columns=drop_columns)
            
            tempdf = self.preprocess(tempdf, humanCheck)
            df_chunks = self.create_chunks(tempdf, chunk_size, global_statistics, humanCheck)
            df_chunks = df_chunks.drop(columns=['calibrated_EAR', 'calibrated_combined_gaze'])
            df_chunks['Yaw'] = df_chunks['Yaw'].abs()
            df_chunks['Roll'] = df_chunks['Roll'].abs()    
            df_chunks = df_chunks.drop(columns=['Frame'])
            df_chunks = df_chunks.fillna(df_chunks.median())
            fullDataframe.append(df_chunks)
        return fullDataframe
    
    
    def nextStep(self, chunk_size=25, chunkBool=1, filename=''):
        kalman = False
        if kalman == True:
            kalman = '_kalman'
        else: kalman = ''
        fullDataFrame = self.preprocess_and_chunk(filename, chunk_size)
        #chunked_df = chunked_df.drop(columns=['Frame'])
        return fullDataFrame
    
    def fullPreProcess(self, filename=''):
        chunk_size = 25
        chunk_bool = 1
        fullDataFrame = self.nextStep(chunk_size, chunk_bool, filename)
        return fullDataFrame
