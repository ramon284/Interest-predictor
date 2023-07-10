import pandas as pd
import numpy as np

class emotionMirroring:
            
    def calculate_mirroring(self, emotions):
        n_people = len(emotions)
        if n_people <= 1:
            return [0] * n_people

        emotion_keys = list(emotions[0].keys())
        mirroring_scores = [0] * n_people
        
        for i in range(n_people):
            max_emotion_i = max(emotions[i], key=emotions[i].get)
            for j in range(n_people):
                if i == j:
                    continue
                max_emotion_j = max(emotions[j], key=emotions[j].get)
                if max_emotion_i == max_emotion_j:
                    similarity = min(emotions[i][max_emotion_i], emotions[j][max_emotion_j])
                    mirroring_scores[i] += similarity
            mirroring_scores[i] /= (n_people - 1)
            mirroring_scores[i] = round(mirroring_scores[i], 3)  # Add this line to round the scores

        
        return mirroring_scores
    
    def get_emotions(self, df):
        return [
            {
                "anger": row["anger"],
                "disgust": row["disgust"],
                "fear": row["fear"],
                "happiness": row["happiness"],
                "sadness": row["sadness"],
                "surprise": row["surprise"],
                "neutral": row["neutral"],
            }
            for _, row in df.iterrows()
        ]
        
    def runModel(self, df_old):
        df = df_old.copy(deep=True)
        mirroring_scores = []
        unique_frames = df['Frame'].unique()
        for frame in unique_frames:
            frame_data = df[df['Frame'] == frame]
            emotions = self.get_emotions(frame_data)
            mirroring_scores.extend(self.calculate_mirroring(emotions))

        # Add the mirroring scores as a new column to t
        df['emotion_mirroring'] = mirroring_scores
        return df
