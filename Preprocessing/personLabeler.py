import numpy as np
import pandas as pd

class personPersistenceManual:
    def __init__(self, overlap_ratio=0.2, error_frames=10):
        self.overlap_ratio = overlap_ratio
        self.error_frames = error_frames

    def compute_overlap(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)

        overlap = interArea / float(boxAArea)

        return overlap

    def assign_person_id(self, df_orig):
        df = df_orig.copy(deep=True)
        df['Person'] = None
        next_id = 0

        person_last_seen = {}
        person_last_box = {}

        for frame, frame_data in df.groupby('Frame'):
            frame_boxes = frame_data[['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight']].values
            frame_boxes[:, 2] += frame_boxes[:, 0]
            frame_boxes[:, 3] += frame_boxes[:, 1]

            assigned = set()
            for i, row in frame_data.iterrows():
                box = frame_boxes[frame_data.index.get_loc(i)]
                best_overlap = 0
                best_person = None
                for person, last_box in person_last_box.items():
                    overlap = self.compute_overlap(box, last_box)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_person = person

                if best_overlap > self.overlap_ratio and best_person not in assigned:
                    df.loc[i, 'Person'] = best_person
                    person_last_seen[best_person] = frame
                    person_last_box[best_person] = box
                    assigned.add(best_person)
                else:
                    df.loc[i, 'Person'] = next_id
                    person_last_seen[next_id] = frame
                    person_last_box[next_id] = box
                    next_id += 1
                    assigned.add(next_id - 1)

            # Check for multiple labels in the same frame
            frame_grouped = df[df['Frame'] == frame].groupby('Person').size()
            multiple_labels = frame_grouped[frame_grouped > 1].index
            for person in multiple_labels:
                person_rows = df[(df['Frame'] == frame) & (df['Person'] == person)]
                min_dist = float('inf')
                best_index = None
                for i, row in person_rows.iterrows():
                    box = frame_boxes[frame_data.index.get_loc(i)]
                    dist = np.linalg.norm(np.array(person_last_box[person][:2]) - np.array(box[:2]))
                    if dist < min_dist:
                        min_dist = dist
                        best_index = i
                df.loc[person_rows.index.difference([best_index]), 'Person'] = 'x'

        return df

# Example usage
# df = pd.read_csv('your_dataframe.csv')
# ppm = personPersistenceManual()
# df_assigned = ppm.assign_person_id(df)
