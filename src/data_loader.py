import pandas as pd
import os

here = os.path.dirname(os.path.realpath(__file__))

key_pts_frame = pd.read_csv(os.path.join(here, '../data/training_frames_keypoints.csv'))

n = 0
image_name = key_pts_frame.iloc[n, 0]
key_pts = key_pts_frame.iloc[n, 1:].as_matrix()
key_pts = key_pts.astype('float').reshape(-1, 2)

print('Image name: ', image_name)
print('Landmarks shape: ', key_pts.shape)
print('First 4 key pts: {}'.format(key_pts[:4]))
