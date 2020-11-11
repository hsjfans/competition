# %%
import pandas as pd
import os

if not os.path.exists('./prediction_merge/'):
    os.mkdir('./prediction_merge/')

path = './prediction/'
files = os.listdir(path)
scores = 0
nums = len(files)
prediction = None
for file in files:
    prediction_f = pd.read_csv(path + file)
    if prediction is None:
        prediction = prediction_f
    else:
        prediction['score'] = prediction_f['score'] + prediction['score']
prediction['score'] = prediction['score'] / nums
print(prediction.columns)
prediction.to_csv(os.path.join('./prediction_merge/',
                               'preditction_merge.csv'), index=False)
print('ok --------------------------------')
# %%
