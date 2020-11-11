# %%
import pandas as pd
import os

path = './feature_importance_topK/'
files = os.listdir(path)
feature_names = []
for file in files:
    data = pd.read_csv(path + file)
    feature_names.extend(list(data['column'].values))
feature_names.append('id')
feature_names = set(feature_names)
print(len(feature_names))
features = pd.read_csv('../feature.csv')
features_merge = features[feature_names]
features_merge.to_csv('../feature_merge.csv', index=False)
# %%
