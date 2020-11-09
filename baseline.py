# %%
from sklearn.impute import KNNImputer
import pandas as pd


base_info = pd.read_csv('./data/train/base_info.csv')
nums, shapes = base_info.shape
# 1. 删掉缺失值较多的特征
for name, cout in base_info.isnull().sum().items():
    if cout * 1.0 / nums >= 0.7:
        print(name, cout)
        del base_info[name]

# 删除类别不同较多的列
for name, cout in base_info.nunique().items():
    if (cout == 1 or cout * 1.0 / nums >= 0.8) and name != 'id':
        print(name, cout)
        del base_info[name]

print(base_info.info())


def knn_fill_nan(X):

    imputer = KNNImputer(n_neighbors=5, weights="distance")
    return imputer.fit_transform(X)


category_cols = ['oplocdistrict', 'industryphy', 'industryphy', 'enttype',
                 'enttypeitem', 'adbusign', 'adbusign', 'regtype', 'compform', 'enttypegb', 'opform']
# 使用 industryphy 类别中数填充 industryco
print(base_info['industryco'].isnull().sum())
grouped = base_info[['industryco', 'industryphy']].groupby('industryphy')
for name, group in grouped:
    median = group.median()
    base_info.loc[base_info['industryphy'] == name, 'industryco'] = base_info[base_info['industryphy']
                                                                              == name]['industryco'].fillna(median.values[0])
print(base_info['industryco'].isnull().sum())
# 使用 enttype 类别填充 enttypeitem
print(base_info['enttypeitem'].isnull().sum())
grouped = base_info[['enttype', 'enttypeitem']].groupby('enttype')
for name, group in grouped:
    base_info.loc[base_info['enttype'] == name,
                  'enttypeitem'] = base_info[base_info['enttype'] == name]['enttypeitem'].fillna(name)
print(base_info['enttypeitem'].isnull().sum())
print(base_info[category_cols].isnull().sum())
print(base_info[category_cols].nunique())
# opform  compform 缺失值多 如何处理呢?
base_info['opform'] = base_info['opform'].replace(
    '01', '01-以个人财产出资').replace('02', '02-以家庭共有财产作为个人出资')
# 暂时删掉试一下啊
del base_info['opform']
del base_info['compform']


# 时间转换, 暂时先抽取年份特征

base_info['opfrom'] = pd.to_datetime(base_info.opfrom)
base_info['opfrom_year'] = base_info['opfrom'].dt.year.astype('int')

base_info['opto'] = pd.to_datetime(base_info.opto)
base_info['opto_year'] = base_info['opto'].dt.year.fillna(-1).astype('int')

del base_info['opfrom']
del base_info['opto']
del base_info['oploc']
del base_info['venind']

# 对数值类进行缺失处理, 采样聚类方法进行处理
category_cols = ['oplocdistrict', 'industryphy', 'industryphy', 'enttype',
                 'enttypeitem', 'regtype', 'enttypegb',  'orgid', 'jobid']


for col in category_cols:
    # 对类别信息采取 one-hot 编码
    base_info[col] = base_info[col].astype('category').cat.codes

# 按照桶填充缺失值
# onehot 编码填充数据
cols = ['oplocdistrict', 'industryphy', 'industryphy', 'enttype',
        'enttypeitem', 'regtype', 'enttypegb', 'townsign', 'adbusign']

base_info.to_csv('base_info_unfill_feature.csv', index=False)
data = base_info.drop('id', axis=1)
base_feaures = data.columns
for col in category_cols:
    # 对类别信息采取 one-hot 编码
    category = pd.get_dummies(
        data[col], prefix=col, drop_first=True)
    data[category.columns] = category
data_features = data.columns
data = knn_fill_nan(data)
features = pd.DataFrame(data, columns=data_features)
base_info[base_feaures] = features[base_feaures]
features.to_csv('one_hot_feature.csv', index=False)
base_info.to_csv('base_info_feature.csv', index=False)


# %%
