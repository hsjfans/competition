# %%
import pandas as pd
from tqdm import tqdm

# #处理base_info数据
base_info = pd.read_csv('../data/train/base_info.csv')
# 把缺失数量作为一种编码
base_info_clean = base_info
base_info_clean['nan_num'] = base_info.isnull().sum(axis=1)

nums, shapes = base_info_clean.shape
# 删除缺失 75%以上的数据
for name, count in base_info_clean.isnull().sum().items():
    if count * 1.0 / nums >= 0.75:
        base_info_clean.drop([name], axis=1, inplace=True)

# 删除类别相同的数据
for name, count in base_info_clean.nunique().items():
    if count == 0:
        base_info_clean.drop([name], axis=1, inplace=True)


# 文本信息，采用jieba 分词处理
# 先去除掉括号里面的内容


# 日期信息，取到年份，缺失值采用-1填充


base_info_clean = base_info_clean.drop(['opscope', 'opfrom', 'opto'], axis=1)


# ............................对object类型进行编码...............................
base_info_clean['industryphy'] = base_info_clean['industryphy'].fillna("无")
base_info_clean['dom'] = base_info_clean['dom'].fillna("无")
base_info_clean['opform'] = base_info_clean['opform'].fillna("无")
base_info_clean['oploc'] = base_info_clean['oploc'].fillna("无")

dic = {}
cate = base_info_clean.industryphy.unique()
for i in range(len(cate)):
    dic[cate[i]] = i

buf = pd.DataFrame()
buf_group = base_info_clean.groupby('industryphy', sort=False)
for name, group in buf_group:
    group['industryphy'] = dic[name]
    buf = pd.concat([buf, group], ignore_index=True)
print('finished 1....')
#
dic = {}
cate = buf.dom.unique()
for i in range(len(cate)):
    dic[cate[i]] = i

buf_group = buf.groupby('dom', sort=False)
buf = pd.DataFrame()
for name, group in buf_group:
    group['dom'] = dic[name]
    buf = pd.concat([buf, group], ignore_index=True)
print('finished 2....')
#
dic = {}
cate = buf.opform.unique()
for i in range(len(cate)):
    dic[cate[i]] = i

buf_group = buf.groupby('opform', sort=False)
buf = pd.DataFrame()
for name, group in buf_group:
    group['opform'] = dic[name]
    buf = pd.concat([buf, group], ignore_index=True)
print('finished 3....')
#
dic = {}
cate = buf.oploc.unique()
for i in range(len(cate)):
    dic[cate[i]] = i

buf_group = buf.groupby('oploc', sort=False)
buf = pd.DataFrame()
for name, group in buf_group:
    group['oploc'] = dic[name]
    buf = pd.concat([buf, group], ignore_index=True)
print('finished 4....')

buf = buf.fillna(-1)

buf_group = buf.groupby('id', sort=False).agg('mean')
base_info_clean = pd.DataFrame(buf_group).reset_index()

print('编码完毕.................')

# %%
