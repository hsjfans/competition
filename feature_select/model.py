# %%
import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cab
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support
from matplotlib import pyplot as plt
from multiprocessing import Process

# from featexp import get_univariate_plots#用于特征筛选，需要先安装featexp
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False


if not os.path.exists('../feature_importance/'):
    os.mkdir('../feature_importance/')

if not os.path.exists('../prediction/'):
    os.mkdir('../prediction/')


def eval_score(y_test, y_pre):
    _, _, f_class, _ = precision_recall_fscore_support(
        y_true=y_test, y_pred=y_pre, labels=[0, 1], average=None)
    fper_class = {'合法': f_class[0], '违法': f_class[1],
                  'f1': f1_score(y_test, y_pre)}
    return fper_class


def k_fold_serachParmaters(model, train_val_data, train_val_kind):
    mean_f1 = 0
    mean_f1Train = 0
    n_splits = 5
    sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)
    for train, test in sk.split(train_val_data, train_val_kind):
        x_train = train_val_data.iloc[train]
        y_train = train_val_kind.iloc[train]
        x_valid = train_val_data.iloc[test]
        y_valid = train_val_kind.iloc[test]

        model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        fper_class = eval_score(y_valid, pred)
        mean_f1 += fper_class['f1'] / n_splits

        pred_Train = model.predict(x_train)
        fper_class_train = eval_score(y_train, pred_Train)
        mean_f1Train += fper_class_train['f1'] / n_splits
    return mean_f1


def cat_model(iter_cnt, lr, max_depth, cat_features):
    clf = cab.CatBoostClassifier(iterations=iter_cnt,
                                 learning_rate=lr,
                                 depth=max_depth,
                                 silent=True,
                                 thread_count=8,
                                 task_type='CPU',
                                 cat_features=cat_features
                                 )
    return clf


def cat_serachParm(train_data, train_labels, cat_features):
    print('cat_serachParm 搜索最佳参数 .......')
    # 搜索最佳参数
    param = []
    best = 0
    best_model = None
    #
    # 55, 60, 70, 80
    for iter_cnt in [55, 60, 70, 80]:
        print('iter_cnt:', iter_cnt)
        for lr in [0.03, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065]:
            for max_depth in [5, 6, 7, 8]:
                clf = cat_model(iter_cnt, lr, max_depth, cat_features)
                mean_f1 = k_fold_serachParmaters(clf,
                                                 train_data, train_labels)
                if mean_f1 > best:
                    param = [iter_cnt, lr, max_depth]
                    best = mean_f1
                    best_model = clf
                    print(param, best)
    print('cat_serachParm 搜索最佳参数 ....... over', param, best)
    return best_model, param, best


def rf_model(n_estimators, max_depth, min_samples_split):
    rf = RandomForestClassifier(oob_score=True, random_state=2020,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split)
    return rf


def rf_searchParam(train_data, train_labels):
    print('rf_searchParam 搜索最佳参数 .......')
    # 搜索最佳参数
    param = []
    best = 0
    best_model = None
    for n_estimators in [50, 55, 57, 60, 65]:
        print('n_estimators:', n_estimators)
        for min_samples_split in [6, 8, 10, 13, 15, 17, 20]:
            for max_depth in [11, 12, 13, 15]:
                rf = rf_model(n_estimators, max_depth, min_samples_split)
                mean_f1 = k_fold_serachParmaters(
                    rf, train_data, train_labels)
                if mean_f1 > best:
                    param = [n_estimators, min_samples_split, max_depth]
                    best = mean_f1
                    best_model = rf
                    print(param, best)
    print('rf_searchParam 搜索最佳参数 ....... over', param, best)
    return best_model, param, best


def lgb_model(n_estimators, max_depth, num_leaves, learning_rate):
    lgb_model = lgb.LGBMClassifier(objective='binary',
                                   boosting_type='gbdt',
                                   tree_learner='serial',
                                   num_leaves=num_leaves,
                                   max_depth=max_depth,
                                   learning_rate=learning_rate,
                                   n_estimators=n_estimators,
                                   subsample=0.8,
                                   feature_fraction=0.8,
                                   reg_alpha=0.3,
                                   reg_lambda=0.5,
                                   random_state=2020,
                                   is_unbalance=True)
    return lgb_model


def lgb_searchParam(train_data, train_labels):
    print('lgb_searchParam 搜索最佳参数 .......')
    # 搜索最佳参数
    param = []
    best = 0
    best_model = None
    # 40, 45, 50, 55, 60, 65, 70
    for n_estimators in [40, 45, 50, 55, 60, 65, 70]:
        print('n_estimators:', n_estimators)
        for max_depth in [6, 7, 8]:
            for num_leaves in [40, 45, 50, 55, 60, 65, 70]:
                for learning_rate in [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25]:
                    lgb_cl = lgb_model(n_estimators, max_depth,
                                       num_leaves, learning_rate)
                    mean_f1 = k_fold_serachParmaters(
                        lgb_cl, train_data, train_labels)
                    if mean_f1 > best:
                        param = [n_estimators, max_depth,
                                 num_leaves, learning_rate]
                        best = mean_f1
                        best_model = lgb_cl
                        print(param, best)
    print('lgb_searchParam 搜索最佳参数 ....... over', param, best)
    return best_model, param, best


def predict(name, model, train_data, train_label, test_data, merge=False, n_splits=5, shuffle=True, random_state=2020):
    mean_f1 = 0
    answers = []
    feature_importance_list = []
    merge_name = ''
    test = test_data.drop('id', axis=1)
    if merge:
        merge_name = 'merge'
    sk = StratifiedKFold(n_splits=n_splits,
                         shuffle=True, random_state=2020)
    for i, (train, valid) in enumerate(sk.split(train_data, train_label)):
        x_train = train_data.iloc[train]
        y_train = train_label.iloc[train]
        x_valid = train_data.iloc[valid]
        y_valid = train_label.iloc[valid]
        model.fit(x_train, y_train)
        pred_cab = model.predict(x_valid)
        f1_score_ = eval_score(y_valid, pred_cab)['f1']
        print('model = {} 第{}次验证的f1:{}'.format(model, i + 1, f1_score_))
        feature_importance = pd.DataFrame({
            'column': train_data.columns,
            'importance': model.feature_importances_,
        })
        feature_importance_list.append(feature_importance)
        mean_f1 += f1_score_ / n_splits
        ans = model.predict_proba(test)
        answers.append(ans)
    print('mean f1:', mean_f1)
    feature_importances = pd.concat(feature_importance_list)
    feature_importances = feature_importances.groupby(
        'column')['importance'].agg('mean').sort_values(ascending=False).reset_index()
    feature_importances.to_csv(
        f'../feature_importance/{name}_{merge_name}_feature_importance.csv')
    prediction = np.sqrt(sum(np.array(answers)**2) / n_splits)[:, 1]
    predict_res = pd.DataFrame(
        {'id': test_data['id'],
         "score": prediction}
    )
    predict_res.to_csv(f'../prediction/{name}_{merge_name}_prediction.csv')
    return prediction, feature_importances


def get_fearures():
    feature = pd.read_csv(
        '../feature.csv')
    entprise_info = pd.read_csv('../data/train/entprise_info.csv')
    data = pd.merge(feature, entprise_info, how='left', on='id')
    cat_features = ['oplocdistrict', 'industryphy', 'industryco', 'enttype', 'enttypeitem',
                    'state', 'orgid', 'jobid', 'adbusign', 'townsign', 'regtype', 'compform', 'opform', 'venind', 'oploc',  'enttypegb', 'industryphy_industryco',
                    'enttypegb_enttypeitem', 'nan_num_bin', 'regcap_bin', 'empnum_bin'
                    ]
    data[cat_features].astype(int)
    # print(data.max())
    train = data[data.label.notna()]
    test = data[data.label.isnull()]

    train_data, train_labels = train.drop(
        ['id', 'label'], axis=1), train['label']
    test_data = test.drop(
        ['label'], axis=1)

    return train_data, train_labels, test_data, cat_features


def get_topK_features(feature_importance_list, k=20):
    topk_feature_names = []
    for feature_importance in feature_importance_list:
        topk_feature_names.extend(
            list(feature_importance['column'].values[:20]))
    topk_feature_names = set(topk_feature_names)
    return topk_feature_names


def train(train_data, train_labels, test_data, cat_features, merge=False):
    merge_name = ''
    if merge:
        merge_name = 'merge'

    best_cat = cat_serachParm(train_data, train_labels, cat_features)

    cat_score, cat_feature_importances = predict('cat',
                                                 best_cat, train_data, train_labels, test_data, merge=merge)
    best_rf = rf_searchParam(train_data, train_labels)
    rf_score, rf_feature_importances = predict('rf',
                                               best_rf, train_data, train_labels, test_data, merge=merge)
    best_lgb = lgb_searchParam(train_data, train_labels)
    lgb_score, lgb_feature_importances = predict('lgb',
                                                 best_lgb, train_data, train_labels, test_data, merge=merge)

    final_score = (rf_score + cat_score + lgb_score) / 3.0
    test_data['score'] = final_score  # 可选:fina_persudo是伪标签的预测结果
    submit_csv = test_data[['id', 'score']]
    submit_csv.to_csv(f'../submit_{merge_name}.csv', index=False)
    return cat_feature_importances, rf_feature_importances, lgb_feature_importances


if __name__ == "__main__":
    train_data, train_labels, test_data, cat_features = get_fearures()
    cat_feature_importances, rf_feature_importances, lgb_feature_importances = train(
        train_data, train_labels, test_data, cat_features)
    topk_feature_names = get_topK_features(
        [cat_feature_importances, rf_feature_importances, lgb_feature_importances])
    train_data = train_data[topk_feature_names]
    new_cat_features = topk_feature_names.intersection(cat_features)
    topk_feature_names.add('id')
    train(train_data,
          train_labels, test_data[topk_feature_names], new_cat_features, merge=True)

    # %%