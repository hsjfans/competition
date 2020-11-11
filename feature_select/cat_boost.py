# %%
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import os

if not os.path.exists('../feature_importance/'):
    os.mkdir('../feature_importance/')

if not os.path.exists('../prediction/'):
    os.mkdir('../prediction/')


def f1(y_true, y_pred):
    y_pred = y_pred.round()
    score = f1_score(y_true, y_pred)
    return "f1", score, True


base_info = pd.read_csv(
    '../../feature.csv')
entprise_info = pd.read_csv('../../data/train/entprise_info.csv')
data = pd.merge(base_info, entprise_info, how='left', on='id')
data = data.fillna(-1)

# print(data.max())
train = data[data.label.notna()]
test = data[data.label.isnull()]


label = 'label'
feature_names = list(
    filter(lambda x: x not in ['label', 'id'], train.columns))
model = CatBoostClassifier(iterations=54,
                           learning_rate=0.07,
                           depth=7,
                           silent=True,
                           thread_count=8,
                           random_state=2020,
                           use_best_model=True,
                           eval_metric='F1',
                           loss_function='Logloss',
                           task_type='CPU',
                           cat_features=feature_names,
                           )

prediction = 0.0
feature_importance_list = []
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
avg_score = 0.0
for fold_id, (train_idx, val_idx) in enumerate(kfold.split(train[feature_names], train[label])):

    X_train = train.iloc[train_idx][feature_names]
    Y_train = train.iloc[train_idx][label]

    X_val = train.iloc[val_idx][feature_names]
    Y_val = train.iloc[val_idx][label]

    print('\nFold_{} Training ================================\n'.format(fold_id + 1))

    cat_model = model.fit(X_train,
                          Y_train)

    y_val_pred = model.predict(
        X_val[feature_names])
    avg_score += f1(y_val_pred, Y_val)[1] / kfold.n_splits
    # 测试集合预测
    y_pred = model.predict(
        test[feature_names])

    feature_importance = pd.DataFrame({
        'column': feature_names,
        'importance': model.feature_importances_,
    })
    feature_importance_list.append(feature_importance)
    prediction += y_pred / kfold.n_splits
    # 网格搜索，参数优化
    # param_grid = {
    #     'learning_rate': [0.05, 0.02, 0.1, 0.2],
    #     'n_estimators': [20, 40, 80, 100],
    #     'num_leaves': [64, 128],
    #     'max_depth': [4, 6, 8]
    # }
    # gbm = GridSearchCV(model, param_grid)
    # gbm.fit(X_train, Y_train)
    # print('Best parameters found by grid search are:', gbm.best_params_)

feature_importances = pd.concat(feature_importance_list)
feature_importances = feature_importances.groupby(
    'column')['importance'].agg('mean').sort_values(ascending=False)
feature_importances.to_csv('../feature_importance/cat_feature_importance.csv')
predict = pd.DataFrame(
    {'id': test['id'],
     "score": prediction}
)
print(' gbm f1 avg_score =  ', avg_score)
predict.to_csv('../prediction/cat_prediction.csv', index=False)
# %%
