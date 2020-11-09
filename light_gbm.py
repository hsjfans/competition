# %%
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


def f1(y_true, y_pred):
    y_pred = y_pred.round()
    return "f1", f1_score(y_true, y_pred), True


base_info = pd.read_csv('base_info_feature.csv')
entprise_info = pd.read_csv('./data/train/entprise_info.csv')
data = pd.merge(base_info, entprise_info, how='left', on='id')

# print(data.max())
train = data[data.label.notna()]
test = data[data.label.isnull()]


label = 'label'
feature_names = list(
    filter(lambda x: x not in ['label', 'id'], train.columns))
model = lgb.LGBMClassifier(objective='binary',
                           boosting_type='gbdt',
                           tree_learner='serial',
                           num_leaves=64,
                           max_depth=8,
                           learning_rate=0.02,
                           n_estimators=10000,
                           subsample=0.8,
                           feature_fraction=0.8,
                           reg_alpha=0.3,
                           reg_lambda=0.5,
                           random_state=2020,
                           is_unbalance=True)

prediction = 0.0
feature_importance_list = []
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
for fold_id, (train_idx, val_idx) in enumerate(kfold.split(train[feature_names], train[label])):

    X_train = train.iloc[train_idx][feature_names]
    Y_train = train.iloc[train_idx][label]

    X_val = train.iloc[val_idx][feature_names]
    Y_val = train.iloc[val_idx][label]

    print('\nFold_{} Training ================================\n'.format(fold_id + 1))
    lgb_model = model.fit(X_train,
                          Y_train,
                          eval_names=['train', 'valid'],
                          eval_set=[(X_train, Y_train), (X_val, Y_val)],
                          verbose=500,
                          eval_metric=lambda y_true, y_pred: f1(
                              y_true, y_pred),
                          early_stopping_rounds=80)

    # 测试集合预测
    y_pred = lgb_model.predict(
        test[feature_names], num_iteration=lgb_model.best_iteration_)

    feature_importance = pd.DataFrame({
        'column': feature_names,
        'importance': lgb_model.feature_importances_,
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
feature_importances.to_csv('feature_importance.csv')
predict = pd.DataFrame(
    {'id': test['id'],
     "score": prediction}
)
predict.to_csv('prediction.csv', index=False)
# %%
