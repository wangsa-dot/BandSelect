from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn import metrics
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

# Create a sample dataset for illustration purposes
path = '../Data/BandSelect/Dataset_red.xlsx'
 #获取数据 直接使用 read_excel() 方法读取

#print("11")
"""
rfc = RandomForestClassifier(n_estimators=100,random_state=90)
score_pre = cross_val_score(rfc,TrainData_X,TrainData_Y1,cv=10).mean()

print(score_pre)

# 先画出超参数学习曲线，在大致范围上进行观察，再小范围取值调优
scorel = []
for i in range(1, 201, 10):
    rfc = RandomForestClassifier(n_estimators=i, random_state=90, n_jobs=-1)
    score = cross_val_score(rfc, TrainData_X, TrainData_Y1, cv=10).mean()
    scorel.append(score)
print(max(scorel), (scorel.index(max(scorel)) * 10))
plt.figure(figsize=[20, 5])
plt.plot(range(1, 201, 10), scorel)
plt.show()

scorel = []
for i in range(65, 75):
    rfc = RandomForestClassifier(n_estimators=i,
                                 n_jobs=-1,
                                 random_state=90)
    score = cross_val_score(rfc, TrainData_X, TrainData_Y1, cv=10).mean()
    scorel.append(score)
print(max(scorel), ([*range(65, 75)][scorel.index(max(scorel))]))
plt.figure(figsize=[20, 5])
plt.plot(range(65, 75), scorel)
plt.show()



有一些参数是没有参照的，很难说清一个范围，这种情况下我们使用学习曲线，看趋势
从曲线跑出的结果中选取一个更小的区间，再跑曲线
param_grid = {'n_estimators':np.arange(0, 200, 10)}
param_grid = {'max_depth':np.arange(1, 20, 1)}  
param_grid = {'max_leaf_nodes':np.arange(25,50,1)}
对于大型数据集，可以尝试从1000来构建，先输入1000，每100个叶子一个区间，再逐渐缩小范围

有一些参数是可以找到一个范围的，或者说我们知道他们的取值和随着他们的取值，模型的整体准确率会如何变化，这
样的参数我们就可以直接跑网格搜索
param_grid = {'criterion':['gini', 'entropy']}
param_grid = {'min_samples_split':np.arange(2, 2+20, 1)}
param_grid = {'min_samples_leaf':np.arange(1, 1+10, 1)}
param_grid = {'max_features':np.arange(5,30,1)}
"""

param_grid = {'max_depth': np.arange(1, 20, 1)}

# 一般根据数据的大小来进行一个试探，采用1~10，或者1~20这样的试探
# 但对于大型数据来说，我们应该尝试30~50层深度
# 更应该画出学习曲线，来观察深度对模型的影响
rfc = RandomForestClassifier(n_estimators=40, random_state=0, n_jobs=-1)
rfc.fit(X_train, y_train)
#print(rfc.best_params_)


pred = rfc.predict(X_train)
print(metrics.classification_report(pred,y_train))

pred1 = rfc.predict(X_test)
print(metrics.classification_report(pred1,y_test))

#筛选变量
importances = rfc.feature_importances_
print('各个feature的重要性：%s '%rfc.feature_importances_)
std=np.std([tree.feature_importances_ for tree in rfc.estimators_],axis=0)

# Fit model using each importance as a threshold
thresholds = sort(rfc.feature_importances_)
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(rfc, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)

    # train model
    selection_model = RandomForestClassifier(n_estimators=40, random_state=0, n_jobs=-1)
    selection_model.fit(select_X_train, y_train)

    #clf_s = cross_val_score(selection_model,select_X_train, y_train, cv=10).mean()

    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))
