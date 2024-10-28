
from mlxtend.feature_selection import SequentialFeatureSelector
import pandas as pd
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
from sklearn.svm import SVC
#读取数据

path = '../Data/BandSelect/Dataset.xlsx'
 #获取数据 直接使用 read_excel() 方法读取

#estimator = RandomForestClassifier(n_estimators=4)
#estimator = KNeighborsClassifier(n_neighbors=3)
estimator = SVC(kernel='rbf')
sfs = SequentialFeatureSelector(estimator,
          k_features=20,
          forward=True,
          floating=False,
          scoring='accuracy',
          verbose=4,
          cv=10)

sfs = sfs.fit(X_train, y_train)

fig1 = plot_sfs(sfs.get_metric_dict(),
                kind='std_dev',
                figsize=(6, 4))

plt.ylim([0.8, 1])
plt.title('Sequential Forward Selection 。(w. StdDev)')
plt.grid()
plt.show()

x,y = [],[]
for i in range(1,20):
    x.append(sfs.subsets_[i]['feature_names'])
    y.append(round(sfs.subsets_[i]['avg_score'],2))
sfs_res = pd.DataFrame({'feature_length':[len(i) for i in x],'feature_names':x,'avg_score':y})
sfs_res.sort_values(['avg_score','feature_length'],ascending=False,inplace=True)
sfs_res
print(x)
print(y)
print(sfs_res)

print('best combination (ACC: %.6f): %s\n' % (sfs.k_score_, sfs.k_feature_idx_))
print('all subsets:\n', sfs.subsets_)
plot_sfs(sfs.get_metric_dict(), kind='std_err');

# 打印SFS选择的特征和真正信息特征的数量
#print("SFS选择的特征：", efs1.get_support())
#print("SFS选择的真正信息特征数：", sum(efs1.get_support()[:10]))
#print("SFS选择的错误信息特征数：", sum(efs1.get_support()[10::]))

# 逆序特征选择（SBS）
#sbs = SequentialFeatureSelector(lda, n_features_to_select=10, cv=strkfold, direction='backward', n_jobs=-1)
#sbs.fit(X_train, y_train)

# 打印SBS选择的特征和真正信息特征的数量
#print("SBS选择的特征：", sbs.get_support())
#print("SBS选择的真正信息特征数：", sum(sbs.get_suppor44、t()[:10]))
#print("SBS选择的错误信息特征数：", sum(sbs.get_support()[10::]))

temp=pd.DataFrame.from_dict(sfs.get_metric_dict()).T
print(temp)