
from sklearn.datasets import load_breast_cancer
from genetic_selection import GeneticSelectionCV
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

if __name__ == '__main__':

    path = '../Data/BandSelect/Dataset.xlsx'
   #获取数据 直接使用 read_excel() 方法读取
    
    # 初始化分类器
    # #estimator = RandomForestClassifier(n_estimators=10)
    # #estimator = SVC()  # 回归函数
    #estimator = KNeighborsClassifier(n_neighbors=3)
    estimator = SVC(kernel='rbf')

    for d in range(1,20):  # d为从60个特征中选择的特征维数
         selector = GeneticSelectionCV(estimator,
                                       cv=10,
                                       verbose=1,
                                       scoring="accuracy",
                                       max_features=d,
                                       n_population=200,
                                       crossover_proba=0.5,
                                       mutation_proba=0.2,
                                       n_generations=200,
                                       crossover_independent_proba=0.5,
                                       mutation_independent_proba=0.05,
                                       tournament_size=3,
                                       n_gen_no_change=10,
                                       caching=True,
                                       n_jobs=-4)
         #拟合特征选择器
         selector = selector.fit(X_train, y_train)
         best_scores=selector.generation_scores_.max()
         # 打印所选择的特征
         print("在取%d维的时候，通过遗传算法得出的最优适应度值为：%.6f" % (d, best_scores))
         print("在取%d维的时候，通过遗传算法获取的有效维度未：%d" % (d, selector.n_features_))
         print("选出的最优染色体为：")
         print('The mask of selected features:', selector.support_)
