# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:10:41 2019
利用机器学习方法对处理后的数据进行分析
@author: 尚梦琦
"""
from bikeUsing import featurePreprocessing
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
#1. 首先获取处理过的数据
X_vec, Y_vec_reg, Y_vec_cas = featurePreprocessing()

#2. 将数据切分为训练集和测试集
cv = cross_validation.ShuffleSplit(len(X_vec), n_iter = 3, test_size=0.2, random_state=0)

#3. 利用不同的模型拟合数据
print("支持向量回归SVR(kernek='rbf', C=10, gamma=0.01)")
for train, test in cv:
    X_train = X_vec[train]
    Y_vec_reg_train =  Y_vec_reg[train]
    Y_vec_cas_train = Y_vec_cas[train]
    
    X_test = X_vec[test]
    Y_vec_reg_test =  Y_vec_reg[test]
    Y_vec_cas_test = Y_vec_cas[test]
    
    #构建并拟合svm模型
    model = svm.SVR(kernel='rbf', C=10, gamma=0.01)
    model.fit(X_train, Y_vec_reg_train)
    
    train_score = model.score(X_train, Y_vec_reg_train)
    test_score = model.score(X_test, Y_vec_reg_test)
    print("train score:{:.2f}, test score:{:.2f}".format(train_score, test_score))

print("随机森林回归RFR(n_estimators=100)")
for train, test in cv:
    X_train = X_vec[train]
    Y_vec_reg_train =  Y_vec_reg[train]
    Y_vec_cas_train = Y_vec_cas[train]
    
    X_test = X_vec[test]
    Y_vec_reg_test =  Y_vec_reg[test]
    Y_vec_cas_test = Y_vec_cas[test]
    
    #构建并拟合svm模型
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, Y_vec_reg_train)
    
    train_score = model.score(X_train, Y_vec_reg_train)
    test_score = model.score(X_test, Y_vec_reg_test)
    print("train score:{:.2f}, test score:{:.2f}".format(train_score, test_score))
