import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from sklearn.pipeline import Pipeline
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer,accuracy_score
from numpy import logical_and as land

class Wome_Scorer():
    def __init__(self,weight_list,intercept=0,default_score=60,double_range_step=0.2,quantile_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]):
        self.p=np.log(2)/double_range_step
        self.q=default_score
        self.quantile_list=quantile_list
        self.sub_factor_quantile=[]
        self.sub_factor_probability=[]
        self.weight_list=weight_list
        self.intercept=intercept
    def fit(self,df_train):
        df_X=df_train.ix[:,0:10]
        sub_factor_quantile=df_X.quantile(self.quantile_list).values.tolist()
        probability_matrix=np.zeros(shape=(len(sub_factor_quantile),10))
        total_good = len(df_train[df_train['PredictClass'] == 0])
        total_bad = len(df_train[df_train['PredictClass'] == 1])
        for i in range(probability_matrix.shape[0]):
            for j in range(probability_matrix.shape[1]):
                if i==0:
                    good=len(df_train[land(df_train.ix[:,'PredictClass']==0,df_train.ix[:,j]<=sub_factor_quantile[i][j])])
                    bad=len(df_train[land(df_train.ix[:,'PredictClass']==1,df_train.ix[:,j]<=sub_factor_quantile[i][j])])
                else:
                    good = len(df_train[land(df_train.ix[:,'PredictClass']==0,land(df_train.ix[:,j]<=sub_factor_quantile[i][j],df_train.ix[:,j]>sub_factor_quantile[i-1][j]))])
                    bad = len(df_train[land(df_train.ix[:,'PredictClass']==1,land(df_train.ix[:,j]<=sub_factor_quantile[i][j],df_train.ix[:,j]>sub_factor_quantile[i-1][j]))])
                if bad==0 and good==0:
                    probability_matrix[i, j]=1
                elif bad==0 and good>0:
                    probability_matrix[i, j]=10
                elif bad>0 and good==0:
                    probability_matrix[i, j] = 0.1
                else:
                    probability_matrix[i,j]=(bad/total_bad)/(good/total_good)
        self.sub_factor_quantile=sub_factor_quantile
        self.sub_factor_probability=probability_matrix.tolist()

    def score(self,df_test):
        score_matrix=np.zeros(shape=df_test.shape)
        for i in range(score_matrix.shape[0]):
            for j in range(score_matrix.shape[1]):

                for k in range(len(self.sub_factor_quantile)):
                    if df_test.ix[i,j]<=self.sub_factor_quantile[k][j]:
                        break

                score_matrix[i,j] = self.p*(self.weight_list[j] * self.sub_factor_probability[k][j])
        df_result=pd.DataFrame(data=score_matrix.tolist(),columns=['x1 score','x2 score','x3 score','x4 score','x5 score','x6 score','x7 score','x8 score','x9 score','x10 score'])

        df_result['base']=self.q-self.p*(self.intercept)
        df_result['total']=df_result['base']-df_result.ix[:,0:9].sum(axis=1)
        return df_result

def SMOTE_sample(inputX,index,test_Ratio=0.5):
    data_array=np.atleast_1d(inputX)
    class_array=np.unique(data_array)
    test_list=[]
    train_list=[]
    for c in class_array:
        temp=[]
        for i,value in enumerate(data_array):
            if value==c:
                temp.append(index[i])
        test_list.extend(sample(temp,int(len(temp)*test_Ratio)))


    return list(set(index)-set(test_list)) ,test_list


def load_data():
    df_train=pd.read_csv('./Credit_Score_Data/cs-training.csv',sep=',',header='infer')
    df_train.drop(['Unnamed: 0'],inplace=True,axis=1)
    #df_train.drop(['SeriousDlqin2yrs'],inplace=True ,axis=1)

    df_test=pd.read_csv('./Credit_Score_Data/cs-test.csv',sep=',',header='infer')
    df_test.drop(['Unnamed: 0'], inplace=True, axis=1)
    df_test.drop(['SeriousDlqin2yrs'], inplace=True,axis=1)
    return df_train,df_test

def data_clean(df_data):
    df_train=df_data.copy()
    df_train.fillna(df_train.mean(axis=0),axis=0,inplace=True)
    #df_test.fillna(df_test.mean(axis=0), axis=0,inplace=True)

    df_train=df_train[df_train['RevolvingUtilizationOfUnsecuredLines']<=16000]
    df_train = df_train[df_train['NumberOfTime30-59DaysPastDueNotWorse'] <= 20]
    df_train = df_train[df_train['NumberOfOpenCreditLinesAndLoans'] <= 50]
    df_train = df_train[df_train['MonthlyIncome'] <= 200000]

    df_train=df_train[np.logical_and(df_train['age']!=0,df_train['age']<97)]
    '''p=df_train[['MonthlyIncome']].boxplot(return_type='dict')
    x_outliers=p['fliers'][0].get_xdata()
    y_outliers = p['fliers'][0].get_ydata()
    for j in range(1):
        plt.annotate(y_outliers[j], xy=(x_outliers[j], y_outliers[j]), xytext=(x_outliers[j] + 0.02, y_outliers[j]))
    plt.show()'''
    '''x=np.array(df_train.values.tolist()).T
    Mcov=np.round(np.corrcoef(x),4)'''
    return df_train

def split_sample(df_train):
    train_list,test_list=SMOTE_sample(df_train['SeriousDlqin2yrs'].tolist(),df_train.index.tolist(),0.5)
    df_train_section=df_train.ix[train_list,:]
    df_test_section=df_train.ix[test_list,:]
    df_Y_train=df_train_section['SeriousDlqin2yrs']
    df_Y_test = df_test_section['SeriousDlqin2yrs']
    df_train_section.drop(['SeriousDlqin2yrs'],inplace=True ,axis=1)
    df_test_section.drop(['SeriousDlqin2yrs'], inplace=True, axis=1)
    return df_train_section.values.tolist(),df_test_section.values.tolist(),df_Y_train.values.tolist(),df_Y_test.values.tolist(),df_test_section

    #x=np.array(df_train.values.tolist()).T
    #Mcov=np.corrcoef(x)
    #print(Mcov)
    #plt.hist(df_train['RevolvingUtilizationOfUnsecuredLines'],bins=100)
    #for j in range(len(x_outliers)):
        #plt.annotate(y_outliers[j], xy=(x_outliers[j], y_outliers[j]), xytext=(x_outliers[j] + 0.08, y_outliers[j]))



# 数据读取和清洗
df_train,df_test=load_data()
df_train=data_clean(df_train)
df_test=data_clean(df_test)
# 数据分割SMOTE法
train_section,test_section,Y_train,Y_test,df_test_section=split_sample(df_train)


# 使用逻辑回归，并使用学习曲线确定参数
pipe_lr = Pipeline([('stdsca', StandardScaler()), ('lr', LogisticRegression(random_state=1,C=0.1))])


#param_range = [2,3,4,5,6,7,8,9,10]
'''param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(estimator=pipe_lr, X=train_section, y=Y_train, param_name='lr__C', param_range=param_range, cv=3)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(param_range, train_mean, color='b', marker='o', markersize=5, label='param_train')
plt.fill_between(param_range, train_mean+train_std, train_mean-train_std, alpha=0.3, color='b')
plt.plot(param_range, test_mean, ls='--', color='g', marker='s', markersize=5, label='param_test')
plt.fill_between(param_range, test_mean+test_std, test_mean-test_std, alpha=0.3, color='g')
plt.grid()
plt.xscale('log')
plt.xlabel('param: PCA')
plt.ylabel('precision')
plt.legend(loc='lower right')

plt.show()'''

#df_test=df_test.isnull()
#print(df_test.sum(axis=0))
# 回归模型测试
'''pipe_lr.fit(train_section, Y_train)
weight=pipe_lr.named_steps['lr'].coef_
y_pred = pipe_lr.predict(test_section)
confmat = confusion_matrix(y_true=Y_test, y_pred=y_pred)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('target')
plt.ylabel('prediction')
print('precision score：%.3f' % precision_score(y_true=Y_test, y_pred=y_pred))
print('recall score：%.3f' % recall_score(y_true=Y_test, y_pred=y_pred))
print('f1 score：%.3f' % f1_score(y_true=Y_test, y_pred=y_pred))
print('accuracy score：%.3f' % accuracy_score(y_true=Y_test, y_pred=y_pred))
plt.show()'''

# 计算回归模型权重
pipe_lr.fit(train_section, Y_train)
weight=pipe_lr.named_steps['lr'].coef_[0]
intercept=pipe_lr.named_steps['lr'].intercept_[0]
Y_pred = pipe_lr.predict(test_section)
#df_test_section['ActualClass']=Y_test
df_test_section['PredictClass']=Y_pred

# 计算各变量的评分值
ws=Wome_Scorer(weight_list=weight,intercept=intercept)
ws.fit(df_test_section)
df_test=df_test.reindex(index=range(len(df_test)))
print(ws.score(df_test[0:1000]))
#df_result=ws.score(df_test)


























