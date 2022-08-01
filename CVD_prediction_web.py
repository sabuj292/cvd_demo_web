#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab



# In[2]:


heart_df=pd.read_csv("framingham.csv")
heart_df.drop(['education'],axis=1,inplace=True)
heart_df.head()


# In[3]:


# heart_df.rename(columns={'male':'Sex_male'},inplace=True)


# In[4]:


heart_df.isnull().sum()


# In[5]:


count=0
for i in heart_df.isnull().sum(axis=1):
    if i>0:
        count=count+1
print('Total number of rows with missing values is ', count)
print('since it is only',round((count/len(heart_df.index))*100), 'percent of the entire dataset the rows with missing values are excluded.')


# In[6]:


heart_df.dropna(axis=0,inplace=True)


# ## <font color=RoyalBlue>Exploratory Analysis<font>

# In[7]:


heart_df.CVD.value_counts()


# In[8]:


heart_df.describe()


# In[9]:


from statsmodels.tools import add_constant as add_constant
heart_df_constant = add_constant(heart_df)
heart_df_constant.head()


# In[10]:


st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)
cols=heart_df_constant.columns[:-1]
model=sm.Logit(heart_df.CVD,heart_df_constant[cols])
result=model.fit()
result.summary()


# The results above show some of the attributes with P value higher than the preferred alpha(5%) and thereby showing  low statistically significant relationship with the probability of heart disease. Backward elemination approach is used here to remove those attributes with highest Pvalue one at a time follwed by  running the regression repeatedly until all attributes have P Values less than 0.05.
# 
# 

# ### <font color=CornflowerBlue>Feature Selection: Backward elemination (P-value approach)<font>

# In[11]:


def back_feature_elem (data_frame,dep_var,col_list):
    """ Takes in the dataframe, the dependent variable and a list of column names, runs the regression repeatedly eleminating feature with the highest
    P-value above alpha one at a time and returns the regression summary with all p-values below alpha"""

    while len(col_list)>0 :
        model=sm.Logit(dep_var,data_frame[col_list])
        result=model.fit(disp=0)
        largest_pvalue=round(result.pvalues,3).nlargest(1)
        if largest_pvalue[0]<(0.05):
            return result
            break
        else:
            col_list=col_list.drop(largest_pvalue.index)

result=back_feature_elem(heart_df_constant,heart_df.CVD,cols)


# In[12]:


result.summary()


# ## <font color=RoyalBlue>Interpreting the results: Odds Ratio, Confidence Intervals and Pvalues<font>

# In[13]:


params = np.exp(result.params)
conf = np.exp(result.conf_int())
conf['OR'] = params
pvalue=round(result.pvalues,3)
conf['pvalue']=pvalue
conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio','pvalue']
print ((conf))


# ### <font color=CornflowerBlue>Splitting data to train and test split<font>

# In[14]:


import sklearn
new_features=heart_df[['age','male','cigsPerDay','totChol','sysBP','glucose','CVD']]
x=new_features.iloc[:,:-1]
y=new_features.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)


# In[15]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)


# In[16]:


output = logreg.predict([[60, 1, 20, 300, 180, 150]])


# In[17]:


output


# In[18]:


y_pred_prob=logreg.predict_proba([[60, 1, 20, 300, 180, 150]])[:,:]
y_pred_prob



# y_pred_prob=logreg.predict_proba(x_test)[:,:]
# y_pred_prob_df=pd.DataFrame(data=y_pred_prob, columns=['Prob of no heart disease (0)','Prob of Heart Disease (1)'])
# y_pred_prob_df.head()


# In[ ]:





# ## <font color=RoyalBlue>Model Evaluation<font>
# 
# ### <font color=CornflowerBlue>Model accuracy<font>

# In[19]:


sklearn.metrics.accuracy_score(y_test,y_pred)


# In[20]:


pickle.dump(logreg, open('model.pkl', 'wb'))


# In[21]:


model = pickle.load(open('model.pkl', 'rb'))


# In[ ]:




