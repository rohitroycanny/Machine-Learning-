#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os


# In[2]:


pwd


# In[5]:


df = pd.read_csv('C:/Users/Rohit Kumar/Downloads/train.txt')


# In[6]:


df #reading original data


# In[7]:


df = pd.read_csv('C:/Users/Rohit Kumar/Downloads/train.txt')


# In[12]:


df #read text,status


# In[ ]:


#USING NAIVE BAYES ALGORITHM


# In[13]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[14]:


X = df.text
y = df.status 


# In[22]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)


# In[24]:


vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(X_train.values)


# In[25]:


classifer = MultinomialNB()
targets = y_train.values
classifer.fit(counts,targets)


# In[26]:


df.text[0]


# In[19]:


_count = vectorizer.transform(df.text)
_prediction = classifer.predict(_count)


# In[20]:


import collections
result_dict = dict(collections.Counter(_prediction))
result_dict


# In[27]:


#CALCULATE THE PROBABILITY OF TRAIN DATA


# In[38]:


total_text1 = int(result_dict['ham']+result_dict['spam'])
spam_text1 = int(result_dict['spam'])
prob1 = spam_text1/total_text1
print("Spam =",prob1)


# In[29]:


df_test = pd.read_csv('C:/Users/Rohit Kumar/Downloads/test.txt')


# In[30]:


df #EDIT text,status in file


# In[31]:


test_dict = {}
_count_test = vectorizer.transform(df_test.text)
_prediction_test = classifer.predict(_count_test)
result_test = dict(collections.Counter(_prediction_test))
result_test


# In[32]:


#CALCULATE THE PROBABILITY OF TEST DATA


# In[37]:


total_text = int(result_test['ham']+result_test['spam'])
spam_text = int(result_test['spam'])
prob = spam_text/total_text
print("Spam =",prob)


# In[52]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
spamtt=['ham_test','spam_test','ham_train','spam_train']
hamtt=[340,68,1662,338]
ypos=np.arange(len(spamtt))
plt.xticks(ypos,spamtt)
plt.bar(ypos,hamtt)
plt.title("Count of SPAM & HAM")


# In[53]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
slices=[0.166,0.169]
labels=['Test','Train']
explode=[0,0.1]
plt.pie(slices,labels=labels,explode=explode,shadow=True,autopct='%1.1f%%')
plt.title("Probability of Test Vs Train")
plt.tight_layout()
plt.show()    


# In[54]:


#ACCURACY CHECK


# In[62]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV


# In[65]:


df=pd.read_csv('C:/Users/Rohit Kumar/Downloads/train.txt')


# In[68]:


X = df.text
y = df.status


# In[69]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)


# In[70]:


vectorizer = CountVectorizer()
features = vectorizer.fit_transform(X_train.values)


# In[74]:


tuned_parameters={'kernel':['linear','rbf'],'gamma':[1e-3,1e-4],'C':[1,10,100,1000]}
model=GridSearchCV(svm.SVC(),tuned_parameters)
model.fit(features,y_train)
features_test=vectorizer.transform(X_test)
print("Accuracy",model.score(features_test,y_test))


# In[14]:


import pandas as pd
import numpy as np


# In[15]:


df = pd.read_csv('C:/Users/Rohit Kumar/Downloads/train.txt', encoding='latin-1')
print(df.head())


# In[16]:


df = df.rename(columns={"v1":"status","v2":"text"})
df


# In[17]:


df['label_dummy'] = df.status.map({'ham':0, 'spam':1})
print(df.head())
print(df.status.value_counts())


# In[18]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[19]:


x_train, x_test, y_train, y_test = train_test_split(df["text"], df["status"], test_size = 0.2, random_state = 42)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[20]:


vect = CountVectorizer()
vect.fit(x_train)
print(vect.get_feature_names()[0:50])
print(vect.get_feature_names()[1000:1050])


# In[21]:


x_train_vect = vect.transform(x_train)
x_test_vect = vect.transform(x_test)


# In[22]:


model = LogisticRegression()
model.fit(x_train_vect, y_train)


# In[23]:


prediction = dict()
prediction['Logistic'] = model.predict(x_test_vect)


# In[48]:


print(accuracy_score(y_test, prediction['Logistic']))


# In[24]:


k_nums = np.arange(1,20)
print(k_nums)


# In[25]:


param_grid = dict(n_neighbors = k_nums)
model2 = KNeighborsClassifier()
gridsearch = GridSearchCV(model2, param_grid)
gridsearch.fit(x_train_vect, y_train)


# In[31]:


model3 = KNeighborsClassifier(n_neighbors = 1)
model3.fit(x_train_vect, y_train)


# In[34]:


prediction['KNN'] = model3.predict(x_test_vect)


# In[35]:


conf_mat_logist = confusion_matrix(y_test, prediction['Logistic'])
conf_mat_knn = confusion_matrix(y_test, prediction['KNN'])
print(conf_mat_logist)


# In[39]:


from sklearn.metrics import classification_report
print(classification_report(y_test, prediction['Logistic']))


# In[ ]:


#copied


# In[9]:


import numpy as np 
import pandas as pd
data = pd.read_csv('C:/Users/Rohit Kumar/Downloads/train.txt')


# In[11]:


from collections import Counter, OrderedDict

# create a dictionary with each word as the key and the number of occurences as the value
cnt = Counter()
total_words = 0
for entry in data['text']:
    total_words += len(entry)    # total number of words in the dataset
    for word in entry:
        cnt[word] += 1

# order the previous dictionary by most common words
ordered_cnt = OrderedDict(cnt.most_common())


# In[12]:


num_words_threshold = 9451


# In[13]:


unique_words_90 = list(OrderedDict(cnt.most_common(num_words_threshold)).keys())


# In[14]:


def create_feature(list):
    
    # list with words from list which are in the unique_words_90 list
    words_in_unique_list = [word for word in list if word in unique_words_90]
    
    # dictionary created from previous list counting the number of occurences
    words_in_unique_dict = Counter(words_in_unique_list)
    
    # array of zeros, with the same number of elements as the unique_words_90 list
    features = np.zeros(len(unique_words_90), dtype=np.uint32)
    
    # for each word:occurence in dictionary, save occurence in the correct slot of features
    for word, occurence in words_in_unique_dict.items():
        features[unique_words_90.index(word)] = occurence
    
    return features


# In[16]:


# creates a matrix of zeros with dimensions of (number of targets)x(number of unique words)
X = np.empty((len(data['status']), len(unique_words_90)), dtype=np.uint32)

# use function create_feature for each row
for i, email in enumerate(data['text']):
    X[i,:] = create_feature(email)


# In[18]:


y = data['status']


# In[19]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)


# In[20]:


from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)


# In[21]:


from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))


# In[22]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[23]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[28]:


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

y_pred_prob = nb.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob,600)

# create plot
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
_ = plt.xlabel('False Positive Rate')
_ = plt.ylabel('True Positive Rate')
_ = plt.title('ROC Curve')
_ = plt.xlim([-0.02, 1])
_ = plt.ylim([0, 1.02])
_ = plt.legend(loc="lower right")

# save figure
plt.savefig('roc_curve.png', dpi=200)


# In[29]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_prob)


# In[ ]:




