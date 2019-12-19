
# coding: utf-8

# In[53]:


import pandas as pd
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


# In[54]:


def cleanUser(user):
    return user.replace('DTAA/', '')


# In[55]:


emailDf = pd.read_csv('r4.1/email.csv', nrows=10)


# In[56]:


emailDf.head()


# In[57]:


emailDf.sort_values(by=['date'], inplace=True, ascending=False)


# In[58]:


emailDf.head()


# In[59]:


emailDf.describe()


# According to dataset, We can say that we have date, size, attachment as numerical dataset and others as string. So we are converting them to categorical dataset. But before that we have to NLP so that we can generate the vectorized matrix from the data.
# 
# for this task we have 3 pipelines.
# - NLP Processing for our unstruvtured data
# - Modeling & Predictions
# 

# In[60]:


emailDf['content'].head()


# ### NLP Processing for our unstruvtured data
# 
# Now we have our four features but by observing above table we have some string data available and as we know that machines can not understand these strings so we have transform these string respresented data in to numerical form.
# 
# To do NLP on title, we have to follow the particular pipeline to get the major words from the content.
# We are applying pipeline given as below.
# 
# - REMOVE PUNCTUATION
# - TOKENIZATION
# - REMOVING STOPWORDS
# - STEMMING (IF NEEDED TO IMPROVE ACCURACY)
# - VECTORIZING

# #####  Remove Punctuation
# 
# In this method, We are removing all the following the punctuations from content

# In[61]:


import string

print(string.punctuation)


# In[62]:


def remove_punc(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct


# In[63]:


''' python is case sensetive for that A and a is diffrent thats why lower()'''
emailDf['content_punc'] = emailDf['content'].apply(lambda x: remove_punc(x.lower()))
print(emailDf.head())


# #####  Tokenization
# Tokenization is the act of breaking up a sequence of strings into pieces such as words, keywords, phrases, symbols and other elements called tokens. Tokens can be individual words, phrases or even whole sentences. In the process of tokenization, some characters like punctuation marks are discarded.

# In[64]:


import re


def tokenize(text):
    # Split word non word
    tokens = re.split('\W+', text)
    return tokens

emailDf['content_tokenize'] = emailDf['content_punc'].apply(lambda x: tokenize(x))
print(emailDf.head())


# So we got the our tokenized column as "content_tokenize"

# ##### Remove StepWords
# 
# StepWords are those which are words which does not play importance in sentences.
# Like I am going to watch movie. where 'I', 'am' and 'to' plays part of the stopwords.
# 
# Here we will use nltk package to remvoe stopwords.

# In[65]:


import nltk
print('NLTK (DOWNLOAD ALL PACKAGES TO PERFORM NLP OPERATION)')

print('UNCOMMENT FOLLOWING LINE To GET NLTK DOWNLOADED')
#nltk.download()
stopword = nltk.corpus.stopwords.words('english')

def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]
    return text


# In[66]:


emailDf['content_remove_stopwords'] = emailDf['content_tokenize'].apply(lambda x: remove_stopwords(x))
print(emailDf.head())


# #### Stemming

# In[67]:



from nltk.stem import PorterStemmer 
ps = PorterStemmer() 
def stemming(words):
    words = [ ps.stem(w) for w in words]
    return words


# In[68]:


emailDf['content_remove_stopwords'] = emailDf['content_remove_stopwords'].apply(lambda x: stemming(x))
print(emailDf.head())


# Now we will create one method which will do the all the 3 steps in one shot.

# In[69]:


def clean_text(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    tokens = re.split('\W+', text_nopunct)
    text = [word for word in tokens if word not in stopword]
    text = [ ps.stem(w) for w in text]
    return text

def clean_text_without_stemming(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    tokens = re.split('\W+', text_nopunct)
    text = [word for word in tokens if word not in stopword]
    return text

df = pd.read_csv('r4.1/email.csv', nrows=10)
df['content_wo_stem'] = df['content'].apply(lambda x: clean_text_without_stemming(x.lower()))
df['content'] = df['content'].apply(lambda x: clean_text(x.lower()))


# In[70]:


df.head()


# In[71]:


df[['content', 'content_wo_stem']].head()


# ##### Vectorizing
# 
# Now we have to create one vector of this tokenize words so that we can easily fit this with our features so that we can easily create classification model to achieve our target.
# 

# In[72]:


# import dask.dataframe as dd
# df = dd.read_csv('r4.1/email.csv')
# type(df)


# In[73]:


# df = df.compute()


# # start

# In[74]:



df = pd.read_csv('r4.1/email.csv', nrows=700)
df.head(1)


# In[75]:


df.shape


# In[76]:


from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(analyzer=clean_text)
X_counts = count_vect.fit_transform(df['content'])

count_vect_wo = CountVectorizer(analyzer=clean_text_without_stemming)
X_counts_wo = count_vect_wo.fit_transform(df['content'])

print("Stemming Content : " + str(X_counts.shape) + " & Without stemming content : "+ str(X_counts_wo.shape))


# In[77]:


pd.DataFrame(X_counts[1].toarray())


# In[78]:


# del df
##Vectorizing output sparse matrix
X_counts_df = pd.DataFrame(X_counts.toarray())

##Assinging Names
X_counts_df.columns = count_vect.get_feature_names()

X_counts_df_wo = pd.DataFrame(X_counts_wo.toarray())
X_counts_df_wo.columns = count_vect_wo.get_feature_names()
X_counts_df.head()


# In[79]:


X_counts_df.shape


# finally we got our vectorized matrix for title column in X_counts_df dataframe. Now we need to join other columns to these data frame

# In[80]:


df['to'].head()


# In[81]:


print(X_counts_df.shape)
print(df.shape)


# creating one method to process the email id as taking the id before the @ and converting all to lowercase.

# In[82]:


def clean_email(text):
    return str(text).lower().split('@', 1)[0]


# In[83]:


df['to'] = df['to'].apply(lambda x: clean_email(x))
print(df['to'].head())


# Taking nan also as name to process null values

# In[84]:


df['cc'] = df['cc'].apply(lambda x: clean_email(x))
df['cc'].head()


# In[85]:


df['bcc'] = df['bcc'].apply(lambda x: clean_email(x))
df['from'] = df['from'].apply(lambda x: clean_email(x))


# In[86]:


df.head()


# In[87]:


str(df.shape) + ' - ' + str(X_counts_df.shape)


# In[88]:


X_counts_df['date'] = list(df['date'])
X_counts_df['user'] = list(df['user'])
X_counts_df['pc'] = list(df['pc'])
X_counts_df['to'] = list(df['to'])
X_counts_df['from'] = list(df['from'])
X_counts_df['cc'] = list(df['cc'])
X_counts_df['bcc'] = list(df['bcc'])

X_counts_df_wo['date'] = list(df['date'])
X_counts_df_wo['user'] = list(df['user'])
X_counts_df_wo['pc'] = list(df['pc'])
X_counts_df_wo['to'] = list(df['to'])
X_counts_df_wo['from'] = list(df['from'])
X_counts_df_wo['cc'] = list(df['cc'])
X_counts_df_wo['bcc'] = list(df['bcc'])


# In[89]:


str(df.shape) + ' - ' + str(X_counts_df.shape)


# In[90]:


df.head(2)


# Now merging the cleaned features to our vector matrix except "id" column which is not relevant

# In[91]:


dfd = pd.get_dummies(X_counts_df['pc'])
dfd_wo = pd.get_dummies(X_counts_df_wo['pc'])


# In[92]:


dfd.head()


# In[93]:


X_counts_df.head()


# In[94]:


del X_counts_df['pc']
X_counts_df = pd.concat([X_counts_df, dfd], axis=1)

del X_counts_df_wo['pc']
X_counts_df_wo = pd.concat([X_counts_df_wo, dfd], axis=1)


# In[95]:


X_counts_df.head()


# making one common method for merging one hot columns

# In[96]:


import gc
def merge_one_hot_cols(X_counts_df, columns):
    for col in columns:
        dfd = pd.get_dummies(X_counts_df[col])
        del X_counts_df[col]
        X_counts_df = pd.concat([X_counts_df, dfd], axis=1)
        gc.collect()
    return X_counts_df



# In[97]:


X_counts_df = merge_one_hot_cols(X_counts_df, ['user', 'to', 'from', 'cc', 'bcc'])


# In[98]:


X_counts_df_wo = merge_one_hot_cols(X_counts_df_wo, ['user', 'to', 'from', 'cc', 'bcc'])


# In[99]:


X_counts_df.head()


# In[100]:


X_counts_df['date']
X_counts_df['date'].head()


# In[101]:


from datetime import datetime


# In[102]:



year = lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M:%S" ).year
month = lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M:%S" ).month
day = lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M:%S" ).day
hour = lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M:%S" ).hour
minute = lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M:%S" ).minute
second = lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M:%S" ).second

X_counts_df['year'] = X_counts_df['date'].map(year)
X_counts_df['month'] = X_counts_df['date'].map(month)
X_counts_df['day'] = X_counts_df['date'].map(day)
X_counts_df['hour'] = X_counts_df['date'].map(hour)
X_counts_df['minute'] = X_counts_df['date'].map(minute)
X_counts_df['second'] = X_counts_df['date'].map(second)

del X_counts_df['date']


X_counts_df_wo['year'] = X_counts_df_wo['date'].map(year)
X_counts_df_wo['month'] = X_counts_df_wo['date'].map(month)
X_counts_df_wo['day'] = X_counts_df_wo['date'].map(day)
X_counts_df_wo['hour'] = X_counts_df_wo['date'].map(hour)
X_counts_df_wo['minute'] = X_counts_df_wo['date'].map(minute)
X_counts_df_wo['second'] = X_counts_df_wo['date'].map(second)
del X_counts_df_wo['date']


# In[103]:


from sklearn.preprocessing import MinMaxScaler
from anamoly.keras_anomaly_detection.library.plot_utils import visualize_reconstruction_error
from anamoly.keras_anomaly_detection.library.recurrent import LstmAutoEncoder, CnnLstmAutoEncoder


# In[104]:


X_counts_df.head()


# In[105]:


X_np_data = X_counts_df.as_matrix()
scaler = MinMaxScaler()
X_np_data = scaler.fit_transform(X_np_data)
print(X_np_data.shape)

X_np_data_wo = X_counts_df_wo.as_matrix()
scaler = MinMaxScaler()
X_np_data_wo = scaler.fit_transform(X_np_data_wo)
print(X_np_data_wo.shape)


# In[106]:


data_dir_path = './data'
model_dir_path = './models'
ae = CnnLstmAutoEncoder()


# In[107]:



# fit the data and save model into model_dir_path
history = ae.fit(X_np_data, model_dir_path=model_dir_path, estimated_negative_sample_ratio=0.9)


# In[108]:


import matplotlib.pyplot as plt
# list all data in history
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



# In[112]:


ae2 = CnnLstmAutoEncoder()
# fit the data and save model into model_dir_path
history_wo = ae2.fit(X_np_data_wo, model_dir_path=model_dir_path, estimated_negative_sample_ratio=0.9)


# In[114]:



# summarize history for loss
plt.plot(history_wo.history['loss'])
plt.plot(history_wo.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[117]:


# load back the model saved in model_dir_path detect anomaly
# ae.load_model(model_dir_path)
anomaly_information = ae2.anomaly(X_np_data_wo)
reconstruction_error = []
for idx, (is_anomaly, dist) in enumerate(anomaly_information):
    print('# ' + str(idx) + ' is ' + ('abnormal' if is_anomaly else 'normal') + ' (dist: ' + str(dist) + ')')
    reconstruction_error.append(dist)


# In[118]:


visualize_reconstruction_error(reconstruction_error, ae2.threshold)

