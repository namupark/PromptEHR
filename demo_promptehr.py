#!/usr/bin/env python
# coding: utf-8

# In[1]:


import transformers


# In[2]:


transformers.__version__


# In[3]:


from pytrial.data.demo_data import load_synthetic_ehr_sequence
from pytrial.tasks.trial_simulation.data import SequencePatient

demo = load_synthetic_ehr_sequence(n_sample=100)


# In[4]:


demo


# In[5]:


len(demo['visit'])


# In[6]:


demo.keys()


# In[7]:


# build sequence dataset
seqdata = SequencePatient(data={'v':demo['visit'], 'y':demo['y'], 'x':demo['feature'],},
    metadata={
        'visit':{'mode':'dense'},
        'label':{'mode':'tensor'}, 
        'voc':demo['voc'],
        'max_visit':20,
        }
    )

print('visit', demo['visit'][0]) # a list of visit events
print('mortality', demo['y'][0]) # array of labels
print('feature', demo['feature'][0]) # array of patient baseline features
print('voc', demo['voc']) # dict of dicts containing the mapping from index to the original event names
print('order', demo['order']) # a list of three types of code
print('n_num_feature', demo['n_num_feature']) # int: a number of patient's numerical features
print('cat_cardinalities', demo['cat_cardinalities']) # list: a list of cardinalities of patient's categorical features


# In[9]:


demo['voc']


# In[14]:


demo['voc']['med'].idx2word


# In[ ]:





# In[18]:


from promptehr import PromptEHR

# fit the model
model = PromptEHR(
    code_type=demo['order'],
    n_num_feature=demo['n_num_feature'],
    cat_cardinalities=demo['cat_cardinalities'],
    num_worker=0,
    eval_step=1,
    epoch=5,
    device=[0],
)
model.fit(
    train_data=seqdata,
    val_data=seqdata,
)


# In[15]:


model.evaluate(seqdata)


# In[17]:


model


# In[ ]:





# In[ ]:


# save the model
model.save_model('./simulation/promptEHR')


# In[12]:


# generate fake records
res = model.predict(seqdata, n_per_sample=10, n=10, verbose=True)


# In[13]:


print(res)


# In[ ]:





# In[1]:


import os
os.chdir('../')


# In[16]:


# if you want pretrained model downloaded
from promptehr import PromptEHR
model = PromptEHR()
model.from_pretrained()


# In[17]:


model.training_args


# In[19]:


model.evaluate(seqdata)


# In[ ]:





# In[ ]:





# In[15]:


model.fit(
    train_data=seqdata,
    val_data=seqdata,
)


# In[ ]:




