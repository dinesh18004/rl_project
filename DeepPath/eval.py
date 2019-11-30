#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import os
from queue import Queue


# In[2]:


from tqdm import tqdm


# In[3]:


from keras import regularizers
from keras import losses
from keras.models import Sequential 
from keras.layers import Dense, Activation
import numpy as np


# In[4]:


import keras


# In[6]:


from keras.layers import Input,Dense
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf


# In[14]:


import keras.backend as k


# In[7]:


tf.keras.backend.set_floatx("float64")


# In[8]:


keras.backend.set_floatx("float64")


# In[9]:


dataPath = '.\\NELL-995\\'


# In[77]:


relation = "concept_athleteplaysforteam"


# In[78]:


state_dim = 200
action_space = 400
eps_start = 1
eps_end = 0.1
epe_decay = 1000
replay_memory_size = 10000
batch_size = 128
embedding_dim = 100
gamma = 0.99
target_update_freq = 1000
max_steps = 50
max_steps_test = 50


# In[79]:


num_paths = 5


# In[80]:


graphpath = dataPath + 'tasks\\' + relation + '\\' + 'graph.txt'
relationPath = dataPath + 'tasks\\' + relation + '\\' + 'train_pos'


# In[81]:


featurePath = dataPath + 'tasks\\' + relation + '\\path_to_use.txt'
feature_stats = dataPath + 'tasks\\' + relation + '\\path_stats.txt'


# In[82]:


def bidir_bfs(ent1,ent2,path,nodes1,nodes_inv1):
    '''the bidirectional search for reasoning'''
    start = 0
    end = len(path)
    left = set()
    right = set()
    left.add(ent1)
    right.add(ent2)

    left_path = []
    right_path = []
    while(start < end):
        left_step = path[start]
        left_next = set()
        right_step = path[end-1]
        right_next = set()

        if len(left) < len(right):
            left_path.append(left_step)
            start += 1
            for ent in left:
                try:
                    for cns in nodes1[ent]:
                        if cns[0] == left_step:
                            left_next.add(cns[1])
                except Exception as e:
                    return False
            left = left_next

        else:
            right_path.append(right_step)
            end -= 1
            for ent in right:
                try:
                    for cns in nodes_inv1[ent]:
                        if cns[0] == right_step:
                            right_next.add(cns[1])
                except Exception as e:
                    return False
            right = right_next

    if len(right & left) > 0:
        return True 
    return False


# In[83]:


f = open(dataPath + 'relation2id.txt')
relation2id_data = f.readlines()
f.close()
relation2id_dict = {}
relations = [] # stores actions(for getting action by index)
for line in relation2id_data:
    strs = line.split()
    rel_str = strs[0]
    rel_id = int(strs[1])
    relation2id_dict[rel_str] = rel_id
    relations.append(rel_str)


# In[84]:


f = open(graphpath)
graph_data = f.readlines()
f.close()

nodes = dict()
nodes_inv = dict()
for line in graph_data:
    e1, rel, e2 = line.rsplit()
    if e1 not in nodes:
        nodes[e1] = list()
    if e2 not in nodes_inv:
        nodes_inv[e2] = []
    nodes[e1].append((rel, e2))
    nodes_inv[e2].append((rel, e1))


# In[85]:


f = open(feature_stats)
path_nums = f.readlines()
f.close()

stats = {}
for line in path_nums:
    strs = line.split('\t')
    stats[strs[0]] = int(strs[1])

# useful_paths = []
named_paths = []
f = open(featurePath)
paths = f.readlines()
f.close()

print(len(paths))

for line in paths:
    path = line.rstrip()
    rels = path.split(' -> ')

    if len(rels) <= 10:
        named_paths.append(rels)

print('How many paths used: ', len(named_paths))


# In[86]:


f = open(dataPath + 'tasks\\' + relation + '\\train.pairs')
train_data = f.readlines()
f.close()


# In[88]:


train_pairs = []
train_labels = []
for line in train_data:
    strs = line.split(',')
    ent1 = strs[0][6:]
    ent2 = strs[1].split(':')[0][6:]
    if (ent1 in nodes) and (ent2 in nodes):
        train_pairs.append((ent1,ent2))
        if line[-2] == '+':
            train_labels.append(1)
        else:
            train_labels.append(0)
training_features = []
for pair in train_pairs:
    feature = []
    for path in named_paths:
        feature.append(int(bidir_bfs(pair[0], pair[1], path, nodes, nodes_inv)))
    training_features.append(feature)
model = Sequential()
input_dim = len(named_paths)
model.add(Dense(1, activation='sigmoid' ,input_dim=input_dim))
model.compile(optimizer = 'rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(np.array(training_features), np.array(train_labels), nb_epoch=300, batch_size=128)


# In[89]:


f = open(dataPath + 'tasks\\' + relation + '\\sort_test.pairs')
test_data = f.readlines()
f.close()
test_pairs = []
test_labels = []
# queries = set()
for line in test_data:
    strs = line.split(',')
    ent1 = strs[0][6:]
    ent2 = strs[1].split(':')[0][6:]
    if (ent1 in nodes) and (ent2 in nodes):
        test_pairs.append((ent1,ent2))
        if line[-2] == '+':
            test_labels.append(1)
        else:
            test_labels.append(0)


# In[95]:


scores = []
aps = []

query = test_pairs[0][0]
y_true = []
y_sc = []
for idx, sample in enumerate(test_pairs):
    #print 'query node: ', sample[0], idx
    if query != sample[0]:
        cnt = list(zip(y_sc, y_true))
        cnt.sort(key = lambda x:x[0], reverse=True)
        crt = 0
        ranks = []
        for idx1, item in enumerate(cnt):
            if item[1]:
                crt +=  1
                ranks.append(crt/(idx1+1.0))
        
        query = sample[0]
        y_true = []
        y_sc = []
        if len(ranks) ==0:
            aps.append(0)
        else:
            aps.append(np.mean(ranks))
    
    features = [int(bfs_two(sample[0], sample[1], p, nodes, nodes_inv)) for p in named_paths]
        
    sc = model.predict(np.reshape(features,[1,-1]))
    
    y_sc.append(sc)
    
    scores.append(sc[0])
    
    y_true.append(test_labels[idx])
        
cnt = list(zip(y_sc, y_true))
cnt.sort(key = lambda x:x[0], reverse=True)

crt = 0
ranks = []
for idx1, item in enumerate(cnt):
    if item[1]:
        crt +=  1
        ranks.append(crt/(idx1+1.0))
aps.append(np.mean(ranks))

sc_lbl = list(zip(scores, test_labels))
sc_lbl_sorted = sorted(scores, key = lambda x:x[0], reverse=True)

mean_ap = np.mean(aps)
print('RL MAP: ', mean_ap)

