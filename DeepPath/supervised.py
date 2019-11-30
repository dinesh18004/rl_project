#!/usr/bin/env python
# coding: utf-8

# In[15]:


dataPath = '.\\NELL-995\\'


# In[16]:


relation = "concept_athletehomestadium"


# In[17]:


import random
import os


# In[51]:


from queue import Queue
from collections import Counter


# In[18]:


from tqdm import tqdm


# In[20]:


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


# In[21]:


num_paths = 5


# In[22]:


graphpath = dataPath + 'tasks\\' + relation + '\\' + 'graph.txt'
relationPath = dataPath + 'tasks\\' + relation + '\\' + 'train_pos'


# In[23]:


from keras import regularizers
from keras import losses
import numpy as np


# In[24]:


from keras.layers import Input,Dense
from keras.models import Model
from keras.optimizers import Adam


# In[25]:


import tensorflow as tf


# In[49]:


from tensorflow.train import AdamOptimizer
from tensorflow.contrib.layers import l2_regularizer, xavier_initializer


# In[45]:


class Sl_policy(object):
    def __init__(self, state_size, action_size, lr = 0.001):
        self.init = xavier_initializer()
        with tf.variable_scope('supervised_policy'):
            self.st = tf.placeholder(tf.float32, [None, state_size], name = 'st')
            self.acts_prob = self.sl_policy_nn(self.st, state_size, action_size, self.init)
            self.act = tf.placeholder(tf.int32, [None], name = 'act')
            
            act_mask = tf.cast(tf.one_hot(self.act, depth = action_size), tf.bool)
            self.act_prob = tf.boolean_mask(self.acts_prob, act_mask)

            self.loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope = 'supervised_policy')) + tf.reduce_sum(-tf.log(self.act_prob))
            self.optimizer = AdamOptimizer(learning_rate = lr)
            self.training_op = self.optimizer.minimize(self.loss)
    
    def sl_policy_nn(self, state, state_size, action_size, init):
        w1 = tf.get_variable('W1', [state_size, 512], initializer = init, regularizer=l2_regularizer(0.01))
        b1 = tf.get_variable('b1', [512], initializer = tf.constant_initializer(0.0))
        h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
        w2 = tf.get_variable('w2', [512, 1024], initializer = init, regularizer=l2_regularizer(0.01))
        b2 = tf.get_variable('b2', [1024], initializer = tf.constant_initializer(0.0))
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
        w3 = tf.get_variable('w3', [1024, action_size], initializer = init, regularizer=l2_regularizer(0.01))
        b3 = tf.get_variable('b3', [action_size], initializer = tf.constant_initializer(0.0))
        acts_prob = tf.nn.softmax(tf.matmul(h2,w3) + b3)
        return acts_prob

    def get_act_probs(self, st, sess = None):
        sess = sess or tf.get_default_session()
        return sess.run(self.acts_prob, {self.st: st})

    def train_batch(self, st, act, sess = None):
        sess = sess or tf.get_default_session()
        _, loss = sess.run([self.training_op, self.loss], {self.st: st, self.act: act})
        return loss


# In[28]:


f = open(relationPath)
train_data = f.readlines()
f.close()


# In[30]:


f = open(dataPath + 'entity2id.txt')
entity2id_data = f.readlines()
f.close()
f = open(dataPath + 'relation2id.txt')
relation2id_data = f.readlines()
f.close()
relation2id_dict = {}
entity2id_dict = {}
relations = [] # stores actions(for getting action by index)
for line in relation2id_data:
    strs = line.split()
    rel_str = strs[0]
    rel_id = int(strs[1])
    relation2id_dict[rel_str] = rel_id
    relations.append(rel_str)
for line in entity2id_data:
    strs = line.split()
    ent_str = strs[0]
    ent_id = int(strs[1])
    entity2id_dict[ent_str] = ent_id
entity2vec_data = np.loadtxt(dataPath + 'entity2vec.bern')
relation2vec_data = np.loadtxt(dataPath + 'relation2vec.bern')


# In[32]:


f = open(dataPath + 'kb_env_rl.txt')
kb_data1 = f.readlines()
f.close()

kb_data = []
rel = ":".join(relation.split("_"))
for line in kb_data1:
    relation1 = line.split()[2]
    kb_data.append(line) if ((relation1 != rel + '_inv') and (relation1 != rel)) else None


# In[37]:


f = open(graphpath)
graph_data = f.readlines()
f.close()


# In[39]:


nodes = dict()
for line in graph_data:
    e1, rel, e2 = line.rsplit()
    if e1 not in nodes:
        nodes[e1] = list()
    nodes[e1].append((rel, e2))


# In[41]:


def bfs(nodes1,e1,e2):
    visited = set()
    path = list()
    q = Queue()
    q.put(e1)
    visited.add(e1)
    path.append(e1)
    while(not q.empty()):
        ent = q.get()
        try:
            for link in nodes1[ent]:
                if link[1] not in visited:
                    visited.add(link[1])
                    q.put(link[1])
                    path.extend(list(link))
                if link[1] == e2:
                    return path, True
        except KeyError:
#             print("KeyError: "+ent)
            pass
    return [], False


# In[48]:


tf.reset_default_graph()
p_nn = Sl_policy(state_dim, action_space)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for ep in tqdm(range(min(len(train_data),4))):
    #     print(train_data[ep])
        source = train_data[ep].split()[0]
        target = train_data[ep].split()[1]

        inters = set()
        inters.add(source)
        inters.add(target)
        while(len(inters)<(num_paths + 2)):
            inters.add(random.choice(list(nodes.keys())))
        inters.remove(source)
        inters.remove(target)
        inters = list(inters)

        for i in range(num_paths):
            path1, found1 = bfs(nodes,source,inters[i])
            path2, found2 = bfs(nodes,inters[i],target)
            if found1 and found2:
    #             print("path found")

                path = path1 + path2[1:]
    #             print(path)
                ents = []
                ent_num = dict()
                for j in  range(0,len(path),2):
                    ent = path[j]
                    ents.append(ent)
                    if ent not in ent_num:
                        ent_num[ent] = 0
                    ent_num[ent] += 1
                ent_num = list(ent_num.items())
                clean_ents = [en for en in ent_num if en[1]>1]
                clean_ents.sort(key = lambda en:en[1], reverse=True)
                for en in clean_ents:
                    ent = en[0]
                    ent_ind = []
                    for j in  range(0,len(path),2):
                        if path[j] == ent:
                            ent_ind.append(j)
                    if len(ent_ind)>1:
                        path = path[:ent_ind[0]] + path[ent_ind[-1]:]
                ents = []
                rels = []
                target_vec = entity2vec_data[entity2id_dict[target]]
                states = np.zeros(shape = (len(path)//2,state_dim),dtype=int)
                acts = []
                for j in  range(0,len(path)-1,2):
                    ents.append(path[j])
                    cur_vec = entity2vec_data[entity2id_dict[path[j]]]
                    states[j//2] = np.concatenate((cur_vec,target_vec-cur_vec))
                ents.append(path[-1])
                for j in  range(1,len(path),2):
                    rels.append(path[j])
                    acts.append(relation2id_dict[path[j]])
                actions = np.array(acts)
                p_nn.train_batch(states,actions)
    saver.save(sess, '.\\models\\policy_supervised_' + relation)
    print('Model saved')


# In[ ]:




