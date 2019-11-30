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
import numpy as np


# In[4]:


import keras


# In[5]:


dataPath = '.\\NELL-995\\'


# In[6]:


relation = "concept_athletehomestadium"


# In[7]:


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


# In[8]:


num_paths = 5


# In[9]:


graphpath = dataPath + 'tasks\\' + relation + '\\' + 'graph.txt'
relationPath = dataPath + 'tasks\\' + relation + '\\' + 'train_pos'


# In[10]:


import keras.backend as k


# In[11]:


from keras.layers import Input,Dense
from keras.models import Model
from keras.optimizers import Adam


# In[12]:


import tensorflow as tf


# In[13]:


f = open(relationPath)
train_data = f.readlines()
f.close()


# In[14]:


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


# In[15]:


f = open(dataPath + 'kb_env_rl.txt')
kb_data1 = f.readlines()
f.close()

kb_data = []
rel = ":".join(relation.split("_"))
for line in kb_data1:
    relation1 = line.split()[2]
    kb_data.append(line) if ((relation1 != rel + '_inv') and (relation1 != rel)) else None


# In[16]:


f = open(graphpath)
graph_data = f.readlines()
f.close()


# In[17]:


nodes = dict()
for line in graph_data:
    e1, rel, e2 = line.rsplit()
    if e1 not in nodes:
        nodes[e1] = list()
    nodes[e1].append((rel, e2))


# In[18]:


import tensorflow as tf
from tensorflow.train import AdamOptimizer
from tensorflow.contrib.layers import l2_regularizer, xavier_initializer


# In[36]:


class Grad_policy(object):
    def __init__(self, state_size, action_size, lr = 0.001):
        self.init = xavier_initializer()
        with tf.variable_scope('supervised_policy'):
            self.st = tf.placeholder(tf.float32, [None, state_size], name = 'st')
            self.acts_prob = self.sl_policy_nn(self.st, state_size, action_size, self.init)
            self.act = tf.placeholder(tf.int32, [None], name = 'act')
            self.reward = tf.placeholder(tf.float32, name = 'reward')
            
            act_mask = tf.cast(tf.one_hot(self.act, depth = action_size), tf.bool)
            self.act_prob = tf.boolean_mask(self.acts_prob, act_mask)

            self.loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope = 'supervised_policy')) + tf.reduce_sum(-tf.log(self.act_prob)*self.reward)
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

    def train_batch(self, st, act, reward, sess = None):
        sess = sess or tf.get_default_session()
        _, loss = sess.run([self.training_op, self.loss], {self.st: st, self.act: act, self.reward: reward})
        return loss


# In[20]:


def bfs(nodes1,e1,e2):
    visited = set()
    path = list()
    q = Queue()
    q.put(e1)
    visited.add(e1)
    path.append(e1)
    while(not q.empty()):
        ent = q.get()
        for link in nodes1[ent]:
            if link[1] not in visited:
                visited.add(link[1])
                q.put(link[1])
                path.extend(list(link))
            if link[1] == e2:
                return path, True
    return [], False


# In[21]:


def interact(curr,target,rel):
    choices = []
    done = 0
    for line in kb_data:
        triple = line.rsplit()
        ent1_id = entity2id_dict[triple[0]]

        if (curr == ent1_id) and (triple[2] == rel) and (triple[1] in entity2id_dict):
            choices.append(triple[1])
    if len(choices) == 0:
        reward = -1
#         die += 1
#         next_state = state # stay in the initial state
#         next_state[-1] = self.die
        return (reward, curr, None, done)
    else: # find a valid step
        ch = random.choice(choices)
#         self.path.append(path[2] + ' -> ' + path[1])
#         self.path_relations.append(path[2])
        # print 'Find a valid step', path
        # print 'Action index', action
#         die = 0
        next_id = entity2id_dict[ch]
        reward = 0
#         new_state = [new_pos, target_pos, self.die]

        if next_id == target:
#             print('Find a path:', self.path)
            done = 1
            reward = 0
            new_state = None
        return (reward, next_id, ch,done)


# In[ ]:


paths_found = []

for path in paths_trav:
    paths_found.append(' -> '.join(path))

paths_num = list(collections.Counter(paths_found).items())
paths_num = sorted(paths_num, key = lambda x:x[1], reverse=True)

f = open(dataPath + 'tasks\\' + relation + '\\' + 'path_stats.txt', 'w')
for tup in paths_num:
    f.write(tup[0]+'\t'+str(tup[1])+'\n')
f.close()
print('Path stats saved')


# In[29]:


die = 0

tf.reset_default_graph()
p_nn = Grad_policy(state_dim, action_space)
hits = 0
hits_list = []
reward_list = []
paths_trav = []
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, '.\\models\\policy_supervised_' + relation)
    for ep in tqdm(range(min(len(train_data),3))):
        source = train_data[ep].split()[0]
        target = train_data[ep].split()[1]
        # sample = train_data[0].split()
        # state_idx = [entity2id_dict[sample[0]], entity2id_dict[sample[1]], 0]

        episode = []
        state_batch_negative = []
        action_batch_negative = []

        state_batch_positive = []
        action_batch_positive = []

        curr_id = entity2id_dict[source]
        target_id = entity2id_dict[target]
        done = 0
        die = 0
        path_rel = []
        path_ent = [source]
        for t in range(max_steps):

            curr_vec = entity2vec_data[curr_id]
            target_vec = entity2vec_data[target_id]
            st_vec = np.concatenate((curr_vec, target_vec - curr_vec))

        #     state_vec = env.idx_state(state_idx)
    #         print(st_vec.shape)
    #         print(p_nn.model.predict(st_vec.reshape((1,-1))))
            action_probs = p_nn.get_act_probs(st_vec.reshape((1,-1)))
            action_chosen = np.random.choice(np.arange(action_space), p = np.squeeze(action_probs))
    #         break
            rel = relations[action_chosen]

            reward, next_id, next_ent,done = interact(curr_id,target_id,rel)


#             act_array = np.zeros(shape=(action_space,),dtype='float64')
#             act_array[action_chosen] = 1
            if reward == -1: # the action fails for this step
                state_batch_negative.append(st_vec)
                action_batch_negative.append(action_chosen)
                die += 1
            else:
                state_batch_positive.append(st_vec)
                action_batch_positive.append(action_chosen)
                path_rel.append(rel)
                path_ent.append(next_ent)
                die = 0
        #     new_state_vec = env.idx_state(new_state)
        #     episode.append(Transition(state = state_vec, action = action_chosen, next_state = new_state_vec, reward = reward))

            if done:
                break

            curr_id = next_id 
    #         state_idx = new_state
    #     break
        if len(state_batch_negative) > 0:
            nn_target = -0.05
            p_nn.train_batch(np.array(state_batch_negative).reshape((-1,200)),np.array(action_batch_negative),nn_target)

        if done:
            paths_trav.append(path_rel)
            length_reward = 1/len(path_rel)
            global_reward = 1

            nn_target = 0.1*global_reward + 0.9*length_reward # total reward
            reward_list.append(nn_target)
            p_nn.train_batch(np.array(state_batch_positive).reshape((-1,200)),np.array(action_batch_positive),nn_taget)
            hits += 1
        else:
            global_reward = -0.05
            nn_target = global_reward # total reward
            reward_list.append(nn_target)
            p_nn.train_batch(np.array(state_batch_positive).reshape((-1,200)),np.array(action_batch_positive),nn_target)


            # teacher guidline for failed episode
            inter = None
            while True:
                inter = random.choice(list(nodes.keys()))
                if inter not in [source,target]:
                    break

            path1, found1 = bfs(nodes,source,inter)
            path2, found2 = bfs(nodes,inter,target)
            if found1 and found2:
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
                nn_target = 1 # reward
                p_nn.train_batch(states,actions,nn_target)
            
            hits_list.append(hits)
    saver.save(sess, '.\\models\\policy_retrained' + relation)

plt.plot([i for i in range(1,len(reward_list)+1)],reward_list)
plt.show()
plt.plot([i for i in range(1,len(hits_list)+1)],hits_list)

paths_found = []

for path in paths_trav:
    paths_found.append(' -> '.join(path))

from collections import Counter
paths_num = list(Counter(paths_found).items())
paths_num = sorted(paths_num, key = lambda x:x[1], reverse=True)

f = open(dataPath + 'tasks\\' + relation + '\\' + 'path_stats.txt', 'w')
for tup in paths_num:
    f.write(tup[0]+'\t'+str(tup[1])+'\n')
f.close()
print('Path stats saved')


# In[30]:


def path_embedding(path):
    embs = [relation2vec_data[relation2id_dict[rel],:] for rel in path]
    embs = np.reshape(embs, (-1,embedding_dim))
    return np.reshape(np.sum(embs, axis=0),(-1, embedding_dim))


# In[39]:


tf.reset_default_graph()
p_nn = Grad_policy(state_dim, action_space)

f = open(relationPath)
test_data = f.readlines()
f.close()

test_num = len(test_data)

success = 0

saver = tf.train.Saver()
path_found = []
path_relation_found = []
path_set = []

with tf.Session() as sess:
    saver.restore(sess, 'models\\policy_retrained' + relation)

    if test_num > 500:
        test_num = 500
    test_num = 3
    path_rel =[]
    path_ent = []
    for ep in range(test_num):
        source = test_data[ep].split()[0]
        target = test_data[ep].split()[1]
        curr_id = entity2id_dict[source]
        target_id = entity2id_dict[target]
        state_batch = []
        action_batch = []
        t = 0
        while True:
            t+=1
            curr_vec = entity2vec_data[curr_id]
            target_vec = entity2vec_data[target_id]
            st_vec = np.concatenate((curr_vec, target_vec - curr_vec))

            action_probs = p_nn.get_act_probs(st_vec.reshape((1,-1)))
            action_chosen = np.random.choice(np.arange(action_space), p = np.squeeze(action_probs))
            rel = relations[action_chosen]
            reward, next_id, next_ent,done = interact(curr_id,target_id,rel)
            path_rel.append(rel)
            state_batch.append(st_vec)
            action_batch.append(action_chosen)

            if done or t == max_steps_test:
                if done:
                    success += 1
                    path = ' -> '.join(path_rel)
                    path_found.append(path)
                break
            curr_id = next_id 

        if done:
            if len(path_set) != 0:
                path_found_embs = [path_embedding(path) for path in path_set]
                curr_path_embs = path_embedding(path_rel)
                path_found_embs = np.reshape(path_found_embs, (-1,embedding_dim))
                from sklearn.metrics.pairwise import cosine_similarity
                cos_sim = cosine_similarity(path_found_embs, curr_path_embs)
                div_reward = -np.mean(cos_sim)
                p_nn.update(np.reshape(state_batch,(-1,state_dim)), action_batch, 0.1*div_reward)
            if path_rel not in path_set:
                path_set.append(path_rel)


for path in path_found:
    path_relation_found.append(' -> '.join(path_rel))

from collections import Counter
paths_num = list(Counter(path_relation_found).items())
paths_num = sorted(paths_num, key = lambda x:x[1], reverse=True)
from collections import Counter
relation_path_stats = list(Counter(path_relation_found).items())
relation_path_stats = sorted(relation_path_stats, key = lambda x:x[1], reverse=True)

ranking_path = []
for tup in paths_num:
    path = tup[0]
    length = len(path.split(' -> '))
    ranking_path.append((path, length))

ranking_path = sorted(ranking_path, key = lambda x:x[1])
print('Success percentage :', success/test_num)

f = open(dataPath + 'tasks\\' + relation + '\\' + 'path_to_use.txt', 'w')
for tup in ranking_path:
    f.write(tup[0] + '\n')
f.close()


# In[ ]:




