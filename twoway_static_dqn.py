
# coding: utf-8

# In[1]:

import sys
sys.path.append('/Users/vitorhadad/Documents/')


# In[2]:

from collections import deque
import tensorflow as tf
import numpy as np
from kidney.environments.twoway import TwoWayKidneyExchange
import tensorflow.contrib.layers as skflow
import matplotlib.pyplot as plt
import pandas as pd
import random
from IPython.display import clear_output
get_ipython().magic('matplotlib inline')


# In[3]:

ENV_NAME = 'twoway_static'  # Environment name
NUM_PAIRS = 20

GAMMA = .99  # Discount factor
LAMBDA = .8  # Regularization factor
KEEP_PROB = .8  # Dropout keep prob rate

NUM_EPISODES = 10000000  # Number of episodes the agent plays
EXPLORATION_STEPS = 500000 # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 0.2  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.01  # Final value of epsilon in epsilon-greedy
INITIAL_REPLAY_SIZE = 256  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 2000  # Number of replay memory the agent uses for training
BATCH_SIZE = 16  # Mini batch size
TARGET_UPDATE_INTERVAL = 250  # The frequency with which the target network is updated
TEST_EVERY = 100
TRAIN_INTERVAL = 1  # The agent selects 4 actions between successive updates
LEARNING_RATE = 0.025  # Learning rate used by optimizer
MOMENTUM = 0.995  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update

TRAIN = True

LOAD_NETWORK = False
LOAD_NETWORK_PATH = None

SAVE_NETWORK_PATH = 'saved_networks/' + ENV_NAME
SUMMARY_PATH = 'summary/' + ENV_NAME
FIGURE_PATH = 'figures/' + ENV_NAME

PRINT_EVERY = 100
PLOT_EVERY = 250
SAVE_EVERY = 5000
SUMMARY_EVERY = 200


# In[4]:

num_actions = 45
state_dim = 16
discount_factor = 1


# In[5]:

class Agent():
    
    def __init__(self, state_dim, num_actions):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.t = 0

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0
        
        self.training_loss = []

        # Create replay memory
        self.replay_memory = deque(maxlen = NUM_REPLAY_MEMORY)


        # Create q network
        with tf.variable_scope("q_network"):
            self.s, self.kprob, self.q_values = self.build_network()
        self.q_network_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_network")
        for v in self.q_network_weights:
            self.variable_summaries(v, v.name)
        
        # Create target network
        with tf.variable_scope("target_network"):
            self.st, self.kprobt, self.target_q_values = self.build_network()
        self.target_network_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_network")        
        for v in self.target_network_weights:
            self.variable_summaries(v, v.name)
        
        
        with tf.name_scope("train"):
            self.a, self.y, self.loss, self.grad_update = self.build_training_op(self.q_network_weights)

        self.merged = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter(SUMMARY_PATH + '/train', sess.graph)
        self.test_writer = tf.train.SummaryWriter(SUMMARY_PATH + '/test')
        
        self.init = tf.initialize_all_variables()
        self.uninit = tf.report_uninitialized_variables()
        
        self.sess = tf.InteractiveSession()
        
        self.init.run()
        
        self.saver = tf.train.Saver()
    
    def save_network(self):
        save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH)
        return save_path
    
    def summary_network(self):
        summary = sess.run(self.merged)
        self.train_writer.add_summary(summary, self.t)
        print('Adding run metadata for', self.t)
      
    
    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.scalar_summary('stddev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)



    def build_network(self):
        s = tf.placeholder(tf.float32, (None, self.state_dim), name="states")
        kp = tf.placeholder(tf.float32)
        
        net = skflow.fully_connected(s, 5,
                                    weights_initializer = skflow.xavier_initializer(uniform = False),
                                    weights_regularizer = skflow.l2_regularizer(LAMBDA))
        net = skflow.dropout(net, kp)
        net = skflow.fully_connected(net, 5,
                                    weights_initializer = skflow.xavier_initializer(uniform = False),
                                    weights_regularizer = skflow.l2_regularizer(LAMBDA))
        net = skflow.dropout(net, kp)
        q_values = skflow.fully_connected(net, self.num_actions,
                                    weights_initializer = skflow.xavier_initializer(uniform = False),
                                    weights_regularizer = skflow.l2_regularizer(LAMBDA))
        return s, kp, q_values
    
    def get_action(self, state, avail, train = True):
        if train:
            if np.random.uniform() <= self.epsilon:
                avail_actions = np.arange(self.num_actions)[avail.flatten()]
                action = np.random.choice(avail_actions)
            else:
                qvals = self.q_values.eval(feed_dict={self.s: state, self.kprob: KEEP_PROB})
                qvals[~avail] = -np.inf 
                action = np.argmax(qvals)
        else: # test
            qvals = self.q_values.eval(feed_dict={self.s: state, self.kprob: 1.0})
            qvals[~avail] = -np.inf 
            action = np.argmax(qvals)
          
        if train:
            # Anneal epsilon linearly over time
            if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
                self.epsilon -= self.epsilon_step
        return action

    
    def run(self, state, action, reward, next_state, terminal):
        
        # Store transition in replay memory
        self.replay_memory.append((state, action, reward, next_state, terminal))
       
        if self.t >= INITIAL_REPLAY_SIZE and TRAIN:
            # Train network
            if self.t % TRAIN_INTERVAL == 0:
                loss = self.train_network()
                self.training_loss.append(loss)
                
            # Update target network
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.update_target_network()
            
        self.t += 1
        

    def update_target_network(self):
        for v_q, v_target in zip(self.q_network_weights, self.target_network_weights):
            v_target.assign(v_q)
    
    
    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert to appropriate types
        state_batch = np.vstack(state_batch).astype(np.float32)
        reward_batch = np.vstack(reward_batch).astype(np.float32)
        action_batch = np.vstack(action_batch).astype(np.float32)
        next_state_batch = np.vstack(next_state_batch).astype(np.float32)
        terminal_batch = np.vstack(terminal_batch).astype(np.int32)
        
        # Find max_a Qhat(s[t+1],a)
        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: next_state_batch, self.kprobt: 1})
        y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values_batch, axis=1).reshape(-1, 1)

        # Compute loss and update
        loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
            self.s: state_batch,
            self.kprob: KEEP_PROB,
            self.a: action_batch,
            self.y: y_batch
        })
        return loss

        
    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None,1], name = "a")
        y = tf.placeholder(tf.float32, [None,1], name = "y")

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.mul(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        #optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        #optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        grad_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grad_update



# In[6]:

sess = tf.Session()
env = TwoWayKidneyExchange(init_size = NUM_PAIRS, static = True)
agent = Agent(num_actions = 45, state_dim = 16)
losses = deque(maxlen = 100)
match_prob = []


# In[ ]:

for i_episode in range(NUM_EPISODES):
    
    if i_episode % 4999 == 0: clear_output()
        
    terminal = False
    state = env.reset(True)
    avail = env.available_actions()
    max_patients = env.get_max_patients()
    if max_patients == 0: 
        i_episode -= 1
        continue
    matched_patients = 0
    while not terminal:
        action = agent.get_action(state, avail)
        next_state, reward, terminal, avail = env.step(action, False)
        matched_patients += 2
        if terminal: 
            penalty = matched_patients - max_patients
            reward += penalty 
        agent.run(state, action, reward, next_state, terminal)

    losses.append(agent.training_loss)
    
    
    if i_episode % PRINT_EVERY == 0 and len(losses) > 20:
        print(i_episode, np.mean(losses), agent.epsilon)
        
    if i_episode % PLOT_EVERY == 0:
        fig, ax = plt.subplots(1, 2, figsize = (15, 3))
        ax[0].plot(match_prob)
        if len(match_prob) > 100:
            pd.Series(match_prob).rolling(100).mean().plot(ax = ax[1])
        fig.savefig(FIGURE_PATH + "/match_prob")
        plt.close("all")
        
    if i_episode % TEST_EVERY == 0:
        eps = agent.epsilon
        agent.epsilon = 0
        terminal = False
        state = env.reset(True)
        avail = env.available_actions()
        max_patients = env.get_max_patients()
        if max_patients == 0: continue
        matched_patients = 0
        while not terminal:
            action = agent.get_action(state, avail)
            next_state, reward, terminal, avail = env.step(action, False)
            matched_patients += 2
            if terminal: 
                penalty = matched_patients - max_patients
                reward += penalty 

        print("Matched {} out of {} possible".format(matched_patients, max_patients))
        match_prob.append(matched_patients / max_patients)
        agent.epsilon = eps
        
    if i_episode % SAVE_EVERY == 0 and i_episode > 1:
        save_path = agent.save_network()
        print("Model saved in file: %s" % save_path)
        
    if i_episode % SUMMARY_EVERY == 0 and i_episode > 1:
        summary = agent.sess.run(agent.merged)
        agent.train_writer.add_summary(summary, agent.t)
        agent.train_writer.close()


# In[ ]:



