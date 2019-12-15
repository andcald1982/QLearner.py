# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 14:45:53 2019

@author: ANDRES CALDERON
"""

import gym
import numpy as np


# EPSILON_MIN : vamos aprendiendo, mientras que el incremento de aprendizaje sea superior a dicho valor
# MAX_NUM_EPISODES:numero maximo de iteracion que estamos dispuestos a realizar
# STEPS_PER_EPISODES: numero maximo de pasos a realizar en cada episodio
# ALPHA: ratio de aprendizaje del agente
# GAMMA: factor de descuente de un paso a otro (para lograr que el agente aprenda en el menor numero de pasos posible)
# NUM_DISCRETE_BINS: numero de divisiones en el caso de discretizar el espacio de estados continuo
EPSILON_MIN = 0.005 #Epsilon representa la probabilidad de que el agente se equivoque
MAX_NUM_EPISODES = 50000
STEPS_PER_EPISODE = 200
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 30

# implementacion de la ecuacion de Q-Learning
class QLearner(object):
    def __init__(self, environment):
        self.obs_shape = environment.observation_space.shape
        self.obs_high = environment.observation_space.high
        self.obs_low = environment.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS
        self.bin_width = (self.obs_high-self.obs_low)/self.obs_bins
        
        self.action_shape = environment.observation_space.n
        self.q = np.zeros(self.obs_bins+1, self.obs_bins+1, self.action_shape) # matriz de 31 x 31 x 3
        self.gamma = GAMMA
        self.alpha = ALPHA
        self.epsilon = 1.0

    # metodo para discretizar los valores observados
    def discretize (self, obs):
        (return tuple ((obs-self.obs_low)/self.bin_width).astype(int))

    #metodo para definir accion del agente
    def get_action(self, obs):
        discrete_obs = self.discretize(obs)
        # Seleccion de la accion en base a Epsilon-Greedy
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon:
            return np.argmax(self.q[discrete_obs])  #con probabilidad 1-epsilon, elegimos la mejor posible
        else:
            return np.random.choice([a for a in range(self.action_shape)]) #con probabilidad epsilon, elegimos uno al azar            
    
    #metodo para implementar la funcion de Q-Learning   
    def learn(self, obs, action, reward, next_obs):
        discrete_obs = self.discretize(obs)
        discrete_next_obs = self.discretize(next_obs)
        td_target = reward + self.gamma * np.max(self.q[disecrete_next_obs]) # R(s,a) + gamma*maxQ(s´,a´)
        td_error = td_target - self.q[discrete_obs][action] # R(s,a) + gamma*maxQ(s´,a´) - Q[t-1](s,a)
        self.q[discrete_obs][action] += self.alpha*td_error # ecuacion de Bellman implementada: Q[t](s,a) = Q[t-1](s,a) + alpha(R(s,a) + gamma*maxQ(s´,a´) - Q[t-1](s,a))

## Metodo para entrenar al agente
def train(agent, environment):
    best_reward = -float('inf')
    for episode in range(MAX_NUM_EPISODES):
        done = false
        obs = environment.reset()
        total_reward = 0.0
        while not done:
            action = agent.get_action(obs) #Accion elegida segun la ecuacion de Q-Learning
            next_obs, reward, done, info = environment.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward
        if total_reward > best_reward:
            best_reward = total_reward
        print("Episodio Numero {} Con Recompensa: {}, Mejor Recompensa: {}, epsilon:{}".format(episode, total_reward, best_reward, agent.epsilon))
    ## De todas las polticas de entrenamiento que hemos obtenido devolvemos la mejor de todas
    return (np.argmax(agent.Q, axis = 2))

## Metodo para hacer test del entrenamiento
def test(agent, environment, policy):
    done = false
    obs = environment.reset()
    total_reward = 0.0
    while noto done:
        action = policy[agent.discretize(obs)]
        next_obs, reward, done, info = environment.step(action)  
        obs = next_obs
        total_reward += reward
    return total_reward

## Metodo para testear y guardar
    
if __name__ == "__main__":
    environment = gym.make("MountainCar-v0")
    agent = QLearner(environment)
    learned_policy = train (agent, environment)
    monitor_path = "./monitor_output"
    environment = gym.wrappers.Monitor(environment, monitor_path, force = True)
    for _ in range(1000):
        test(agent, environment, learned_policy)
    environment.close()