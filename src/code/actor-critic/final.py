import collections
import gym
import numpy as np
import tensorflow as tf
import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

# This was run in a google collab notebook, copy and paste the code there for best results, running it locally too should work
# https://colab.research.google.com/drive/153_LXoeTUAoDkh80v1XG20M2bDxkhAj-#scrollTo=8mXTDwPNZ-pJ

# defining the actor critic network
class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self,
      num_actions: int,
      num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.common = layers.Dense(num_hidden_units, activation="relu")
    self.actor = layers.Dense(num_actions)
    self.critic = layers.Dense(1)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.common(inputs)
    # x will be the 128 or num_hidden_units neurons
    return self.actor(x), self.critic(x)


# reading CSV
df = pd.read_csv("soccerdatacsv.csv")
# dropping columns if shots is nan
df = df[df['Shots'].notna()]
# use shots, tackles, passes, dribbles, disp
# adjusting all entries to between 0 and 1 - need to zscore stuff
cols = [2,4,7,9,16,27]
df = df[df.columns[cols]]
df = df.apply(zscore)

# set up actor critic network
num_actions = 5
num_hidden_units = 32

model = ActorCritic(num_actions, num_hidden_units)
model.build((None,5))
model.summary()
a = np.array(model.layers[1].get_weights())
model.layers[1].set_weights(a + 0.1)
# start modeling with 5 attributes
start_state = np.array([1,1,1,1,1])
tensor_state = tf.constant([1,1,1,1,1], dtype=tf.float32)

# this is simulating an environment, basically everyone step will get a new row in the soccerdatacsv,
# to kinda model new stats and performance for a game just played in an environment
step = 0
def custom_env_step2(action: tf.Tensor) -> (tf.Tensor, float, bool):
  # check if done
  if step > len(df):
    return np.array([1,1,1]), 2.0, True
  # action affects state
  tempstate = tensor_state + action
  print('temp', tempstate)
  # calculate the reward, mutliply values by 10 so exponents work bc .5^2 is smaller than .5
  fromcsv = tf.convert_to_tensor(df.iloc[[step],[2,4,7,9,16]].values[0])
  fromcsv = tf.cast(fromcsv, tf.float32)
  model_predict_perf = tf.pow(fromcsv*10,tempstate)
  true_perf = tf.convert_to_tensor(df.iloc[step]['performance'])
  true_perf = tf.cast(true_perf, tf.float32)
  reward = tf.abs(10 / (tf.reduce_sum(model_predict_perf) - true_perf))
  return tempstate, reward, False

losses = []
# simulation for train data, this will return the actions or how to modify the model, the critic values telling you
# if your modification is good or bad, and the reward that you are trying to optimize for.
def run_perfomance_episode(
    initial_state: np.array,
    model: tf.keras.Model,
    max_steps: int) -> List[tf.Tensor]:
  """Runs a single episode to collect training data."""
  # action_probs = np.array([])
  # critic_values = np.array([])
  # rewards = np.array([])
  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  critic_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

  state = initial_state

  for t in range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
    state = tf.expand_dims(state, 0)

    # Run the model and to get action probabilities and critic value
    action_values, critic_value = model(state)
    b = tf.keras.activations.tanh(action_values)
    b = tf.round(b)
    # actions_list = [round(e, 0) for e in b.numpy()]

    # Store critic values
    # squeeze. Removes dimensions of size 1 from the shape of a tensor.
    critic_values = critic_values.write(0, tf.squeeze(critic_value))

    # Store log probability of the action chosen
    action_probs = action_probs.write(0, b)
    # action_probs =np.append(action_probs,actions_list)

    # Apply action to the environment to get next state and reward
    state, reward, done = custom_env_step2(b)

    # Store reward
    rewards = rewards.write(t, reward)

    action_probs = action_probs.stack()
    critic_values = critic_values.stack()
    rewards = rewards.stack()
  return action_probs, critic_values, rewards

#store actions, values, rewards in global vars
print('tensor state', tensor_state)
episoderesults = run_perfomance_episode(tensor_state,model,1 )
print(episoderesults)


# calculating actor critic loss, loss function
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
# values from critic, returns computed from rewards
def compute_loss(
    action_probs: tf.Tensor,
    values: tf.Tensor,
    returns: tf.Tensor) -> tf.Tensor:
  """Computes the combined actor-critic loss."""

  advantage = returns - values

  action_log_probs = tf.math.log(action_probs)
  # reduce sum is just summing all the elements
  actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

  critic_loss = huber_loss(values, returns)

  return actor_loss + critic_loss


# getting expected returns or sum of rewards, rewards now are worth more than rewards later, this helps fcn converge
def get_expected_return(
    rewards: tf.Tensor,
    gamma: float,
    standardize: bool = True) -> tf.Tensor:
  """Compute expected returns per timestep."""

  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)

  # Start from the end of `rewards` and accumulate reward sums
  # into the `returns` array
  rewards = tf.cast(rewards[::-1], dtype=tf.float32)
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward + gamma * discounted_sum
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
  returns = returns.stack()[::-1]

  if standardize:
    returns = ((returns - tf.math.reduce_mean(returns)) /
               (tf.math.reduce_std(returns) + eps))

  return returns

  # updating NN weights for optimization
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# initial_state: tf.Tensor,
# @tf.function
def train_step(
    initial_state: np.array,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    gamma: float,
    max_steps_per_episode: int) -> tf.Tensor:
  """Runs a model training step."""
  with tf.GradientTape() as tape:
    tape.watch(model.trainable_variables)
    # Run the model for one episode to collect training data
    action_probs, values, rewards  = run_perfomance_episode(initial_state,model,1 )
    print(rewards)
    print(get_expected_return(rewards, 0.95))
    # return ^^ in tensors to not destory the graph chain
    lossey = compute_loss(action_probs, values, rewards)
  grads = tape.gradient(lossey, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return episode_reward

new_state = tf.constant([1,1,1,1,1], dtype=tf.float32)

for i in range(5):
  print('episode', i)
  print('OG new', new_state)
  episoderesults = run_perfomance_episode(new_state,model,1 )
  print(episoderesults)
  # train_step(new_state,model,optimizer,0.8, max_steps_per_episode  )
  print(episoderesults[0][0])
  new_state = (episoderesults[0][0] + new_state)[0]
  print('new state' , new_state)
  print('equation weights' , new_state)
