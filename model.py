import numpy as np
import tensorflow as tf

from shared import Shared

class ModelObject(object):
  pass

class Model:
  
  DEFAULT_CONFIG = {
      'batch_size': 16,
      'conv_size': 3,
      'final_rnn_units': 128,
      'final_rnn_layers': 2,
      'num_actions': 18,
      'sequence_length': 15,
      'dropout_keep_prob': 0.5,
      'train_max_step': 200000
  }
  
  def __init__(self, is_training, config):
    self.is_training = is_training
    
    # config
    for (key, value) in self.DEFAULT_CONFIG.items():
      if not hasattr(config, key):
        setattr(config, key, value)
    self.config = config
    
    # shared
    self.shared = Shared()
    
    # graph and feeds
    self.graph = ModelObject()
    self.graph.single = []
    
    self.feeds = ModelObject()
    self.feeds.single = []
  
  
  def build(self, data_size_list):
    
    def final_rnn_cell():
      cell = tf.contrib.rnn.BasicLSTMCell(
          self.config.final_rnn_units,
          forget_bias=1.0,
          state_is_tuple=True
      )
      if ((self.is_training) and (self.config.dropout_keep_prob < 1.0)):
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.config.dropout_keep_prob)
      return cell
    
    
    # CNNs
    shared_size = 0
    for (i, data_size) in enumerate(data_size_list):
      single_graph = ModelObject()
      
      single_feed = ModelObject()
      single_feed.input = tf.placeholder(tf.float32)
    
      with tf.variable_scope('single%d/conv1' % i):
        conv1_input = tf.reshape(single_feed.input, [self.config.batch_size, self.config.sequence_length, data_size, 1])
        conv1_output = self.shared.cnn_2d_unequal_inference(conv1_input, 1, self.config.conv_size, 1, 64, padding='VALID')
        conv1_output = tf.nn.relu(conv1_output)
      
      with tf.variable_scope('single%d/conv2' % i):
        conv2_input = conv1_output
        conv2_output = self.shared.cnn_2d_unequal_inference(conv2_input, 64, self.config.conv_size, 1, 64, padding='VALID')
        conv2_output = tf.nn.relu(conv2_output)
      
      with tf.variable_scope('single%d/conv3' % i):
        conv3_input = conv2_output
        conv3_output = self.shared.cnn_2d_unequal_inference(conv3_input, 64, self.config.conv_size, 1, 64, padding='VALID')
        conv3_output = tf.nn.relu(conv3_output)
      
      with tf.variable_scope('single%d/conv4' % i):
        conv4_input = conv3_output
        conv4_output = self.shared.cnn_2d_unequal_inference(conv4_input, 64, self.config.conv_size, 1, 64, padding='VALID')
        conv4_output = tf.nn.relu(conv4_output)
      
      single_output = tf.reshape(conv4_output, [self.config.batch_size, (self.config.sequence_length-((self.config.conv_size-1)*4)), data_size * 64])
      
      single_graph.output = single_output
      shared_size += (data_size * 64)
      
      self.graph.single.append(single_graph)
      self.feeds.single.append(single_feed)
      
        
    # shared
    shared_representation = tf.concat([self.graph.single[i].output for i in range(len(self.graph.single))], axis=2)
    
    
    # RNNs
    with tf.variable_scope('rnn'):
      rnn_input = shared_representation
      rnn_cell = tf.contrib.rnn.MultiRNNCell([final_rnn_cell() for _ in range(self.config.final_rnn_layers)], state_is_tuple=True)
      rnn_initial_state = rnn_cell.zero_state(self.config.batch_size, tf.float32)
      rnn_sequence_length = [self.config.sequence_length-((self.config.conv_size-1)*4)] * self.config.batch_size
      rnn_output, rnn_last_state = tf.nn.dynamic_rnn(
          cell=rnn_cell,
          dtype=tf.float32,
          initial_state=rnn_initial_state,
          inputs=rnn_input,
          sequence_length=rnn_sequence_length,
          swap_memory=True
      )
    
    
    # Final
    final_graph = ModelObject()
    final_feed = ModelObject()
    
    with tf.variable_scope('final'):
      final_input = tf.reshape(rnn_output, [-1, self.config.final_rnn_units])
      final_output = self.shared.fully_connected(final_input, self.config.final_rnn_units, self.config.num_actions)
      final_output = tf.reshape(final_output, [self.config.batch_size, -1, self.config.num_actions])
      final_output_prob = tf.nn.softmax(final_output)
      final_output_class = tf.argmax(final_output_prob, axis=2)
      
    final_graph.output = final_output
    final_graph.prob = final_output_prob
    final_graph.predicted = final_output_class
    
    self.graph.final = final_graph
    self.feeds.final = final_feed
    
    