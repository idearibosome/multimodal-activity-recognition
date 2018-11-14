import socket

import numpy as np
import tensorflow as tf

from dataloader import DataLoader

class Shared:
  
  TRAIN_DATA_LIST = [
      'S1-ADL1.dat',
      'S1-ADL2.dat',
      'S1-ADL3.dat',
      'S1-ADL4.dat',
      'S1-ADL5.dat',
      'S1-Drill.dat',
      'S2-ADL1.dat',
      'S2-ADL2.dat',
      'S2-Drill.dat',
      'S3-ADL1.dat',
      'S3-ADL2.dat',
      'S3-Drill.dat'
  ]
  VALIDATION_DATA_LIST = [
      'S2-ADL3.dat',
      'S3-ADL3.dat'
  ]
  TEST_DATA_LIST = [
      'S2-ADL4.dat',
      'S2-ADL5.dat',
      'S3-ADL4.dat',
      'S3-ADL5.dat'
  ]
  
  def __init__(self):
    self.data_normalization = 'rescale'
    self.data_base_src = '/tmp/data/'
  
  
  def load_train_data(self):
    self.train_data_loader = DataLoader(self.data_base_src, normalization=self.data_normalization)
    
    for filename in Shared.TRAIN_DATA_LIST:
      self.train_data_loader.load_data(filename)
  
  
  def load_validation_data(self):
    self.validation_data_loader = DataLoader(self.data_base_src, normalization=self.data_normalization)
    
    for filename in Shared.VALIDATION_DATA_LIST:
      self.validation_data_loader.load_data(filename)
  
  
  def load_test_data(self):
    self.test_data_loader = DataLoader(self.data_base_src, normalization=self.data_normalization)
    
    for filename in Shared.TEST_DATA_LIST:
      self.test_data_loader.load_data(filename)
  
  
  def get_prob_result(self, num_classes, class_list, label_list):
    accuracy = np.sum(class_list == label_list) / len(label_list)
    
    f1_scores = 0.0
    for i in range(num_classes):
      tp = np.sum((class_list == i) & (label_list == i))
      tn = np.sum((class_list != i) & (label_list != i))
      fp = np.sum((class_list == i) & (label_list != i))
      fn = np.sum((class_list != i) & (label_list == i))
    
      precision = 0.0
      if (tp + fp > 0):
        precision = tp / (tp + fp)
      
      recall = 0.0
      if (tp + fn > 0):
        recall = tp / (tp + fn)
      
      f1_score = 0.0
      if (precision + recall > 0):
        f1_score = 2.0 * (np.sum(label_list == i) / len(label_list))
        f1_score = f1_score * (precision * recall) / (precision + recall)
      f1_scores += f1_score
    
    f1_scores_except_null = 0.0
    for i in range(1, num_classes):
      tp = np.sum((label_list != 0) & (class_list == i) & (label_list == i))
      tn = np.sum((label_list != 0) & (class_list != i) & (label_list != i))
      fp = np.sum((label_list != 0) & (class_list == i) & (label_list != i))
      fn = np.sum((label_list != 0) & (class_list != i) & (label_list == i))
    
      precision = 0.0
      if (tp + fp > 0):
        precision = tp / (tp + fp)
      
      recall = 0.0
      if (tp + fn > 0):
        recall = tp / (tp + fn)
      
      f1_score = 0.0
      if (precision + recall > 0):
        f1_score = 2.0 * (np.sum(label_list == i) / np.sum(label_list != 0))
        f1_score = f1_score * (precision * recall) / (precision + recall)
      f1_scores_except_null += f1_score
    
    return accuracy, f1_scores, f1_scores_except_null
    
  
  def cnn_1d_inference(self, input_data, input_depth, conv_size, conv_depth, padding='SAME'):
    kernel = tf.get_variable(
        'weights', 
        [conv_size, input_depth, conv_depth], 
        dtype=tf.float32
    )
    biases = tf.get_variable(
        'biases', 
        [conv_depth], 
        dtype=tf.float32
    )
    conv = tf.nn.conv1d(input_data, kernel, 1, padding=padding)
    conv = tf.nn.bias_add(conv, biases)
    
    return conv
  
  
  def cnn_2d_inference(self, input_data, input_depth, conv_size, conv_depth, padding='SAME'):
    kernel = tf.get_variable(
        'weights', 
        [conv_size, conv_size, input_depth, conv_depth],
        dtype=tf.float32
    )
    biases = tf.get_variable(
        'biases', 
        [conv_depth], 
        dtype=tf.float32
    )
    conv = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding=padding)
    conv = tf.nn.bias_add(conv, biases)
    
    return conv
  
  
  def cnn_2d_unequal_inference(self, input_data, input_depth, conv_width, conv_height, conv_depth, padding='SAME'):
    kernel = tf.get_variable(
        'weights', 
        [conv_width, conv_height, input_depth, conv_depth], 
        dtype=tf.float32
    )
    biases = tf.get_variable(
        'biases', 
        [conv_depth], 
        dtype=tf.float32
    )
    conv = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding=padding)
    conv = tf.nn.bias_add(conv, biases)
    
    return conv
  
  
  def dcnn_2d_unequal_inference(self, input_data, input_depth, conv_width, conv_height, conv_depth, strides=(1, 1), padding='SAME'):
    conv = tf.layers.conv2d_transpose(input_data, filters=conv_depth, kernel_size=(conv_width, conv_height), strides=strides, padding=padding)
    
    return conv
  
  
  def fully_connected(self, input_data, input_size, output_size):
    reshape = tf.reshape(input_data, [-1, input_size])
    weights = tf.get_variable(
        'weights', 
        [input_size, output_size], 
        dtype=tf.float32
    )
    biases = tf.get_variable(
        'biases', 
        [output_size], 
        dtype=tf.float32
    )
    output_data = tf.matmul(reshape, weights) + biases
    
    return output_data
  