import os
import time

import numpy as np
import tensorflow as tf

from model import Model

TARGET_CLASS = 'ml' # ml, loc
TRAIN_DIR = '/tmp/mmactrec/target_%s' % TARGET_CLASS
MAX_STEP = 800000

DATA_SIZE_LIST = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 9, 9, 9, 9, 9, 16, 16]
TARGET_DATA_LIST = [
    'acc_rkn^', 'acc_hip', 'acc_lua^', 'acc_rua_', 'acc_lh', 'acc_back', 'acc_rkn_', 'acc_rwr', 'acc_rua^', 'acc_lua_', 'acc_lwr', 'acc_rh',
    'ine_back', 'ine_rua', 'ine_rla', 'ine_lua', 'ine_lla', 'ine_lshoe', 'ine_rshoe'
]

EXCLUDE_NULL_CLASS = False

def train(train_dir):
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  
  class ModelConfig(object):
    pass
  
  config = ModelConfig()
  config.batch_size = 64
  config.max_step = MAX_STEP
  config.sequence_length = 15
  
  if (TARGET_CLASS == 'ml'):
    config.num_actions = 18
  else:
    config.num_actions = 5
  
  with tf.Graph().as_default():
    model = Model(
        is_training=True,
        config=config
    )
    model.shared.data_normalization = 'rescale'
    model.shared.load_train_data()
    
    model.shared.train_data_loader.fill_missing_data()
    
    
    global_step = tf.contrib.framework.get_or_create_global_step()
    
    learning_rate = tf.placeholder(tf.float32)
    label_list = tf.placeholder(tf.int64)
    
    model.build(DATA_SIZE_LIST)
    
    logit_list = tf.reshape(model.graph.final.output, [model.config.batch_size, -1, model.config.num_actions])
    
    last_logit_list = tf.reshape(logit_list[:, -1, :], [model.config.batch_size, model.config.num_actions])
    last_label_list = tf.reshape(label_list[:, -1], [model.config.batch_size])
    
    loss_list = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=last_logit_list, labels=last_label_list)
    total_loss = tf.reduce_mean(loss_list)
    
    tf.summary.scalar('total_loss', total_loss)
    
    opt = tf.train.AdamOptimizer(learning_rate, epsilon=1e-2)
    grads = opt.compute_gradients(total_loss)
    
    train_op = opt.apply_gradients(grads, global_step=global_step)
    
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=200)
    summary_op = tf.summary.merge_all()
        
    init = tf.global_variables_initializer()

    
    # normal session  
    config_train = tf.ConfigProto(
        log_device_placement=False,
        allow_soft_placement=True
    )
    #config_train.gpu_options.allow_growth = True
    sess = tf.Session(config=config_train)
    
    sess.run(init)
    
    tf.train.start_queue_runners(sess=sess)
    
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
    
    
    # num trainable parameters
    total_variable_parameters = 0
    for variable in tf.trainable_variables():
      # shape is an array of tf.Dimension
      shape = variable.get_shape()
      variable_parameters = 1
      for dim in shape:
        variable_parameters *= dim.value
      print(' - %s: %d' % (variable.name, variable_parameters))
      total_variable_parameters += variable_parameters
    print('total: %d' % (total_variable_parameters))
    
    
    for step in range(0, MAX_STEP+1):
      start_time = time.time()
      
      # learning rate
      r_learning_rate = 1e-3
      
      
      # data and toggle
      r_data_list = []
      r_label_list = []
      for _ in range(len(TARGET_DATA_LIST)):
        r_data_list.append([])
      
      for _ in range(model.config.batch_size):
        while True:
          data_list_original = model.shared.train_data_loader.get_random_range_data(model.config.sequence_length)
          if (EXCLUDE_NULL_CLASS and (data_list_original[-1][TARGET_CLASS] == 0)):
            continue
        
          r_batch_data_list = []
          r_batch_data_missing = []
          r_batch_label_list = []
          for _ in range(len(TARGET_DATA_LIST)):
            r_batch_data_list.append([])
            r_batch_data_missing.append(False)
        
          r_sequence_length = len(data_list_original)
          for each_data in data_list_original:
            for (i, data_name) in enumerate(TARGET_DATA_LIST):
              if (np.sum(np.isnan(each_data[data_name])) > 0):
                r_batch_data_list[i].append([0] * len(each_data[data_name]))
                r_batch_data_missing[i] = True
              else:
                r_batch_data_list[i].append(each_data[data_name])
            
            if (np.sum(r_batch_data_missing) >= len(r_batch_data_missing)):
              break
            r_batch_label_list.append(each_data[TARGET_CLASS])
          
          if (np.sum(r_batch_data_missing) >= len(r_batch_data_missing)):
            continue
        
          for i in range(len(TARGET_DATA_LIST)):
            if (r_batch_data_missing[i]):
              for j in range(len(r_batch_data_list[i])):
                r_batch_data_list[i][j] = [0] * len(r_batch_data_list[i][j])
          
          for i in range(len(TARGET_DATA_LIST)):
            r_data_list[i].append(r_batch_data_list[i])
          
          r_label_list.append(r_batch_label_list)
          break
      
      
      # feed dict
      feed_dict = {
          learning_rate: r_learning_rate,
          label_list: r_label_list
      }
      for i in range(len(TARGET_DATA_LIST)):
        feed_dict[model.feeds.single[i].input] = r_data_list[i]
      
      
      # run
      if (step > 0) and (step % 200 == 0):
        _, r_final_class_data, r_total_loss, summary_str = sess.run(
            [train_op, model.graph.final.predicted, total_loss, summary_op],
            feed_dict=feed_dict
        )
        summary_writer.add_summary(summary_str, step)
      else:
        _, r_final_class_data, r_final_output_data, r_total_loss = sess.run(
            [train_op, model.graph.final.predicted, model.graph.final.output, total_loss],
            feed_dict=feed_dict
        )
      
      duration = time.time() - start_time

      assert not np.isnan(r_total_loss), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        sec_per_batch = float(duration)
        accuracy = np.sum(np.array(r_final_class_data)[:, -1].flatten() == np.array(r_label_list)[:, -1].flatten()) / len(np.array(r_final_class_data)[:, -1].flatten())
        print('step %d, accuracy = %.6f, loss = %.6f (%.3f sec/batch)' % (step, accuracy, r_total_loss, sec_per_batch))

      if (step > 0) and (step % 5000 == 0):
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
    
    

def main(argv=None):
  if tf.gfile.Exists(TRAIN_DIR):
    tf.gfile.DeleteRecursively(TRAIN_DIR)
  tf.gfile.MakeDirs(TRAIN_DIR)
  
  train(TRAIN_DIR)


if __name__ == '__main__':
  tf.app.run()
    