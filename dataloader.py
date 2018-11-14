import copy

import numpy as np

class DataLoader:
  
  LOC_MAPPING = {'NaN': 0, '0': 0, '1': 1, '2': 2, '4': 3, '5': 4}
  ML_MAPPING = {
      'NaN': 0,
      '0': 0,
      '406516': 1, # Open Door 1
      '406517': 2, # Open Door 2
      '404516': 3, # Close Door 1
      '404517': 4, # Close Door 2
      '406520': 5, # Open Fridge
      '404520': 6, # Close Fridge
      '406505': 7, # Open Dishwasher
      '404505': 8, # Close Dishwasher
      '406519': 9, # Open Drawer 1
      '404519': 10, # Close Drawer 1
      '406511': 11, # Open Drawer 2
      '404511': 12, # Close Drawer 2
      '406508': 13, # Open Drawer 3
      '404508': 14, # Close Drawer 3
      '408512': 15, # Clean Table
      '407521': 16, # Drink from Cup
      '405506': 17, # Toggle Switch
  }
  
  def __init__(self, base_src, normalization='rescale'):
    self.normalization = normalization # None, 'mean', 'rescale'
    
    self.min_list = []
    self.max_list = []
    self.mean_list = []
    
    self.base_src = base_src
    
    self.data_keys = [
        'acc_rkn^', 'acc_hip', 'acc_lua^', 'acc_rua_', 'acc_lh', 
        'acc_back', 'acc_rkn_', 'acc_rwr', 'acc_rua^', 'acc_lua_', 
        'acc_lwr', 'acc_rh', 'ine_back', 'ine_rua', 'ine_rla', 
        'ine_lua', 'ine_lla', 'ine_lshoe', 'ine_rshoe'
    ]
    self.data_size =  [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 9, 9, 9, 9, 9, 16, 16]
    
    self.unload_all_data()
    
  
  def static_interpolate(data):
    y = data
    nans, x = np.isnan(y), lambda z: z.nonzero()[0]
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y 
  

  def calculate_min_max_mean(self):
    original_normalization = self.normalization
    self.normalization = None
    
    self.min_list = []
    self.max_list = []
    self.mean_list = []
    
    for model_index in range(len(self.data_keys)):      
      data_name = self.data_keys[model_index]
      data_size = self.data_size[model_index]
      
      data_per_channel_list = []
      for _ in range(data_size):
        data_per_channel_list.append([])
      
      for filename in self.filename_list:
        data_list = self.get_all_data(filename)
        for each_data in data_list:
          current_data = each_data[data_name]
          if (np.sum(np.isnan(current_data)) > 0):
            continue
          for (i, data) in enumerate(current_data):
            data_per_channel_list[i].append(data)
      
      data_per_channel_list = np.array(data_per_channel_list)
      
      model_min = np.amin(data_per_channel_list, axis=1)
      model_max = np.amax(data_per_channel_list, axis=1)
      model_mean = np.mean(data_per_channel_list, axis=1)
      
      self.min_list.append(model_min)
      self.max_list.append(model_max)
      self.mean_list.append(model_mean)
    
    self.normalization = original_normalization
  
  
  def load_data(self, filename):
    pass
  
  
  def unload_all_data(self):
    self.filename_list = []
    self.data_list = {}
  
  
  def interpolate_data(self):
    for filename in self.filename_list:      
      for key in self.data_keys:        
        data = []
        for each_data in self.data_list[filename]:
          data.append(each_data[key])
        data = np.array(data)
        data = np.swapaxes(data, 0, 1)
        
        for i in range(len(data)):
          data[i] = DataLoader.static_interpolate(data[i])
        
        data = np.swapaxes(data, 0, 1)
        
        for (i, each_data) in enumerate(data):
          self.data_list[filename][i][key] = each_data
  
  
  def fill_missing_data(self):
    for filename in self.filename_list:
      for key in self.data_keys:
        last_data = []
        for (i, each_data) in enumerate(self.data_list[filename]):
          if ((len(last_data) > 0) and (np.sum(np.isnan(each_data[key])) > 0)):
            self.data_list[filename][i][key] = copy.deepcopy(last_data)
          elif (np.sum(np.isnan(each_data[key])) <= 0):
            last_data = each_data[key]
  
  
  def introduce_block_missing_data(self, missing_rate):
    total_count, missing_count = self.get_total_and_missing_count()
    target_missing_count = int(np.ceil(missing_rate * total_count))
    
    while (missing_count < target_missing_count):
      filename_index = np.random.randint(low=0, high=len(self.filename_list))
      filename = self.filename_list[filename_index]
      
      data_start_index = np.random.randint(low=1, high=len(self.data_list[filename]))
      data_range = np.random.randint(low=300, high=900)
      data_range = min(data_range, target_missing_count - missing_count)
      data_range = min(data_range, len(self.data_list[filename])-data_start_index-1)
      data_end_index = data_start_index+data_range
      
      data_key_index = np.random.randint(low=0, high=len(self.data_keys))
      data_key = self.data_keys[data_key_index]
      
      for data_index in range(data_start_index, data_end_index):
        if (np.sum(np.isnan(self.data_list[filename][data_index][data_key])) <= 0):
          missing_count += 1
        self.data_list[filename][data_index][data_key] = ([np.nan] * len(self.data_list[filename][data_index][data_key]))
  
  
  def get_total_and_missing_count(self):
    total_count = 0
    missing_count = 0
    
    for filename in self.filename_list:      
      for each_data in self.data_list[filename]:
        for key in self.data_keys:
          total_count += 1
          if (np.sum(np.isnan(each_data[key])) > 0):
            missing_count += 1
    
    return total_count, missing_count
  
  
  def get_ml_proportions(self):
    ml_count_list = []
    
    for filename in self.filename_list:
      for each_data in self.data_list[filename]:
        ml = each_data['ml']
        while len(ml_count_list) <= ml:
          ml_count_list.append(0)
        ml_count_list[ml] += 1
    
    ml_count_list = np.array(ml_count_list, dtype=np.float32)
    ml_count_list = ml_count_list / np.sum(ml_count_list)
    
    return ml_count_list
  
  
  def normalize_data(self, data):
    data = copy.deepcopy(data)
    
    if (self.normalization == None):
      return data
    
    for i in range(len(data)):
      for (key_index, key) in enumerate(self.data_keys):
        if (self.normalization == 'mean'):
          data[i][key] = (data[i][key] - self.mean_list[key_index]) / (self.max_list[key_index] - self.min_list[key_index])
        elif (self.normalization == 'rescale'):
          data[i][key] = (data[i][key] - self.min_list[key_index]) / (self.max_list[key_index] - self.min_list[key_index])
    
    return data
  
  
  def get_all_data(self, filename):
    return self.normalize_data(self.data_list[filename])
  
  
  def get_random_range_data(self, length):
    filename_index = np.random.randint(low=0, high=len(self.filename_list))
    filename = self.filename_list[filename_index]
    
    range_start = np.random.randint(low=0, high=len(self.data_list[filename])-length)
    range_end = range_start + length
    
    return self.normalize_data(self.data_list[filename][range_start:range_end])
  
  
  def get_range_data(self, filename, start, length):
    data = self.data_list[filename]
    
    if (len(data) < start+length):
      return self.normalize_data(data[start:-1])
    return self.normalize_data(data[start:start+length])
  
    
    