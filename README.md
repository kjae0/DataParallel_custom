# DataParallel_custom

pytorch DataParallel with chunk size

You can set chunk size of each gpus on your own.

  Example
  
'''
  from dataparallel_v2 import CustomDataParallel
  
  # your model
  model = Model()

  # set chunk size manually
  chunk_size = [10, 15, 15, 15]
  # chunk_size = None -> same as default DataParallel
  
  model = CustomDataParallel(model, chunk_size=chunk_size)
'''
