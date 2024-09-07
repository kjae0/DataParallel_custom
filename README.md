# pytorch DataParallel with chunk size

DataParallel with varying chunk sizes.

## Example
  
```
  from custom_dataparallel import CustomDataParallel

  import torch.nn as nn

  # your model
  model = Model()

  # pytorch DataParallel
  model = nn.DataParalel(model) # memory imbalance. (GPU memory usage : GPU 0 > GPU 1, 2, 3...)

  # set chunk size manually
  chunk_size = [10, 15, 15, 15] # larger chunk size to GPU 1, 2, 3 
  # chunk_size = None -> same as default DataParallel
  
  model = CustomDataParallel(model, chunk_size=chunk_size)
```
