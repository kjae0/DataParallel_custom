# pytorch DataParallel with chunk size

You can set chunk size of each gpus on your own.

Pytorch DataParallel is very simple and easy to use.

But, sometimes multi GPU training with 'DataParallel' makes severe GPU memory imbalance(especially, NLP task) because DataParallel acutally operates on only one main GPU.
So, you can train with larger batch if you allocate larger data chunk to other GPUs.

## Example
  
```
  from dataparallel_v2 import CustomDataParallel

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
