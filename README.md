# DataParallel_custom

pytorch DataParallel with chunk size

You can set chunk size of each gpus on your own.

Pytorch DataParallel is very simple and easy to use.

But, sometimes multi GPU training with 'DataParallel' makes severe GPU memory imbalance(especially, NLP task) because DataParallel acutally operates on only one main GPU.
You can train with larger batch if you allocate larger data chunk to other GPUs.



## Example
  
```
  from dataparallel_v2 import CustomDataParallel
  
  # your model
  model = Model()

  # set chunk size manually
  chunk_size = [10, 15, 15, 15]
  # chunk_size = None -> same as default DataParallel
  
  model = CustomDataParallel(model, chunk_size=chunk_size)
```
