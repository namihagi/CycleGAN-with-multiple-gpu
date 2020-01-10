# CycleGAN-with-multiple-gpu

# train
~~~
python3 main.py --phase train --gpu_num 2 --global_batch_size 8
~~~
- --gpu_num: the number of gpu you use  
- --global_batch_size: overall batch size (which means each batch size per gpu is global_batch_size / gpu_num)
- If you change other parameters, check config.json.

# test
~~~
python3 main.py --phase test --gpu_num 2 --global_batch_size 8
~~~
- Options are the same as training.

Reference
the base implementation: https://github.com/xhujoy/CycleGAN-tensorflow.git
