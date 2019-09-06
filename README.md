# CycleGAN-with-multiple-gpu

for training
~~~
python3 main.py --phase train --gpu_num 2 --global_batch_size 8
~~~
- --gpu_num: the number of gpu you use  
- --global_batch_size: overall batch size (which means each batch size per gpu is global_batch_size / gpu_num)