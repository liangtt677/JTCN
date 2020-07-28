# Joint Training Capsule Network for Cold Start Recommendation

Tensorflow implementation of the paper: "Joint Training Capsule Network for Cold Start Recommendation", SIGIR, short paper, 2020.

# Requirements
- Python 3.7
- Tensorflow 1.14

# Usage

To reproduce the experiments mentioned in the paper, you can run the following commands:

<pre>python  main.py  --regs [1e-3] --embed_size 256  --lr 0.0005 --save_flag 1 --batch_size 128 --epoch 100 --verbose 1 --dataset 'CiteU'</pre>


# Citation

Please cite our paper if you use this code in your own work:

<pre>@article{liang2020joint,
  title={Joint Training Capsule Network for Cold Start Recommendation},
  author={Liang, Tingting and Xia, Congying and Yin, Yuyu and Yu, Philip S},
  journal={arXiv preprint arXiv:2005.11467},
  year={2020}
} </pre>

For any questions and comments, please send your email to liangtt@hdu.edu.cn.
