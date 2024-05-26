import os
from os.path import join as opj
from utils import PN2_BNMomentum, PN2_Scheduler

exp_name = "teacger_ESTIMATION_Release"
work_dir = opj("./log/openad_pn2", exp_name)
seed = 1
try:
    os.makedirs(work_dir)
except:
    print('Working Dir is already existed!')

scheduler = dict(
    type='lr_lambda',
    lr_lambda=PN2_Scheduler(init_lr=0.001, step=20,
                            decay_rate=0.5, min_lr=1e-5)
)

optimizer = dict(
    type='adam',
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-4
)

model = dict(
    type='teacher_point2',
    weights_init='pn2_init'
)

training_cfg = dict(
    model=model,
    estimate=True,
    partial=False,
    rotate='None',  # z,so3
    semi=False,
    rotate_type=None,
    batch_size=128,
    epoch=200,
    seed=1,
    dropout=0.5,
    gpu='4',
    workflow=dict(
        train=1,
        val=1
    ),
    bn_momentum=PN2_BNMomentum(origin_m=0.1, m_decay=0.5, step=20),
    train_affordance = ['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
               'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
               'listen', 'wear', 'press', 'cut', 'stab', 'none'],
    val_affordance = ['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
               'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
               'listen', 'wear', 'press', 'cut', 'stab', 'none'],
)

data = dict(
    data_root = '/home/tuan.vo1/IROS2023_Affordance-master/Data',
    category = ['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
               'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
               'listen', 'wear', 'press', 'cut', 'stab', 'none']
)
