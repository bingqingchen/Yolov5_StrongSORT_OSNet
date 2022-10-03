import torch
from typing import OrderedDict

ckpt= torch.load('weights/osnet_x0_25_distilled_e99.ckpt', map_location=torch.device('cpu'))
model_dict = ckpt['state_dict']
student_dict = {}

import pdb; pdb.set_trace()
for k, v in model_dict.items():
    if k.split('.')[0] == 'student' and k.split('.')[2]!='classifier':
        print(len(k.split('.')[1:]))
        new_key = '.'.join( k.split('.')[2:])
        student_dict[new_key] = v
    elif k.split('.')[0] == 'student' and k.split('.')[2]=='classifier':
        if k.split('.')[2] == 'classifier':
            new_key = '.'.join(['0',k.split('.')[3]])
            student_dict[new_key] = v

torch.save(OrderedDict(student_dict), 'weights/osnet_x0_25_distilled_e99.pt')
