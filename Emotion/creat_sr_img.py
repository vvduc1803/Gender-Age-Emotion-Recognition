# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import necessary package"""
import os
from srgan import super_reso

# Convert image 48x48 -> 384x384
root_in = 'raw/train/'
root_out = 'data/train/'
paths_in = os.listdir(f'{root_in}')
paths_out = os.listdir(f'{root_out}')
for path in paths_in:
    name_images = os.listdir(f'{root_in}{path}')
    for name_image in name_images:
        super_reso(f'{root_in}{path}/{name_image}', f'{root_out}{path}/{name_image}')


