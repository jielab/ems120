#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 画地图
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# pip install keplergl pandas jupyter
# jupyter notebook 
# 民航地图示例：https://github.com/wybert/minhang
import pandas as pd
dat = pd.read_csv("/mnt/d/files/120.txt", sep = "\t"); # dat.head()
import keplergl
map = keplergl.KeplerGl(height = 500); map # 需要把这个作为最后一行命令
map.add_data(data = dat.copy(), name = "house")
# %run 120.config.py
# with open('120.config.py', 'w') as f: f.write('config = {}'.format(map.config))
