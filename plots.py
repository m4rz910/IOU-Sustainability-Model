import matplotlib.pyplot as plt
from model import Fuzzification, InferenceEngine
import pandas as pd
import numpy as np


plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titleweight'] = 'bold'

fig = plt.figure(figsize=(10,5))
Fuzzification.wms().plot();
plt.grid();plt.xlabel('z');plt.ylabel('y');
fig.savefig('./outputs/wms.png', dpi=300, bbox_inches='tight');

fig = plt.figure(figsize=(10,5))
Fuzzification.vbbagvg().plot()
plt.grid();plt.xlabel('z');plt.ylabel('y')
fig.savefig('./outputs/vbbagvg.png', dpi=300, bbox_inches='tight')

fig = plt.figure(figsize=(10,5))
Fuzzification.elvllflifhhvheh().plot()
plt.grid();plt.xlabel('z');plt.ylabel('y')
fig.savefig('./outputs/elvllflifhhvheh.png', dpi=300, bbox_inches='tight')