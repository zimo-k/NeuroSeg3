import os

import pandas as pd
import matplotlib.pyplot as plt

root_dir = ''
log_dir = os.path.join(root_dir, '/runs/ssl/ABO/TiCo/TiCo_128_20240206T2259')  # TiCo20231115T2054
data = pd.read_csv(os.path.join(log_dir, 'log.csv'))

epochs = data['epoch']
loss = data['loss']

plt.plot(epochs, loss, label='')

plt.legend()
plt.savefig(os.path.join(root_dir, 'loss.png'))
plt.show()
