import pandas as pd
import matplotlib.pyplot as plt

pd_result = pd.read_csv('../data/output/KGC_factor2.csv')
run_time = pd.DataFrame({'average cover size': pd_result['average cover size']})
run_time.index = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
ax = run_time.plot.bar()
ax.set_xlabel('f')
ax.set_ylabel('average cover size')
ax.set_title('KGC')
plt.show()
fig = ax.get_figure()
fig.savefig('../data/output/KGC/average_cover_size.png')
