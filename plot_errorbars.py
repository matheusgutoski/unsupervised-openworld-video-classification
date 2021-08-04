import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np



methods = np.array(['BIC','EEIL','EWC','IL2M','LUCIR','LWF','MAS','Path Int','R WALK','ICARL','Ours'])

forgetting = np.array([0.019,0.009,0.025,0.020,0.002,0.027,0.013,0.018,0.019,0.006,0.033])
forgetting_std = np.array([0.012,0.004,0.002,0.003,0.001,0.004,0.002,0.005,0.003,0.001,0.004])

nmi = np.array([0.863,0.857,0.812,0.817,0.816,0.889,0.818,0.818,0.815,0.861,0.912])
nmi_std = np.array([0.048,0.060,0.071,0.073,0.081,0.027,0.088,0.072,0.077,0.068,0.011])


x_pos = np.arange(len(methods))
CTEs = [forgetting_std]



# Build the plot
fig = plt.figure()
fig.bar(forgetting, forgetting_std, yerr=forgetting_std, align='center', alpha=0.5, ecolor='black', capsize=10)

# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
