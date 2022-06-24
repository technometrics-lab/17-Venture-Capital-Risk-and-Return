import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import norm

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15,8))
axs[0].tick_params(left = False, right = False, labelleft = False,
                labelbottom = False, bottom = False)
axs[1].tick_params(left = False, right = False, labelleft = False,
                labelbottom = False, bottom = False)
axs[2].tick_params(left = False, right = False, labelleft = False,
                labelbottom = False, bottom = False)

def pipo(x):
    a,b = 3, 0.1
    return 0.4 / (1 + np.exp(-a * (x - b)))

def pbkp(x):
    k, x0 = -1, -2
    return  (x <= k) * (1 - (x - x0) / (k - x0)) * (0.2)
                
n = 3
mu, sigma = 0, 1
x = np.linspace(-3, 3, 240)
dist = norm.pdf(x, loc=mu, scale=sigma)
ipo = pipo(x)
bkp = pbkp(x)

bkp_hist = dist * bkp * 3
ipo_hist = dist * ipo * 1.5

axs[0].plot(x, dist, 'black')
axs[0].plot(x, ipo, 'seagreen')
axs[0].plot(x, bkp, 'tomato')
axs[0].text(-1, 0.33, '$P(V_t)$', style='italic', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 2})
axs[0].text(0.29, 0.2, '$P(Exit, V_t)$', style='italic', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 2})
axs[0].text(-2.3, 0.07, '$P(Close, V_t)$', style='italic', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 2})
markerline, stemlines, baseline = axs[0].stem(x[::n], ipo_hist[::n], 'seagreen', markerfmt='C1 ', basefmt=' ', use_line_collection=True)
stemlines.set_linewidth(3)
markerline, stemlines, baseline = axs[0].stem(x[::n], bkp_hist[::n], 'tomato', markerfmt=' ', basefmt=' ', use_line_collection=True)
stemlines.set_linewidth(3)


dist1 = dist - bkp_hist - ipo_hist
markerline, stemlines, baseline = axs[1].stem(x[::n], dist1[::n], 'black', markerfmt='C1 ', basefmt=' ', use_line_collection=True)
stemlines.set_linewidth(3)
axs[1].text(0.5, 0.23, '$P(Private, V_t)$', style='italic', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 2})

bkp_hist = dist1 * bkp * 3
ipo_hist = dist1 * ipo * 1.5

axs[2].plot(x, dist1, 'black')
axs[2].plot(x, ipo, 'seagreen')
axs[2].plot(x, bkp, 'tomato')

axs[2].text(-1, 0.33, '$P(V_{t+1})$', style='italic', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 2})
axs[2].text(2, 0.35, '$P(Exit, V_{t+1})$', style='italic', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 2})
axs[2].text(-3, 0.23, '$P(Close, V_{t+1})$', style='italic', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 2})
markerline, stemlines, baseline = axs[2].stem(x[::n], ipo_hist[::n], 'seagreen', markerfmt='C1 ', basefmt=' ', use_line_collection=True)
stemlines.set_linewidth(3)
markerline, stemlines, baseline = axs[2].stem(x[::n], bkp_hist[::n], 'tomato', markerfmt=' ', basefmt=' ', use_line_collection=True)
stemlines.set_linewidth(3)

plt.show()