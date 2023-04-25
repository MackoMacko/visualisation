import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

x = np.linspace(300,900,61)


EQE_before =  np.exp(-((x-450)**2)/2500) + np.exp(-((x-700)**2)/2500) + 2*np.exp(-((x-575)**2)/20000)
EQE_after = np.exp(-((x-480)**2)/2500) + np.exp(-((x-700)**2)/2500) + 2.2*np.exp(-((x-575)**2)/20000)


EQE_before = 0.8*EQE_before/max(EQE_before)
EQE_after = 0.96*EQE_after/max(EQE_after)

EQE_before_interpolated = make_interp_spline(x,EQE_before, k=3)
EQE_after_interpolated = make_interp_spline(x,EQE_after, k=3)

x_inter = np.linspace(300,900,610)

#colors
red_c = (0.992, 0.102, 0.03)
blue_c = (0.012,0.224,0.99)


#figure setting
fontsize = 22


#figure = plt.figure(figsize = (9,6), dpi = 500)

fig, ax = plt.subplots(figsize = (9,6), dpi = 500)

ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# ax.yaxis.get_majorticklabels()[0].set_y(1)

ax.plot(x,EQE_before, c = red_c, 
         linestyle = '-',
         linewidth = 2,
         marker = '.',
         markersize = 12)

ax.plot(x,EQE_after,  c = blue_c,
         linestyle = '-',
         linewidth = 2,
         marker = '.',
         markersize = 12,
         zorder = 10)

ax.set_yticks([0.2,0.4,0.6,0.8,1.0])


plt.xlim([300,800])
plt.ylim([0,1])

plt.tick_params(direction = 'in', length = 12, width = 1.5)
plt.grid(which = 'major', axis = 'both', linestyle = '--', color = (0.6,0.6,0.6), alpha = 0.8)    
plt.xticks(fontsize = fontsize)
plt.yticks(fontsize = fontsize)
   
plt.xlabel('Wavelength [nm]', fontsize = fontsize)
plt.ylabel('Normalized EQE', fontsize = fontsize)
#"'MgF$_{2}$'
plt.legend(['Without ARC', 'Honeycomb'], loc = (0.615,0.802), fontsize = fontsize - 2.5 )


enhancement = EQE_after_interpolated(x_inter) -EQE_before_interpolated(x_inter)

enhancment_plus = np.abs(np.ma.masked_where(enhancement < 0, enhancement))
enhancment_minus = np.abs(np.ma.masked_where(enhancement > 0, enhancement))


plt.fill_between(x_inter,enhancment_plus, 
                 alpha = 0.6, 
                 color = blue_c
                 )

plt.plot(x_inter,enhancment_minus,  c = 'r',
         linestyle = '-',
         linewidth = 2,
         zorder = 1
         )

plt.fill_between(x_inter,enhancment_minus, 
                 alpha = 0.6, 
                 color = red_c
                 )

plt.plot(x_inter,enhancment_plus,  c = blue_c,
         linestyle = '-',
         linewidth = 2,
         )

#xy = 510,0.125 / 35,30
plt.annotate(
    r"EQE gain",
    xy=(520, 0.185),
    xytext=(35, 30),
    textcoords="offset points",
    fontsize=24,
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.3", lw = 2),
)

#xy =350,0.075 / 35,45

#xy=(350, 0.14),
#xytext=(35, 35),

plt.annotate(
    r"EQE loss",
    xy=(420, 0.18),
    xytext=(35,45),
    textcoords="offset points",
    fontsize=24,
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.3", lw = 2),
)



fig.text(0.0, 0.9, '(b)', fontsize=26)