import numpy as np
import matplotlib.pyplot as plt


cold_blue = np.array([30,70,190])/255
warm_red = np.array([214,27,35])/255

#real data taken from: #https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series
# year = gl_temp[::2,0] 
# temp = gl_temp[::2,1]


#example data generated from the following formula:
year = np.arange(1880,2026,2)
temp = (-np.random.rand(len(year))+ np.exp((year-1980)/50))/1.75

fig, ax = plt.subplots(figsize = (8,6), dpi = 300)

im = ax.bar(year[temp < 0],
            temp[temp < 0],
            color = cold_blue,
            edgecolor = 'black',
            zorder = 3,
            width = 1.3,
          )

im = ax.bar(year[temp > 0],
            temp[temp > 0],
            color = warm_red,
            edgecolor = 'black',
            zorder = 3,
            width = 1.3,            
          )

ax.set_xlim([1880,2024])
ax.set_ylim([-0.6, 1.4])

ax.set_yticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4])

ax.set_xlabel('Year', fontsize = 22)
ax.set_ylabel(f'Temperature Deviation [\N{DEGREE SIGN}C]', fontsize = 20)

ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)

ax.spines['bottom'].set_zorder(10)



ax.grid(axis = 'both',
          which = 'major', 
          linestyle = '--', 
          color = [0.3,0.3,0.3],
          alpha = 0.4,
          zorder = 1)

ax.tick_params(axis = 'y',
               which = 'major', 
               labelsize = 18,
               width = 1.75, 
               direction = 'in',
               length = 10)

ax.tick_params(axis = 'x',
               which = 'major', 
               labelsize = 18,
               width = 1.75, 
               direction = 'in',
               length = 0)


plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right",
          rotation_mode="anchor")
