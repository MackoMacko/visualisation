import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import simps
import pandas as pd

from scipy.optimize import curve_fit

#https://ourworldindata.org/grapher/solar-pv-prices-vs-cumulative-capacity
costs = 'G:\\Mój dysk\\STUDIA\\PROGRAMY I SKRYPTY\\Python\\PV_data_PhD\\solar-pv-prices-vs-cumulative-capacity_2.txt'


#https://www.iea.org/reports/solar-pv
#https://ourworldindata.org/grapher/installed-global-renewable-energy-capacity-by-technology
#https://www.iea.org/data-and-statistics/charts/global-installed-solar-pv-capacity-by-scenario-2010-2030
ren_share = 'G:\\Mój dysk\\STUDIA\\PROGRAMY I SKRYPTY\\Python\\PV_data_PhD\\Renewables_IRENA.txt'


#https://ourworldindata.org/grapher/solar-pv-cumulative-capacity
#https://ourworldindata.org/grapher/electricity-demand?country=USA~GBR~FRA~DEU~IND~BRA
cum_gwp = 'G:\\Mój dysk\\STUDIA\\PROGRAMY I SKRYPTY\\Python\\PV_data_PhD\\Cumulative_Power.txt'

#https://yearbook.enerdata.net/electricity/electricity-domestic-consumption-data.html
#https://www.statista.com/statistics/280704/world-power-consumption/
#https://www.iea.org/reports/electricity-information-overview/electricity-consumption
el_consumption = 'G:\\Mój dysk\\STUDIA\\PROGRAMY I SKRYPTY\\Python\\PV_data_PhD\\Energy_consumption.txt'


tot_ren = 'G:\\Mój dysk\\STUDIA\\PROGRAMY I SKRYPTY\\Python\\PV_data_PhD\\installed-global-renewable-energy-capacity-by-technology.csv'

tot_ren = np.genfromtxt(tot_ren, delimiter = ',')
costs = np.genfromtxt(costs, delimiter = ',')

tot_ren = np.vstack([costs[24:,0],costs[24:,2],tot_ren[:,1]]).T

eff_table = 'G:\\Mój dysk\\STUDIA\\PROGRAMY I SKRYPTY\\Python\\PV_data_PhD\\PV_efficiency.csv'

eff_table = pd.read_csv(eff_table, delimiter = ';')

#https://www.sciencedirect.com/science/article/abs/pii/S0038092X16001110
SQ_lim =  'G:\\Mój dysk\\STUDIA\\PROGRAMY I SKRYPTY\\Python\\PV_data_PhD\\SQ_limit.txt'
SQ_lim = np.genfromtxt(SQ_lim, delimiter = ' ')

#keeling_curve
#https://keelingcurve.ucsd.edu/permissions-and-data-sources/
keel = 'G:\\Mój dysk\\STUDIA\\PROGRAMY I SKRYPTY\\Python\\PV_data_PhD\\keeling_curve.csv'

keel_c = keel.iloc[:,[0,9]]


#global temp anomaly
#https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series
gl_temp = 'G:\\Mój dysk\\STUDIA\\PROGRAMY I SKRYPTY\\Python\\PV_data_PhD\\global_temperature anomally.txt'
gl_temp = np.genfromtxt(gl_temp, delimiter= ',')





ren_share = pd.read_csv(ren_share)

cum_gwp = np.genfromtxt(cum_gwp)

el_consumption = np.genfromtxt(el_consumption)


def fitted_func(x,f0,c,x0):
    return f0*c**(x-x0)

#%% GWP vs year






ise_green = np.array([23, 156, 125])/255
ise_blue = np.array([31,130,192])/255

labels= ['2000', '2002','2004','2006','2008','2010','2012','2014','2016','2018','2020','2021', '2022']

pv_cum = tot_ren[:,1]
ren_wpv = (tot_ren[:,2] -  pv_cum)/1000
pv_cum = pv_cum/1000

pv_cum = np.append(pv_cum[::2],pv_cum[-1])

ren_wpv = np.append(ren_wpv[::2],ren_wpv[-1])



perc = np.round(100*(tot_ren[:,1]/tot_ren[:,2]),0)
perc = np.append(perc[::2],perc[-1])

#%% GWP vs year tot renewables

fig, ax = plt.subplots(figsize = (8,6), dpi = 500)

im = ax.bar(labels,
            ren_wpv,
            color = ise_blue,
            zorder = 3,
            width = 0.8
)


im = ax.bar(labels,
            pv_cum,
            color = ise_green,
            bottom= ren_wpv,
            zorder = 3,
            width = 0.8
)

tot_ren_final = pv_cum + ren_wpv



for ind,el in enumerate(labels):
        ax.annotate(f'{int(perc[ind])}%', xy = [el, tot_ren_final[ind]],
                        xytext=(-14,5),
                        textcoords="offset points",
                        fontsize = 14,
                        color = ise_green,
                        weight = 'bold'
                        )



# Show all ticks and label them with the respective list entries


# ax.set_yticks([100,200,300,400,500,600,700,800])

# ax.set_xticklabels(X_labels[::4].astype(int))
# ax.set_yticklabels(Y_labels[::4].astype(int))

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right",
         rotation_mode="anchor")

ax.set_xlabel('Year', fontsize = 22)
ax.set_ylabel('Gigawatts', fontsize = 22)

ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)

ax.spines['bottom'].set_zorder(10)
ax.set_title('Global Renewable Energy Power Capacity',
             fontsize = 22)

ax.set_ylim([0,3750])

ax.grid(axis = 'y',
          which = 'major', 
          linestyle = '--', 
          color = [0.3,0.3,0.3],
          alpha = 0.6,
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

ax.set_axisbelow(True)

ax.annotate(f'{int(ren_wpv[-1])}', xy = ['2022',ren_wpv[-1]/2],
                xytext=(-8,0),
                textcoords="offset points",
                fontsize = 14,
                color = 'white',
                rotation = 270,
                weight = 'bold'
                )

ax.annotate(f'{1046}', xy = ['2022',tot_ren_final[-1] - pv_cum[-1]*0.7],
                xytext=(-8,0),
                textcoords="offset points",
                fontsize = 14,
                color = 'white',
                rotation = 270,
                weight = 'bold'
                )



ax.legend(['Wind, Hydro, Bio', 'Photovoltaics'],
            title = 'Technology',
            fontsize = 18,
            title_fontsize = 22)






#%% Cost vs year



fig, ax = plt.subplots(figsize = (8,6), dpi = 300)

colors = [np.array([23, 156, 125])/255,
        np.array([242,148,0, 230])/255]


ax.scatter(costs[:,0], costs[:,1],
           color = colors[0],
           s = 100)


# Show all ticks and label them with the respective list entries
ax.set_xticks(costs[::2,0])
# ax.set_xticklabels(X_labels[::4].astype(int))
# ax.set_yticklabels(Y_labels[::4].astype(int))
ax.set_title('PV Electricity Generation Costs',
             fontsize = 22)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right",
         rotation_mode="anchor")

ax.set_xlabel('Year', fontsize = 22)
ax.set_ylabel('EUR / kWh', fontsize = 22)

ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)

ax.spines['bottom'].set_zorder(10)



ax.grid(axis = 'y',
          which = 'major', 
          linestyle = '--', 
          color = [0.3,0.3,0.3],
          alpha = 0.6,
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

ax.set_axisbelow(True)





#%%Swanson law

from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch

fig, ax = plt.subplots(figsize = (8,6), dpi = 300)

cost_plot = costs[4:]
colors = [np.array([23, 156, 125])/255,
        np.array([242,148,0, 230])/255]

ax.plot(cost_plot[:,2], cost_plot[:,1],
           color = colors[0],
           linewidth = 4,
           )






# popt, _ = curve_fit(fitted_func, swanson_law[0],swanson_law[1])
# # summarize the parameter values
# a, b = popt
# # plot input vs output
# # calculate the output for the range
# y_line = objective(x_line, a, b)








# ax.plot(swanson_law[0],swanson_law[1], '--', color = [0.9, 0.04,0.1])
# #ax.plot(swanson_law[0],y_line, '--', color = [0.9, 1,0.1])

ax.set_xscale('log')

ax.set_title('PV Energy Costs vs. Worldwide Capacity',
             fontsize = 22)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(),
         rotation_mode="anchor")

#

ax.set_xlabel('Cumulative PV Capacity [MW]', fontsize = 22)
ax.set_ylabel('Electricity Production Costs \n [USD / Watt ]', fontsize = 22)

ax.set_xlim([2,2500000])
ax.set_ylim([0,40])

ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)

ax.spines['bottom'].set_zorder(10)


for i,el in enumerate(cost_plot[:,0]):
    print(el,i)
    if i % 5 == 0:
        ax.annotate(int(el), xy = [cost_plot[i,2], cost_plot[i,1]],
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize = 14,
                   #arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color = [0.5,0.5,0.3], alpha = 0.8),
                    )
        ax.scatter(cost_plot[i,2], cost_plot[i,1],
           color = np.array([242,90,0])/255,
           s = 50,
           zorder = 100
           )


ax.grid(axis = 'y',
          which = 'major', 
          linestyle = '--', 
          color = [0.3,0.3,0.3],
          alpha = 0.6,
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
               length = 10)

ax.tick_params(axis = 'x',
               which = 'minor', 
               labelsize = 18,
               width = 1.25, 
               direction = 'in',
               length = 0)

ax.set_axisbelow(True)



rect = Rectangle(
    (30000, 0.25),
    2300000,
    4,
    edgecolor=np.array([242,10,0])/255,
    facecolor="None",
    linestyle="--",
    linewidth=1.75,
)
ax.add_patch(rect)

rect2 = Rectangle(
    (1000, 15.5),
    2250000,
    24,
    edgecolor=np.array([242,10,0])/255,
    facecolor="None",
    linestyle="--",
    linewidth=1.75,
)
ax.add_patch(rect2)



ax.annotate('', xy = [4*10**4, 15.6],
                    xytext=(0,-95),
                    textcoords="offset points",
                    fontsize = 14,
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color = np.array([242,10,0])/255, linewidth = 1.75, alpha = 1),
                    )

# con = ConnectionPatch(
#         xyA=(x, y),
#         coordsA=ax.transData,
#         xyB=(0, 0.5),
#         coordsB=sax.transAxes,
#         linestyle="--",
#         linewidth=0.75,
#         patchA=rect,
#         arrowstyle="->",
#     )
#     fig.add_artist(con)





#%% Swanson law zoom
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch




fig, ax = plt.subplots(figsize = (8,6), dpi = 300)

cost_plot = costs[34:]

colors = [np.array([23, 156, 125])/255,
        np.array([242,148,0, 230])/255]

ax.plot(costs[32:,2], costs[32:,1],
           color = colors[0],
           linewidth = 4.5,
           )


ax.scatter(cost_plot[:,2], cost_plot[:,1],
   color = 'red',
   s = 50,
   zorder = 100
   )

ax.set_xscale('log')

ax.set_title(' ',
             fontsize = 22)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(),
         rotation_mode="anchor")

#

ax.set_xlabel('Cumulative PV Capacity [MW]', fontsize = 22)
ax.set_ylabel('USD / Watt', fontsize = 22)

ax.set_xlim([20000,1100000])
ax.set_ylim([0,2.5])

ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)

ax.spines['bottom'].set_zorder(10)


for i,el in enumerate(cost_plot[:,0]):
    print(el,i)
    if i % 2 == 0:
        ax.annotate(int(el), xy = [cost_plot[i,2], cost_plot[i,1]],
                    xytext=(6, 3),
                    textcoords="offset points",
                    fontsize = 14,
                   #arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color = [0.5,0.5,0.3], alpha = 0.8),
                    )
        ax.scatter(cost_plot[i,2], cost_plot[i,1],
           color = np.array([242,90,0])/255,
           s = 75,
           zorder = 100
           )


ax.grid(axis = 'y',
          which = 'major', 
          linestyle = '--', 
          color = [0.3,0.3,0.3],
          alpha = 0.6,
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
               length = 10)

ax.tick_params(axis = 'x',
               which = 'minor', 
               labelsize = 18,
               width = 1.25, 
               direction = 'in',
               length = 8)

ax.set_axisbelow(True)




# con = ConnectionPatch(
#     xyA=(x, y),
#     coordsA=ax.transData,
#     xyB=(0, 0.5),
#     coordsB=sax.transAxes,
#     linestyle="--",
#     linewidth=0.75,
#     patchA=rect,
#     arrowstyle="->",
# )
# fig.add_artist(con)





#%%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch


fig = plt.figure(figsize=(5, 4))

n = 2
gs = GridSpec(n, n + 1)

ax = plt.subplot(
    gs[:n, :n], xlim=[-1, +1], xticks=[], ylim=[-1, +1], yticks=[], aspect=1
)


#%%




#%% Electricity consumption


fig, ax = plt.subplots(figsize = (8,6), dpi = 300)

colors = [np.array([23, 156, 125])/255,
        np.array([242,148,0, 230])/255]


orange_bars = [np.array([242,148,0, 225])/255, np.array([242,148,0, 205])/255,
               np.array([242,148,0, 170])/255, np.array([242,148,0, 140])/255]

el_con = el_consumption[el_consumption[:,0] % 5 == 0]

im = ax.bar(el_con[0:9,0],
            el_con[0:9,1]/1000,
            width = 4,
            color = colors[0],
            zorder = 3
)

im = ax.bar(el_con[9:,0],
            el_con[9:,1]/1000,
            width = 4,
            color = orange_bars,
            zorder = 3
)

# Show all ticks and label them with the respective list entries
ax.set_xticks(el_con[:,0])
ax.set_yticks([0,5,10,15,20,25,30,35,40])

# ax.set_xticklabels(X_labels[::4].astype(int))
# ax.set_yticklabels(Y_labels[::4].astype(int))
ax.set_title('Global Electricity Demand',
             fontsize = 22)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right",
         rotation_mode="anchor")

ax.set_xlabel('Year', fontsize = 22)
ax.set_ylabel('1000 x TWh', fontsize = 22)

ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)

ax.spines['bottom'].set_zorder(10)



ax.grid(axis = 'y',
          which = 'major', 
          linestyle = '--', 
          color = [0.3,0.3,0.3],
          alpha = 0.6,
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

ax.set_axisbelow(True)

ax.axvline(x = 2022.5,
           alpha = 0.7,
           linestyle = ':',
           linewidth = 3,
           color = colors[0])

ax.annotate(f"FORECAST",
    xy=[2022.5,30],
    xytext=(1, 1),
    textcoords="offset points",
    rotation = 270,
    color = [0.6,0.6,0.6],
    fontsize = 14 )


#%% Solar spectrum
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch

nrel = pd.read_csv('C:\\Users\\Razer\\Desktop\\Freiburg_rough\\Desktop\\Data Collected\\References\\ASTMG173.csv', skiprows = 1, delimiter = ',')

nrel = np.array(nrel)

wavelen = nrel[:,0]

power = nrel[:,1]

power2 = nrel[:,2]




fig, ax = plt.subplots(figsize = (8,6), dpi = 300)

colors = [np.array([23, 156, 125])/255,
        np.array([242,148,0, 230])/255]

ax.plot(wavelen,power,
        color = [0.9,0.3,0.2],
        linewidth = 1.75)


ax.plot(wavelen, power2,
           color = colors[0],
           linewidth = 1.75,
           )

ax.fill_between(wavelen[:950], power2[:950],
                alpha = 0.3,
                color = colors[0])


ax.fill_between(wavelen[950:], power2[950:],
                alpha = 0.3,
                color = [0.5,0.3,0.7])


ax.set_xticks([250,500,750,1000,1250,1500,1750,2000,2250,2500])
ax.set_yticks([0,0.5,1,1.5,2,2.5])


ax.set_xlim([250,2500])
ax.set_ylim([0,2.5])


ax.set_title('Solar Spectrum',
             fontsize = 22)




# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right",
         rotation_mode="anchor")

ax.set_xlabel('Wavelength [nm]', fontsize = 22)
ax.set_ylabel('Power Density [Wm$^{-2}$nm$^{-1}$]', fontsize = 22)


ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)

ax.spines['bottom'].set_zorder(10)

ax.legend(['AM0', 'AM1.5G'],
          fontsize = 20,
          )


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

ax.set_axisbelow(True)

# ax.axvline(x = 1107,
#            alpha = 0.8,
#            linestyle = ':',
#            linewidth = 2,
#            color = [0.98,0.2,0.6],
#            zorder = 5)




ax.axvline(x = 1107,
           alpha = 0.8,
           linestyle = '--',
           linewidth = 2.5,
           color = colors[0],
           zorder = 5)

ax.annotate(f"Silicon Bandgap",
    xy=[1107, 1.475],
    xytext=(-4, 1),
    textcoords="offset points",
    rotation = 270,
    color = [0.4,0.4,0.4],
    backgroundcolor = 'white',
    fontsize = 16,
    zorder = 15)


ax.annotate(f"43.2 mA/cm$^2$",
    xy=[427.5, 0.25],
    xytext=(-15, 1),
    textcoords="offset points",
    rotation = 0,
    color = colors[0],
    fontsize = 16,
    weight = 'bold',
    zorder = 15)


ax.axvline(x = 926,
           alpha = 0.8,
           linestyle = '--',
           linewidth = 2.5,
           color = [0.2,0.2,0.9],
           zorder = 5)

ax.annotate(f"SQ Limit Maximum",
    xy=[926, 1.3],
    xytext=(-4, 1),
    textcoords="offset points",
    rotation = 270,
    color = [0.4,0.4,0.4],
    backgroundcolor = 'white',
    fontsize = 16,
    zorder = 15)


#%% SQ_limit

eg = SQ_lim[:,0]

jsc = SQ_lim[:,2]

voc = SQ_lim[:,5]

eff = SQ_lim[:,-1]

fig, ax = plt.subplots(figsize = (8,6), dpi = 300)

colors = [np.array([23, 156, 125])/255,
        np.array([242,148,0, 230])/255]

ax.plot(eg,eff,
        color = [0.9,0.3,0.2],
        linewidth = 1.75)


ax.plot(eg, power2,
           color = colors[0],
           linewidth = 1.75,
           )

ax.fill_between(wavelen[:950], power2[:950],
                alpha = 0.3,
                color = colors[0])


ax.fill_between(wavelen[950:], power2[950:],
                alpha = 0.3,
                color = [0.5,0.3,0.7])


ax.set_xticks([250,500,750,1000,1250,1500,1750,2000,2250,2500])
ax.set_yticks([0,0.5,1,1.5,2,2.5])


ax.set_xlim([250,2500])
ax.set_ylim([0,2.5])


ax.set_title('Global Electricity Demand',
             fontsize = 22)




# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right",
         rotation_mode="anchor")

ax.set_xlabel('Wavelength [nm]', fontsize = 22)
ax.set_ylabel('Efficiency', fontsize = 22)

ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)

ax.spines['bottom'].set_zorder(10)

ax.legend(['AM0', 'AM1.5g'],
          fontsize = 20,
          )


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

ax.set_axisbelow(True)

ax.axvline(x = 1107,
           alpha = 0.8,
           linestyle = ':',
           linewidth = 2,
           color = [0.98,0.2,0.6],
           zorder = 5)


ax.axvline(x = 1107,
           alpha = 0.8,
           linestyle = '--',
           linewidth = 2.5,
           color = colors[0],
           zorder = 5)

ax.annotate(f"Silicon Bandgap",
    xy=[1107, 1.475],
    xytext=(-4, 1),
    textcoords="offset points",
    rotation = 270,
    color = [0.4,0.4,0.4],
    backgroundcolor = 'white',
    fontsize = 16,
    zorder = 15)


ax.annotate(f"43.2 mA/cm$^2$",
    xy=[430, 0.25],
    xytext=(-15, 1),
    textcoords="offset points",
    rotation = 0,
    color = colors[0],
    fontsize = 16,
    weight = 'bold',
    zorder = 15)


ax.axvline(x = 926,
           alpha = 0.8,
           linestyle = '--',
           linewidth = 2.5,
           color = [0.2,0.2,0.9],
           zorder = 5)

ax.annotate(f"SQ Limit Maximum",
    xy=[926, 1.3],
    xytext=(-4, 1),
    textcoords="offset points",
    rotation = 270,
    color = [0.4,0.4,0.4],
    backgroundcolor = 'white',
    fontsize = 16,
    zorder = 15)

#%%keeling curve

#year = keel_c.iloc[:,0]
year = np.arange(1958,2024,1/12)[:-1]
ppm = keel_c.iloc[:,1]

fig, ax = plt.subplots(figsize = (8,6), dpi = 300)


pars = np.polyfit(year[10:780], ppm[10:780],3)


ax.scatter(year, ppm,
           marker = '.',
           s = 30)

def poly_coef(coef):
    n = len(coef) - 1
    return lambda x: sum([coef[i] * x**(n-i) for i in range(n+1)])


y_fit = poly_coef(pars)
year_fitted = np.arange(1960,2030,1/12)
ax.plot(year_fitted, y_fit(year_fitted), 'r--')



ax.set_xlim([1970,2030])
ax.set_ylim([320,440])





ax.set_xlabel('Year', fontsize = 22)
ax.set_ylabel(f'CO$_{2}$ Concentration [ppm]', fontsize = 22)

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

ax.legend(['Experimental data', 'Polynomial Fit'],
          loc = [0.05,0.8],
          fontsize = 18,
          facecolor = 'white',
          labelspacing= 0.4,
          framealpha = 1
          )


#%% Efficiency Table


eff_table = r'C:\Users\admin\Dysk Google\STUDIA\PROGRAMY I SKRYPTY\Python\PV_data_PhD\PV_efficiency.csv'

eff_table = pd.read_csv(eff_table, delimiter = ';')

year = np.array(eff_table['Year'])

mono = np.array(eff_table['mono si'])
pero_tan = np.array(eff_table['perovskite on si'])
pero = np.array(eff_table['perov'])



fig, ax = plt.subplots(figsize = (8,6), dpi = 400)

         
colors = [np.array([23, 156, 125])/255,
        np.array([242,148,0])/255,
        np.array([31,130,192])/255]

ax.set_xticks([1990,1995,2000,2005,2010,2015,2020, 2022,2025])
ax.set_yticks([10,12,14,16,18,20,22,24,26,28,30,32])


ax.set_xlim([1995,2023])
ax.set_ylim([14,32])


ax.set_title('Maximum Efficiency by Technology',
             fontsize = 22)




# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right",
         rotation_mode="anchor")

ax.set_xlabel('Year', fontsize = 22)
ax.set_ylabel('Efficiency [%]', fontsize = 22)

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
               length = 10)


m_size = 20
line = 4
f_size = 20

ax.plot(year[mono != 0], mono[mono != 0], color = colors[2],
           linestyle = '--',
           linewidth = line,
           marker = '.',
           markersize = m_size)

ax.plot(year[pero_tan != 0], pero_tan[pero_tan != 0], color = colors[1],
           linestyle = '--',
           linewidth = line,
           marker = '.',
           markersize = m_size)

ax.plot(year[pero != 0], pero[pero != 0], linestyle = '--', color = colors[0],
           linewidth = line,
           marker = '.',
           markersize = m_size)


ax.axvline(x = 2022,
           alpha = 0.8,
           linestyle = '--',
           linewidth = 2.0,
           color = [0.8,0.2,0.3],
           zorder = 5)

ax.annotate(f"",
    xy=[2022.1, 31.25],
    xytext=(35, 0),
    textcoords="offset points",
    zorder = 15,
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.0", lw = 3, color = colors[1]))

ax.annotate("$\\bf{pero–silicon}$\n31.3%",
    xy=[2022.1, 29.7],
    xytext=(35, 0),
    textcoords="offset points",
    fontsize = f_size,
    color = colors[1])


ax.annotate(f"",
    xy=[2022.1, 26.8],
    xytext=(35, 0),
    textcoords="offset points",
    color = colors[1],
    zorder = 15,
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.0", lw = 3, color = colors[2]))

ax.annotate("$\\bf{silicon}$\n26.8%",
    xy=[2022.1, 25.3],
    xytext=(35, 0),
    textcoords="offset points",
    fontsize = f_size,
    color = colors[2])


ax.annotate(f"",
    xy=[2022.1, 23.7],
    xytext=(35, 0),
    textcoords="offset points",
    color = colors[2],
    zorder = 15,
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.0", lw = 3, color = colors[0]))

ax.annotate("$\\bf{perovskite}$\n23.6%",
    xy=[2022.1, 22.2],
    xytext=(35, 0),
    textcoords="offset points",
    fontsize = f_size,
    color = colors[0])





#%%









