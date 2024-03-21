import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import wcs
import scipy.constants as c
import sys
import gas_cubes_tools as gm
import math

from astropy.io import fits
from astropy.coordinates import SkyCoord

""" 
directory = '/Users/camilogonzalezruilova/Documents/ODISEA_Gas/BDs_ODISEA/RA16_31_36.770/' #Path where is the cube
data_cube = directory+'RA16_31_36.770_CO_cube.fits' #Fits cube 
data_cont = directory+'RA16_31_36.770_cont_HR.fits' #Fits continuum for the contours

v0 = -0.3 #Initial velocity channel for the plot [km/s]
vf =  5.6 #Final velocity channel for the plot [km/s]

n_rms = 5 #Number of channels at the beginning and the end of the cube for rms measurement

d = 17.0 #diameter of image in arcsecs

 """
################################################################################################################################
################################################################################################################################
################################################################################################################################

def open_continuum(cont_fits):
   hdulist_cont = fits.open(cont_fits)
   hdr_cont = hdulist_cont[0].header
   data_cont = hdulist_cont[0].data
   images_cont = data_cont[:,:,0,0] if np.shape(np.shape(data_cont))[0] >= 4 else data_cont
   return images_cont, hdr_cont

def plot_channel_maps(cube_fits, velocity_range=None, output_name=None, contours=None, continuum_contours=(None,None), rms_nchan=None, cmap=None, radius_map=None):
   data_cube, hdr_cube = gm.open_cube(cube_fits)
   n_chan = rms_nchan if rms_nchan!=None else 5 
   rms = gm.get_rms(cube_fits, n_chan=n_chan)

   w_cube = wcs.WCS(hdr_cube)
   
   if continuum_contours != (None,None):
      continuum_fits, range_contours = continuum_contours
      data_cont, hdr_cont = open_continuum(continuum_fits)
      maxval_cont = np.nanmax(data_cont)
      w_cont = wcs.WCS(hdr_cont)
      max_index_cont = np.unravel_index(np.nanargmax(data_cont, axis=None), data_cont.shape)
      max_index_cont = np.array([max_index_cont], dtype=np.float64)
      world = w_cont.wcs_pix2world(max_index_cont,0)[0]
      pix_coord = w_cube.wcs_world2pix([[world[0],world[1],0]], 0)[0]

   if continuum_contours == (None,None):
      ind_chan = data_cube[0]
      xl, yl = np.shape(ind_chan)[1]/2 , np.shape(ind_chan)[0]/2
      pix_coord = [int(xl), int(yl)]


   hdr_freq = hdr_cube['CRVAL3']/1e9 #Frequency in GHz
   hdr_delt_freq = hdr_cube['CDELT3']/1e9 #GHz
   restfreq = hdr_cube['RESTFRQ']/1e9 #GHz
   v0_cube = np.round((1 - hdr_freq/restfreq)*c.c/1000, decimals=2) #Initial velocity of the cube [km/s]
   dchannel = np.round(abs(hdr_delt_freq/restfreq)*c.c/1000.0,decimals=3) #Delta velocity [km/s]


   velocity_array = np.array([])
   for vels_ran in velocity_range.split(','):
      if ':' in vels_ran: 
         v0 , vf = float(vels_ran.split(':')[0]), float(vels_ran.split(':')[1])
         velocity_array = np.append(velocity_array,np.arange(v0,vf+dchannel,dchannel))
      else:
         velocity_array = np.append(velocity_array,np.array([float(vels_ran)]))

   channels_array = (velocity_array - v0_cube)/dchannel

   n_plots = np.shape(channels_array)[0]

   y_figure = int(np.sqrt(n_plots))
   x_figure = int(n_plots/y_figure)+1 if n_plots%y_figure != 0 else int(n_plots/y_figure)

   y0_recentered, x0_recentered = pix_coord[0], pix_coord[1]
   dx=hdr_cube['CDELT1']*3600. #deg2arcsec
   dy=hdr_cube['CDELT2']*3600. #deg2arcsec

   x_extent = (np.array([0., data_cube.shape[1]]) - (x0_recentered)) * -dx 
   y_extent = (np.array([0., data_cube.shape[2]]) - (y0_recentered)) * dy 
   extent = [x_extent[0], x_extent[1], y_extent[0], y_extent[1]]

   
   region, threshold = contours
   mask_contour, hdr_mask = gm.get_mask(cube_fits,region=region,threshold=threshold)

   fig, axs = plt.subplots(y_figure,x_figure,figsize=(x_figure*1.05,y_figure))

   for i in np.arange(n_plots):
      i = int(i)
      j, k = int(i/x_figure), int(i%x_figure) #subplot coordinate
      channel_map = data_cube[int(np.round(channels_array[i]))] #Jy/beam
      mask_map = mask_contour[int(np.round(channels_array[i]))]

      maxval_cube = np.nanmax(data_cube)
      minval_cube = np.nanmin(data_cube)

###############################################################
      maxval_chan = np.nanmax(channel_map*mask_map)
      color_map = cmap if cmap!=None else 'inferno'
      color_map = 'Greys' if maxval_chan < threshold else cmap
################################################################
      im = axs[j,k].imshow(channel_map,extent=extent,vmin=-rms,  \
                           vmax=maxval_cube,cmap=color_map,origin='lower',  \
                           aspect='auto')


      axs[j,k].xaxis.set_ticklabels([]) if j!=y_figure-1 or k>0 else ''
      axs[j,k].yaxis.set_ticklabels([]) if j!=y_figure-1 or k>0 else ''
      axs[j,k].tick_params(direction='in',color='white')
      axs[j,k].set_ylabel('$\Delta$DEC [arcsec]') if j==y_figure-1 and k==0 else '' 
      axs[j,k].set_xlabel('$\Delta$RA [arcsec]') if j==y_figure-1 and k==0 else ''
      
      d = radius_map if radius_map!=None else x_extent[1]
      axs[j,k].set_xlim(-d,d)
      axs[j,k].set_ylim(-d,d)

      from matplotlib.patches import Ellipse        
      bmaj = hdr_cube['BMAJ'] * 3600.
      bmin = hdr_cube['BMIN'] * 3600.
      bpa = hdr_cube['BPA'] - 0.
      e = Ellipse(xy=[-d*0.8,-d*0.8], width=bmin, height=bmaj, angle=bpa, edgecolor='black', facecolor='white')
      axs[j,k].add_artist(e)
      text_color = 'black' if color_map=='Greys' else 'white'
      axs[j,k].text(0.3,0.9,'{0:.1f}'.format(velocity_array[i])+ ' km s$^{-1}$',fontsize=5.5,transform=axs[j,k].transAxes,color=text_color)
       
      axs[j,k].contour(mask_map,[1.],colors='white',origin='lower',extent=extent, linewidths=0.5)

      axs[j,k].axis('off') if (x_figure*j)+(k+1)>n_plots else '' 

      for n in np.arange(n_plots,y_figure*x_figure)%x_figure:
         axs[int(y_figure-1),int(n)].axis('off')

   plt.subplots_adjust(wspace=0.01,hspace=0.01)
   cbar_ax = fig.add_axes([0.91, 0.11, 0.03, 0.75])
   cbar = fig.colorbar(im,cax=cbar_ax)
   cbar.ax.tick_params(labelsize=8)
   fig.axes[-1].set_title('mJy beam$^{-1}$ km s$^{-1}$',fontsize=8.0,loc='left')

   fig.savefig(output_name,format='pdf', dpi=300,bbox_inches = 'tight')


"""
hdulist_cube = fits.open(data_cube)
hdr_cube = hdulist_cube[0].header
data_cube = hdulist_cube[0].data
hdulist_cont = fits.open(data_cont)
hdr_cont = hdulist_cont[0].header
data_cont = hdulist_cont[0].data

print(hdr_cube)

rms = (np.nanstd(data_cube[:n_rms+1])+np.nanstd(data_cube[-n_rms:]))*1000/2 #mJy/Beam
print('RMS is {0} mJy/beam'.format(rms))

w_cont = wcs.WCS(hdr_cont)

max_index_cont = np.unravel_index(np.nanargmax(data_cont, axis=None), data_cont.shape)
max_index_cont = np.array([max_index_cont], dtype=np.float64)
world = w_cont.wcs_pix2world(max_index_cont,0)[0]
w_cube = wcs.WCS(hdr_cube)
pix_coord = w_cube.wcs_world2pix([[world[0],world[1],0]], 0)[0]

hdr_freq = hdr_cube['CRVAL3']/1e9 #Frequency in GHz
hdr_delt_freq = hdr_cube['CDELT3']/1e9 #GHz
restfreq = hdr_cube['RESTFRQ']/1e9 #GHz
v0_cube = np.round(((restfreq - hdr_freq)/restfreq)*c.c/1000, decimals=2) #Initial velocity of the cube [km/s]
dchannel = np.round(abs(1 - (restfreq-hdr_delt_freq)/restfreq)*c.c/1000.0,decimals=3) #Delta velocity [km/s]

n_plots = np.round((vf - v0)/dchannel)+1

chan0 = int((v0-v0_cube)/dchannel)-1

data_cube_vel = data_cube[chan0:int(chan0+n_plots)]

y_figure = int(np.sqrt(n_plots))
x_figure = int(n_plots/y_figure)+1 if n_plots%y_figure != 0 else int(n_plots/y_figure)

y0_recentered, x0_recentered = pix_coord[0], pix_coord[1]
dx=hdr_cube['CDELT1']*3600. #deg2arcsec
dy=hdr_cube['CDELT2']*3600. #deg2arcsec

x_extent = (np.array([0., data_cube.shape[1]]) - (x0_recentered)) * -dx 
y_extent = (np.array([0., data_cube.shape[2]]) - (y0_recentered)) * dy 
extent = [x_extent[0], x_extent[1], y_extent[0], y_extent[1]]

cmap='inferno'

fig, axs = plt.subplots(x_figure,y_figure,figsize=(8.5,9.5))

for i in np.arange(n_plots):
   i = int(i)
   j, k = int(i/y_figure), int(i%y_figure) #subplot coordinate
   channel_map = data_cube_vel[int(i)]*1000 #mJy
   
   maxval_cube = np.nanmax(data_cube_vel)*1000
   minval_cube = np.nanmin(data_cube_vel)

###############################################################
   maxval_cont = np.nanmax(data_cont)
   maxval_chan = np.nanmax(channel_map)
################################################################


   im = axs[j,k].imshow(channel_map,extent=extent,vmin=-rms,  \
                   vmax=maxval_cube,cmap=cmap,origin='lower',  \
                   aspect='auto')


   axs[j,k].xaxis.set_ticklabels([]) if j<x_figure-1 or k>0 else ''
   axs[j,k].yaxis.set_ticklabels([]) if j<x_figure-1 or k>0 else ''
   axs[j,k].tick_params(direction='in',color='white')
   axs[j,k].set_ylabel('$\Delta$DEC [arcsec]') if j==x_figure-1 and k==0 else '' 
   axs[j,k].set_xlabel('$\Delta$RA [arcsec]') if j==x_figure-1 and k==0 else ''

   axs[j,k].set_xlim(-d,d)
   axs[j,k].set_ylim(-d,d)

   #divider = make_axes_locatable(axs[j,k])
   #cax = divider.new_vertical(size = '5%', pad = 0.02)
   #fig.add_axes(cax)
   #cbar = fig.colorbar(im,cax=cax,orientation='horizontal',ticklocation='top') 
   #cbar.ax.tick_params(direction='in')
   #cbar.set_label('mJy beam$^{-1}$') if j==0 and k==0 else ''

   from matplotlib.patches import Ellipse        
   bmaj = hdr_cube['BMAJ'] * 3600.
   bmin = hdr_cube['BMIN'] * 3600.
   bpa = hdr_cube['BPA'] - 0.
   e = Ellipse(xy=[-d*0.8,-d*0.8], width=bmin, height=bmaj, angle=bpa, edgecolor='black', facecolor='white')
   axs[j,k].add_artist(e)
   axs[j,k].text(0.3,0.9,'{0:.1f}'.format(v0+dchannel*i)+ ' km s$^{-1}$',fontsize=5.5,transform=axs[j,k].transAxes,color='white')

   levels = (5*np.arange(1,16))*rms

   axs[j,k].contour(channel_map,levels,colors='white',origin='lower',extent=extent, linewidths=0.5)


   #axs[4,3].axis('off')

for n in np.arange(n_plots,x_figure*y_figure)%y_figure:
   axs[int(x_figure-1),int(n)].axis('off')

plt.subplots_adjust(wspace=0.01,hspace=0.01)
cbar_ax = fig.add_axes([0.91, 0.11, 0.03, 0.75])
cbar = fig.colorbar(im,cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
fig.axes[-1].set_title('mJy beam$^{-1}$ km s$^{-1}$',fontsize=8.0,loc='left')

fig.savefig(directory+'Channel_maps.pdf',format='pdf', dpi=300,bbox_inches = 'tight')
 """