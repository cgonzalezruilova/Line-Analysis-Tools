import numpy as np
import matplotlib.pyplot as plt
from astropy import wcs
import gas_cubes_tools as gm

""" 
This routine produces channel maps (pdf) from a data cube with FITS format. 
Inputs:
- cube_fits: The path and name of the data cube in string format, e.g: cube_fits='my_path/my_cube.fits'; cube_fits='my_cube.fits'.
- channel_range: A string with the chosen channels to plot, separating by ":" a range of channels, and by a "," for sub-ranges, 
  e.g: channel_range = '2:20,28:53' corresponding to channels between 2 to 20 and 28 to 53.
- contours: 
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

def plot_channel_maps(cube_fits, channel_range=None, output_name=None, contours=None, continuum_contours=(None,None), rms_nchan=None, cmap=None, radius_map=None):
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

   vel_range, chan_range = gm.chans2vel(cube_fits, channel_range=channel_range)
   n_plots = np.shape(chan_range)[0]

   y_figure = int(np.sqrt(n_plots))
   x_figure = int(n_plots/y_figure)+1 if n_plots%y_figure != 0 else int(n_plots/y_figure)

   y0_recentered, x0_recentered = pix_coord[0], pix_coord[1]
   dx=hdr_cube['CDELT1']*3600. #deg2arcsec
   dy=hdr_cube['CDELT2']*3600. #deg2arcsec

   x_extent = (np.array([0., data_cube.shape[1]]) - (x0_recentered)) * dx 
   y_extent = (np.array([0., data_cube.shape[2]]) - (y0_recentered)) * dy 
   extent = [x_extent[0], x_extent[1], y_extent[0], y_extent[1]]

   
   region, threshold = contours if contours != None else None, 3*rms
   mask_contour, hdr_mask = gm.get_mask(cube_fits,region=region,threshold=threshold)

   fig, axs = plt.subplots(y_figure,x_figure,figsize=(x_figure*1.05,y_figure))

   for i in np.arange(n_plots):
      i = int(i)
      j, k = int(i/x_figure), int(i%x_figure) #subplot coordinate
      channel_map = data_cube[int(chan_range[i])] #Jy/beam
      channel_map = np.nan_to_num(channel_map, nan=np.nanmin(channel_map))
      mask_map = mask_contour[int(chan_range[i])] if mask_contour != None else 1.

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


      #axs[j,k].xaxis.set_ticklabels([]) if j!=y_figure-1 or k>0 else ''
      #axs[j,k].yaxis.set_ticklabels([]) if j!=y_figure-1 or k>0 else ''
      axs[j,k].xaxis.set_ticklabels([]) if j!=y_figure-1 else axs[j,k].xaxis.set_tick_params(labelsize=7)
      axs[j,k].yaxis.set_ticklabels([]) if k>0 else axs[j,k].yaxis.set_tick_params(labelsize=7)
      axs[j,k].tick_params(direction='in',color='white')

      d = radius_map if radius_map!=None else x_extent[1]
      axs[j,k].set_xlim(d,-d)
      axs[j,k].set_ylim(-d,d)

      from matplotlib.patches import Ellipse        
      bmaj = hdr_cube['BMAJ'] * 3600.
      bmin = hdr_cube['BMIN'] * 3600.
      bpa = hdr_cube['BPA'] - 0.
      e = Ellipse(xy=[d*0.8,-d*0.8], width=bmin, height=bmaj, angle=bpa, edgecolor='black', facecolor='white')
      axs[j,k].add_artist(e)
      text_color = 'black' if color_map=='Greys' else 'white'
      axs[j,k].text(0.3,0.9,'{0:.1f}'.format(vel_range[i])+ ' km s$^{-1}$',fontsize=5.5,transform=axs[j,k].transAxes,color=text_color)
       
      axs[j,k].contour(mask_map,[1.],colors='white',origin='lower',extent=extent, linewidths=0.5)

      axs[j,k].axis('off') if (x_figure*j)+(k+1)>n_plots else '' 

      for n in np.arange(n_plots,y_figure*x_figure)%x_figure:
         axs[int(y_figure-1),int(n)].axis('off')
      print('Plotted channel {0}'.format(int(chan_range[i])))

   plt.subplots_adjust(wspace=0.01,hspace=0.01)
   fig.text(0.5, 0.01*y_figure, '$\Delta$RA [$arcsec$]', ha='center', va='center')
   fig.text(0.01*x_figure, 0.5, '$\Delta$DEC [$arcsec$]', ha='center', va='center',rotation='vertical')
   #fig.supylabel('$\Delta$DEC [$arcsec$]')
   #fig.supxlabel('$\Delta$RA [$arcsec$]')
   cbar_ax = fig.add_axes([0.91, 0.11, 0.03, 0.75])
   cbar = fig.colorbar(im,cax=cbar_ax)
   cbar.ax.tick_params(labelsize=8)
   cbar.set_label('mJy beam$^{-1}$ km s$^{-1}$',rotation=270,fontsize=8, labelpad=10)
   #fig.axes[-1].set_title('mJy beam$^{-1}$ km s$^{-1}$',fontsize=8.0,loc='left')


   fig.savefig(output_name,format='pdf', dpi=300,bbox_inches = 'tight')
