from astropy.io import fits
import numpy as np
import time
import scipy.constants as c
import matplotlib.pyplot as plt
import bettermoments as bm
from scipy.ndimage import rotate
from decimal import Decimal

def open_cube(cube_fits):
    hdulist_cube = fits.open(cube_fits)
    hdr_cube = hdulist_cube[0].header
    data_cube = hdulist_cube[0].data
    images_cube = datacube[:,:,:,0] if np.shape(np.shape(data_cube))[0] >= 4 else data_cube 
    return images_cube, hdr_cube


def get_rms(cube_fits,n_chan=None):
    rms = []
    n_chan = 5 if n_chan==None else n_chan
    cube, hdr = open_cube(cube_fits)
    for i in np.append(np.arange(n_chan),-(np.arange(n_chan)+1)),:
        rms += [np.sqrt(np.nanmean(cube[i]**2))]
    mean_rms = np.mean(rms)
    print('The RMS for the first and final {0} channels is: {1} Jy/beam'.format(n_chan,mean_rms))
    return mean_rms


def total_flux(cube_fits,threshold=None,region=None,velocity_range=None):
    start = time.time()
    cube, hdr = open_cube(cube_fits)
    beam_area_FWHM = abs((hdr['BMAJ']/hdr['CDELT1'])*(hdr['BMIN']/hdr['CDELT1'])*np.pi/(4*np.log(2)))
    channel_flux = 0.0
    chan_width = abs((hdr['CDELT3']/hdr['RESTFRQ'])*c.c/1000.0) #in km/s
    if region!=None: mask_region = define_fig(image=cube[0], form=region[0], center=region[1], radius=region[2])
    chans_range = np.arange(np.shape(cube)[0]) if velocity_range==None else vel2chans(cube_fits,velocity_range=velocity_range)
    #for image in cube:
    for chan in chans_range:
        image = cube[int(chan)]
        if region!=None: image = image*mask_region
        if np.nanmax(image) < threshold: continue
        for array in image:
            for pixel in array:
                channel_flux += pixel if pixel >= threshold else 0.0
    total_int = channel_flux*chan_width
    total_flux = channel_flux*chan_width/beam_area_FWHM
    print('The total intensity is {0} Jy/beam km/s'.format(total_int))
    print('The total flux is {0} Jy km/s'.format(total_flux))
    end = time.time()
    print('{0} s'.format(end-start))
    return total_int, total_flux


def get_mask(cube_fits,region=None,threshold=None):
    start = time.time()
    mask = []
    cube, hdr = open_cube(cube_fits)
    mask_region = define_fig(cube[0], form=region[0], center=region[1], radius=region[2]) if region!=None else 1.0 
    for image in cube:
        new_channel = []
        image = image*mask_region
        for array in image:
            new_row = []
            for pixel in array:
                new_row += [1.0] if pixel >= threshold else [0.0]
            new_channel += [new_row]
        mask += [new_channel]
    end = time.time()
    print('{0} s'.format(end-start))
    return mask, hdr


def mask_fits(cube_fits, region=None, threshold=None, output_name=None):
    mask, hdr = get_mask(cube_fits,threshold=threshold,region=region)
    hdu = fits.PrimaryHDU(data=mask, header=hdr)
    hdul = fits.HDUList([hdu])
    hdul.writeto(output_name,overwrite=True)

def define_fig(image, form=None, center=None, radius=None):
    if center != None:
        y0, x0 = center      
    if center == None:
        y0, x0 = np.shape(image)[0]/2 , np.shape(image)[1]/2
    blank_image = np.zeros([np.shape(image)[0],np.shape(image)[1]])
    blank_image[:] = np.nan
    I,J=np.meshgrid(np.arange(blank_image.shape[0]),np.arange(blank_image.shape[1]))
    if form == 'square':
        return image[y0-radius:y0+radius, x0-radius:x0+radius]
    if form == 'circle':
        dist=np.sqrt((I-y0)**2+(J-x0)**2)
        blank_image[np.where(dist<radius)]=1
        return blank_image
    if form == 'ellipse':
        a, b = radius
        dist=np.sqrt(((I-y0)/a)**2+((J-x0)/b)**2)
        blank_image[np.where(dist<1)]=1
        return blank_image


def vel2chans(cube_fits,velocity_range=None):
    import scipy.constants as c
    cube, hdr = open_cube(cube_fits)
    hdr_freq = hdr['CRVAL3']/1e9 #Frequency in GHz
    hdr_delt_freq = hdr['CDELT3']/1e9 #GHz
    restfreq = hdr['RESTFRQ']/1e9 #GHz
    v0_cube = np.round((1 - hdr_freq/restfreq)*c.c/1000, decimals=2) #Initial velocity of the cube [km/s]
    dchannel = np.round(abs(hdr_delt_freq/restfreq)*c.c/1000.0,decimals=3) #Delta velocity [km/s]
    velocity_array = np.array([])
    for vels_ran in velocity_range.split(','):
        if ':' in vels_ran: 
           v0 , vf = float(vels_ran.split(':')[0]), float(vels_ran.split(':')[1])
           velocity_array = np.append(velocity_array,np.arange(v0,vf+dchannel,dchannel))
        else:
           velocity_array = np.append(velocity_array,np.array([float(vels_ran)]))
    channels_array = np.round((velocity_array - v0_cube)/dchannel)
    return channels_array[:]


def chans2vel(cube_fits,channel_range=None):
    import scipy.constants as c

    cube, hdr = open_cube(cube_fits)
    if channel_range == None:
        channels_array = np.arange(np.shape(cube)[0])
    else:
        channels_array = []
        for chan_ran in channel_range.split(','):
            if ':' in chan_ran:
                chan_0 , chan_f = int(chan_ran.split(':')[0]), int(chan_ran.split(':')[1])
                channels_array += [np.arange(chan_0, chan_f+1)]
            else:
                channels_array += [int(chan_ran)] 

    hdr_freq = hdr['CRVAL3']/1e9 #Frequency in GHz
    hdr_delt_freq = hdr['CDELT3']/1e9 #GHz
    restfreq = hdr['RESTFRQ']/1e9 #GHz
    v0_cube = np.round((1 - hdr_freq/restfreq)*c.c/1000, decimals=2) #Initial velocity of the cube [km/s]
    dchannel = np.round(abs(hdr_delt_freq/restfreq)*c.c/1000.0,decimals=3) #Delta velocity [km/s]

    velocity_array = np.round([channel*dchannel+v0_cube for channel in channels_array],2)
    return velocity_array, channels_array


def F2T(Flux, freq, bmaj, bmin):
    '''
    Flux in Jy/Beam
    freq in GHz
    bmaj and bmin in arcsec
    '''
    conversion = 1.222e6/(freq**2*bmaj*bmin)
    print('conversion = {:.2E} K / Jy beam-1'.format(Decimal(conversion)))
    return conversion*Flux

def spectrum(cube_fits,region=None,channel_range=None):
    cube, hdr = open_cube(cube_fits)
    velocities, channels = chans2vel(cube_fits,channel_range=channel_range)
    if region!=None:
        mask_region = define_fig(image=cube[0], form=region[0], center=region[1], radius=region[2])
        fluxes = [np.nansum(cube[channel]*mask_region) for channel in channels]
        mean_flux = np.array([np.nanmean(cube[channel]*mask_region) for channel in channels])
    else:
        
        fluxes = [np.nansum(cube[channel]) for channel in channels]
        mean_flux = np.array([np.nanmean(cube[channel]) for channel in channels])

    temperatures = F2T(mean_flux,hdr['RESTFRQ']/1e9,hdr['BMAJ']*3600.,hdr['BMIN']*3600.)
    return fluxes, mean_flux,temperatures, velocities, channels



def auto_vel(cube_fits,n_chan=None,rms_times=None,region=None,channel_range=None):
    n_chan = 5 if n_chan==None else n_chan
    rms_times = 3 if rms_times==None else rms_times
    fluxes, temperatures, velocities, channels = spectrum(cube_fits,region=region,channel_range=channel_range)
    auto_velocities = []
    auto_channels = []
    rms = np.sqrt(np.mean(np.array((fluxes[0:n_chan]+fluxes[np.shape(fluxes)[0]-1-n_chan:]))**2))
    threshold = rms*rms_times
    for flux in fluxes:
        if flux >= threshold:
            fluxes_arr = np.array(fluxes)
            flux_pos = np.where(fluxes_arr==flux)[0][0]
            auto_velocities += [velocities[flux_pos]]
            auto_channels += [channels[flux_pos]]
    return auto_velocities, auto_channels


def generate_moment_map(cube_fits,channel_range=None,smooth=None,polyorder=None,N_chan_rms=None,mask=None,clipping=None,smooth_threshold_level=None,outname=None,moment=None):
    cube = cube_fits
    path = cube_fits
    data, velax = bm.load_cube(cube)
    data = data[int(channel_range[0]):int(channel_range[-1]+1),:,:]
    smoothing = smooth if smooth != None else 3
    polyorder_smooth = polyorder if polyorder != None else 0
    N = N_chan_rms if N_chan_rms != None else 5
    smooth_threshold_mask = smooth_threshold_level if smooth_threshold_level != None else 3.0
    first_channel = channel_range[0] if channel_range != None else 0
    last_channel = channel_range[1] if channel_range != None else -1
    last_channel = last_channel if last_channel == -1 else int(last_channel-np.shape(data)[2])
    output_name = outname if outname != None else cube_fits.split('.fits')[0]
    if moment == None: return "Choose a moment: zeroth or first" 

    smoothed_data = bm.smooth_data(data=data, smooth=smoothing, polyorder=polyorder_smooth)

    rms = bm.estimate_RMS(data=data, N=N)
    rms_smoothed = bm.estimate_RMS(data=smoothed_data, N=N)

    print('RMS = {:.1f} mJy/beam (original)'.format(rms * 1e3))
    print('RMS = {:.1f} mJy/beam (smoothed)'.format(rms_smoothed * 1e3))

    user_mask = bm.get_user_mask(data=data, user_mask_path=mask)
    
    threshold_mask = bm.get_threshold_mask(data=data,
                                           clip=clipping,
                                           smooth_threshold_mask=smooth_threshold_level)

    channel_mask = bm.get_channel_mask(data=data,
                                       firstchannel=0,
                                       lastchannel=-1)

    mask = bm.get_combined_mask(user_mask=user_mask,
                                threshold_mask=threshold_mask,
                                channel_mask=channel_mask,
                                combine='and')
    masked_data = smoothed_data * mask

    if moment == 'zeroth':
        moments_map = bm.collapse_zeroth(velax=velax, data=masked_data, rms=rms)
        bm.save_to_FITS(moments=moments_map, method='zeroth',outname=output_name,path=path)
    if moment == 'first':
        moments_map = bm.collapse_first(velax=velax, data=masked_data, rms=rms)
        bm.save_to_FITS(moments=moments_map, method='first',outname=output_name,path=path)

def get_channel_range(cube_fits,N_chan_rms=None,region=None,rms_threshold=None,blue_red_chans=None):
    fluxes, temperatures, velocities, channels = spectrum(cube_fits,region=region)
    peak_flux, pos_peak = np.max(fluxes), np.argmax(fluxes)
    delta_flux_0 = peak_flux - fluxes[0]
    delta_flux_f = peak_flux - fluxes[-1]
    N = N_chan_rms if N_chan_rms != None else 5
    rms = np.mean([np.sum(np.array(fluxes[0:N])**2)**(0.5)/N,np.sum(np.array(fluxes[-N-1:-1])**2)**(0.5)/N])
    real_emission = []
    real_emission_channel = []
    rms_threshold=rms_threshold if rms_threshold != None else 5
    #SNR = peak_flux/rms
    #SNR_ratio = SNR_ratio if SNR_ratio != None else 0.1
    #SNR_perc = SNR*SNR_ratio

    for flux in fluxes:
        if flux < rms*rms_threshold:
        #if flux/rms < SNR_perc:
            continue
        else:
            real_emission += [flux]
            real_emission_channel += [np.where(fluxes==flux)[0][0]]

    if blue_red_chans == True:
        dist_chans = []
        for i in np.arange(np.shape(real_emission_channel)[0]-1):
            dist_chans += [real_emission_channel[int(i+1)] - real_emission_channel[int(i)]]
        if np.shape(np.array(dist_chans))[0] == 0: return None,None
        where_peak = np.argmax(np.array(dist_chans))
        blue_chans = real_emission_channel[0:int(where_peak)+1]
        red_chans = real_emission_channel[int(where_peak)+1:]
        return blue_chans, red_chans
    else:
        return real_emission_channel


def blueshift_fits(cube_fits,region=None,channel_range=None,rms_threshold=None,smooth=None,polyorder=None,N_chan_rms=None,mask=None,clipping=None,smooth_threshold_level=None,outname=None,moment=None):
    blue_chans, red_chans = get_channel_range(cube_fits,N_chan_rms=N_chan_rms,region=region,rms_threshold=rms_threshold,blue_red_chans=True)
    if blue_chans==None or red_chans==None: return
    output_name = outname if outname != None else '{0}_blueshift'.format(cube_fits.split('.fits')[0])
    if np.shape(np.array(blue_chans))[0] == 1: blue_chans = [int(blue_chans[0]-1),int(blue_chans[0])]
    print('Blueshift Channels range: {0}'.format(blue_chans))
    
    generate_moment_map(cube_fits,
    channel_range=blue_chans,
    smooth=smooth,
    polyorder=polyorder,
    N_chan_rms=N_chan_rms,
    mask=mask,
    clipping=clipping,
    smooth_threshold_level=smooth_threshold_level,
    outname=output_name,
    moment=moment)

def redshift_fits(cube_fits,region=None,channel_range=None,rms_threshold=None,smooth=None,polyorder=None,N_chan_rms=None,mask=None,clipping=None,smooth_threshold_level=None,outname=None,moment=None):

    blue_chans, red_chans = get_channel_range(cube_fits,N_chan_rms=N_chan_rms,region=region,rms_threshold=rms_threshold,blue_red_chans=True)
    if blue_chans==None or red_chans==None: return
    output_name = outname if outname != None else '{0}_redshift'.format(cube_fits.split('.fits')[0])
    if np.shape(np.array(red_chans))[0] == 1: red_chans = [int(red_chans[0]),int(red_chans[0]+1)]
    print('Redshift Channels range: {0}'.format(red_chans))

    generate_moment_map(cube_fits,
    channel_range=red_chans,
    smooth=smooth,
    polyorder=polyorder,
    N_chan_rms=N_chan_rms,
    mask=mask,
    clipping=clipping,
    smooth_threshold_level=smooth_threshold_level,
    outname=output_name,
    moment=moment)


def pv_diagrams(cube_fits,coords=None,channels=None,av_width=None,outputname=None):

    if av_width == None: av_width = 0
    x0,y0,x1,y1 = coords[0][0],coords[0][1],coords[1][0],coords[1][1]
    cube, hdr = open_cube(cube_fits)
    if channels == None: channels = [0,np.shape(cube)[0]]
    av_width = int(av_width)
    av_width_a, av_width_b = int(av_width/2), av_width%2
    chan0, chanf = int(channels[0]),int(channels[1])
    cube_final = cube[chan0:chanf]
    alpha = np.rad2deg(np.arctan(np.sqrt((y0-y1)**2/(x0-x1)**2)))
    print(np.rad2deg(np.arctan(np.sqrt((y0-y1)**2/(x0-x1)**2))))
    pv_grid = []
    x0 = int(x0 - np.shape(cube)[2]/2)+1
    y0 = int(y0 - np.shape(cube)[1]/2)+1
    x1 = int(x1 - np.shape(cube)[2]/2)+1
    y1 = int(y1 - np.shape(cube)[1]/2)+1

    for chan in cube_final:
        chan[np.isnan(chan)] = 0
        image_rotated = rotate(chan, alpha, reshape=False)
        alpha_0 = np.deg2rad(-alpha)
        xn0 = int((x0*np.cos(alpha_0))-(y0*np.sin(alpha_0))+np.shape(chan)[1]/2)
        xn1 = int((x1*np.cos(alpha_0))-(y1*np.sin(alpha_0))+np.shape(chan)[1]/2)
        yn0 = int((x0*np.sin(alpha_0))+(y0*np.cos(alpha_0))+np.shape(chan)[0]/2)
        yn1 = int((x1*np.sin(alpha_0))+(y1*np.cos(alpha_0))+np.shape(chan)[0]/2)
        if av_width_a == 0 and av_width_b == 1:
            pv_row = image_rotated[yn0,min([xn0,xn1]):max([xn0,xn1])+1]
            pv_grid += [list(reversed(pv_row))]
        elif av_width_a != 0 and av_width_b != 1:
            pv_row = image_rotated[yn0-av_width_a:yn0+(av_width_a),min([xn0,xn1]):max([xn0,xn1])+1]
            pv_transpose = np.transpose(pv_row)
            pv_row_av = [np.mean(col) for col in pv_transpose]
            pv_grid += [list(reversed(pv_row_av))]
        else:
            pv_row = image_rotated[yn0-av_width_a:yn0+(av_width_a+1),min([xn0,xn1]):max([xn0,xn1])+1]
            pv_transpose = np.transpose(pv_row)
            pv_row_av = [np.mean(col) for col in pv_transpose]
            pv_grid += [list(reversed(pv_row_av))]
    if outputname==None: outputname='test'
    plt.imshow(cube[114],origin='lower',cmap='rainbow')
    plt.plot([coords[0][0],coords[1][0]],[coords[0][1],coords[1][1]],color='red')
    plt.savefig('{0}_with_PV_line.pdf'.format(outputname))
    image_rotated = rotate(cube[114], alpha, reshape=False)
    plt.imshow(image_rotated,origin='lower',cmap='rainbow')
    plt.plot([xn0,xn1],[yn0,yn1],color='red')
    plt.savefig('{0}_with_PV_line_rotated.pdf'.format(outputname))
    plt.imshow(pv_grid,origin='lower',vmin=np.min(pv_grid),vmax=np.max(pv_grid),cmap='rainbow',aspect='auto')
    plt.savefig('{0}_PV.pdf'.format(outputname))
    return pv_grid
    
[[182,168],[233,187]]

