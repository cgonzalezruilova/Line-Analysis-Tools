from astropy.io import fits
import numpy as np
import time
import scipy.constants as c
import matplotlib.pyplot as plt

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
    y0 , x0 = center
    blank_image = np.zeros([np.shape(image)[0],np.shape(image)[1]])
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


def spectrum(cube_fits,region=None,channel_range=None):
    cube, hdr = open_cube(cube_fits)
    velocities, channels = chans2vel(cube_fits,channel_range=channel_range)
    if region!=None:
        mask_region = define_fig(image=cube[0], form=region[0], center=region[1], radius=region[2])
        fluxes = [np.nansum(cube[channel]*mask_region) for channel in channels]
    else:
        fluxes = [np.nansum(cube[channel]) for channel in channels]
    return fluxes, velocities, channels


def plot_spectrum(cube_fits,region=None,channel_range=None,velocities=None, molecule=None):
    fluxes, velocities, channels = fluxes, velocities, channels
    outputname = cube_fits.split('.fits')[0]
    label = molecule if molecule != None else ''
    if velocities == True:
        plt.plot(velocities, fluxes, label = label)
        plt.xlabel('Velocity [km s$^{-1}$]')
    else:
        plt.plot(channels, fluxes, label = label)
        plt.xlabel('Channels')
    plt.legend()
    plt.savefig('{0}_{1}_spectrum.pdf'.format(outputname,label))


def auto_vel(cube_fits,n_chan=None,rms_times=None,region=None,channel_range=None):
    n_chan = 5 if n_chan==None else n_chan
    rms_times = 3 if rms_times==None else rms_times
    fluxes, velocities, channels = spectrum(cube_fits,region=region,channel_range=channel_range)
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






