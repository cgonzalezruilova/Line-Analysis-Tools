import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
from astropy.visualization import make_lupton_rgb
import math
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
import gas_cubes_tools as gct



source = 'RA16_27_04.097'



'''


# Moment maps for 12CO
gct.generate_moment_map(source+'_CO_cube.fits',channel_range=[30,95],moment='zeroth')
gct.generate_moment_map(source+'_CO_cube.fits',channel_range=[30,95],moment='first')
# Moment maps for 13CO
gct.generate_moment_map(source+'_13CO_cube.fits',channel_range=[30,90],moment='zeroth')
gct.generate_moment_map(source+'_13CO_cube.fits',channel_range=[30,90],moment='first')
# Moment maps for C18O
gct.generate_moment_map(source+'_C18O_cube.fits',channel_range=[30,90],moment='zeroth')
gct.generate_moment_map(source+'_C18O_cube.fits',channel_range=[30,90],moment='first')


'''

##################################################################
##################################################################
##################################################################


source = source

directory='/Volumes/Elements/ODISEA_GAS/'

sufix_0 = '_cube_M0.fits'

trans = ['CO','13CO','C18O']

rms = np.array([86, 121, 68])*3 #[12co,13co,c18o]

fov =  12.0  #radius of FOV in arcsecs

distance = 140.0 #Distance to star in pcs

CDELT1 = 2.777777777778e-05




fov = int(fov/(CDELT1*3600))  #radius of FOV in pixels
reference_label = 500*(1/(CDELT1*3600))/distance #500 au reference line length in pixels



moms_0 = [directory+source+'_'+line+sufix_0 for line in trans] 




##################################################################
##################################################################
##################################################################



import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import matplotlib.colors as mcolors

def load_fits_image(fits_path):
    with fits.open(fits_path) as hdul:
        # Suponemos que la imagen está en la extensión 0
        image_data = hdul[0].data
    return image_data

def preprocess_image(image_data, threshold):
    # Convertir píxeles menores que el umbral en NaN
    processed_image = np.where(image_data < threshold, np.nan, image_data)
    return processed_image

def stretch_contrast(image_data):
    # Reemplazar NaNs por 0 para el estiramiento de contraste
    image_data = np.nan_to_num(image_data, nan=0)
    # Estirar el contraste al rango [0, 1]
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    if max_val > min_val:
        stretched_image = (image_data - min_val) / (max_val - min_val)
    else:
        stretched_image = np.zeros_like(image_data)
    return stretched_image

def adjust_brightness(image_data, factor):
    # Ajustar el brillo de la imagen multiplicando por un factor
    brightened_image = np.clip(image_data, 0, 1) * factor
    return brightened_image

def overlay_fits_images_rgb(image_paths, thresholds, brightness_factors, output_path):
    # Cargar y procesar las imágenes para cada canal de color
    red_image = load_fits_image(image_paths[0])
    green_image = load_fits_image(image_paths[1])
    blue_image = load_fits_image(image_paths[2])


    red_image = preprocess_image(red_image, thresholds[0])
    green_image = preprocess_image(green_image, thresholds[1])
    blue_image = preprocess_image(blue_image, thresholds[2])

    # Aplicar estiramiento de contraste
    red_image = stretch_contrast(red_image)
    green_image = stretch_contrast(green_image)
    blue_image = stretch_contrast(blue_image)

    # Ajustar el brillo de la imagen roja
    red_image = adjust_brightness(red_image, brightness_factors[0])
        # Ajustar el brillo de la imagen roja
    green_image = adjust_brightness(green_image, brightness_factors[1])
        # Ajustar el brillo de la imagen roja
    blue_image = adjust_brightness(blue_image, brightness_factors[2])


    # Crear una imagen RGB combinada
    rgb_image = np.stack([red_image, green_image, blue_image], axis=-1)


    # Configurar el tamaño de la figura
    plt.figure(figsize=(10, 10))

    # Crear un colormap personalizado para manejar NaNs
    cmap = plt.get_cmap('gray')
    cmap.set_bad(color='black')  # Establecer el color para NaNs como negro

    # Mostrar la imagen RGB
    #plt.imshow(rgb_image, cmap=cmap)
    rgb_default = make_lupton_rgb(red_image, green_image, blue_image, Q=20, stretch=0.5)
    plt.imshow(rgb_default, origin='lower')
    
    # Guardar la imagen combinada
    y0, y1 = int(np.shape(red_image)[0]/2 - fov), int(np.shape(red_image)[0]/2 + fov)
    x0, x1 = int(np.shape(red_image)[1]/2 - fov), int(np.shape(red_image)[1]/2 + fov)
    plt.plot([0.5*(np.shape(red_image)[0]-reference_label), 0.5*(np.shape(red_image)[0]+reference_label)], [int(y1 - (y1-y0)*0.92), int(y1 - (y1-y0)*0.92)], color="white", linewidth=5)
    plt.text( int((x0+x1)/2), int(y1 - (y1-y0)*0.95), '500 au',color='white', fontsize='xx-large',ha="center", va="center")
    plt.axis('off')  # Opcional: para quitar los ejes
    plt.xlim(x0, x1)
    plt.ylim(y0, y1)
    plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0)
    plt.show()

# Rutas de las imágenes FITS a superponer (en orden: rojo, verde, azul)
image_paths = moms_0

# Umbrales para cada imagen, píxeles menores a estos valores se convertirán en NaN
thresholds = rms

# Factores de brillo para cada imagen (1.0 = sin cambio, >1.0 = más brillante)
brightness_factors = [2.0, 4.0, 10.0]  # Ejemplo: hacer la imagen roja 1.5 veces más brillante

# Ruta de salida para la imagen combinada
output_path = '{0}_combined_rgb_image.png'.format(source)

overlay_fits_images_rgb(image_paths, thresholds, brightness_factors, output_path)

