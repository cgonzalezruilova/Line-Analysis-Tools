from astropy.visualization import make_lupton_rgb
from gas_cubes_tools import *




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

def rms_mask(cube_data, threshold=None, n_chan_rms=None):
    rms = get_rms(cube_data, n_chan=n_chan_rms)
    mask, hdr = get_mask(cube_data, threshold=threshold*rms)
    rms_mask = np.array(np.sum(mask, axis=0))
    rms_mask = np.where(rms_mask > 0.0, np.nan, rms_mask)
    rms_mask = np.where(rms_mask == 0.0, 1.0, rms_mask)
    return rms_mask

def threshold_moment_map(cube_data, moment_map, threshold=None, n_chan_rms=None):
    moment_image, moment_hdr = open_cube(moment_map)
    moment_image_residual = moment_image*rms_mask(cube_data, threshold=threshold, n_chan_rms=n_chan_rms)
    rms = np.sqrt(np.array(np.nanmean(moment_image_residual**2)))
    return rms



def overlay_fits_images_rgb(image_paths, thresholds, brightness_factors, output_path, fov=None, distance=None):
    # Cargar y procesar las imágenes para cada canal de color
    red_image, red_hdr = open_cube(image_paths[0])
    green_image, green_hdr = open_cube(image_paths[1])
    blue_image, blue_hdr = open_cube(image_paths[2])


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

    fov = int(fov/(float(red_hdr['CDELT1'])*3600)) if fov!=None else int(float(red_hdr['NAXIS1'])/2)-1
    reference_label = 500*(1/(red_hdr['CDELT1']*3600))/distance
    y0, y1 = int(np.shape(red_image)[0]/2 - fov), int(np.shape(red_image)[0]/2 + fov)
    x0, x1 = int(np.shape(red_image)[1]/2 - fov), int(np.shape(red_image)[1]/2 + fov)
    plt.plot([0.5*(np.shape(red_image)[0]-reference_label), 0.5*(np.shape(red_image)[0]+reference_label)], [int(y1 - (y1-y0)*0.92), int(y1 - (y1-y0)*0.92)], color="white", linewidth=5)
    plt.text( int((x0+x1)/2), int(y1 - (y1-y0)*0.95), '500 au',color='white', fontsize='xx-large',ha="center", va="center")
    plt.axis('off')  # Opcional: para quitar los ejes
    plt.xlim(x0, x1)
    plt.ylim(y0, y1)
    plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0)


