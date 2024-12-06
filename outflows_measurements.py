import numpy as np
from decimal import Decimal
from gas_cubes_tools import *

def F2T(Flux, freq, bmaj, bmin):
    '''
    Flux in Jy/Beam
    freq in GHz
    bmaj and bmin in arcsec
    '''
    conversion = 1.222e6/(freq**2*bmaj*bmin)
    print('conversion = {:.2E} K / Jy beam-1'.format(Decimal(conversion)))
    return conversion*Flux


def N_CO(T_ext, J, tau0, Tb):
    column_density = 4.33e13*(T_ext/J**2)*np.exp((2.77*J*(J+1))/T_ext)*(tau0/(1-np.exp(-tau0)))*Tb
    print('{:.2E}'.format(Decimal(column_density)))
    return column_density


def mass_outflows(Area, dist_star, column_density):
    '''
    Area in arcsec2
    column_density in cm-2
    '''
    Area_pc = Area*(dist_star/206265)**2
    Mass = 2.25e-16*Area_pc*column_density
    print('The total mass is {:.2E} M_sun'.format(Decimal(Mass)))
    return Mass