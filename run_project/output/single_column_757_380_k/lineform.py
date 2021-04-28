from helita.sim import rh15d
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from cmcrameri import cm
import numpy as np
from astropy import constants as const
from scipy.integrate import cumtrapz


class Rh15d_calc:
    """
    Class to calculate certain quantites from RH output.
    """
    def __init__(self, rh_object):
        self.data = rh_object

    def formation_diagram(self, x, y, wave_ref=None, vrange=None, zrange=None,
                          sbtrange=None, irange=None, drop_last_index=True,
                          cmap='gist_gray', cscale=None, color=None, linestyle=None ):
        """


        Parameters
        ----------
        color : 1D array
                see linestyle
        linestyle : 1D array
                Sets the line colors in the plot with following order:
                entry 0 -> tau = 1 curve
                entry 1 -> velocity curve
                entry 2 -> Atmospheric Temperature
                entry 3 -> Source function converted to Temp. units
                entry 4 -> Brightness Temperature of line profile
        """

        dim = self.data.ray.intensity.shape[0:2]
        if (x >= dim[0] or x < 0):
            print("X-coordinate out of range, setting to 0")
            x = 0
        if (y >= dim[0] or y < 0):
            print("Y-coordinate out of range, setting to 0")
            y = 0
        if not isinstance(x, int):
            raise ValueError("Insert a singe integer value for x coordinate")
        if not isinstance(y, int):
            raise ValueError("Insert a singe integer value for y coordinate")
        if wave_ref is None:
            raise ValueError('No reference wavelength specified!')
        if vrange is None:
            print('No velocity range specified, setting to 50 km/s')
            vrange = [-50.0, 50.0]
        if zrange is None:
            print('No height range specified, setting from 0 to 2 Mm')
            zrange = [0.0, 2.0]
        if sbtrange is None:
            print('No temperature range specified, setting from 4 to 10 kK')
            sbtrange = [3.5, 8.0]
        if irange is None:
            print('No brightness temperature range specified, setting from 4.5 to 8 kK')
            irange = [4.7, 5.1]
        if cscale is None:
            print('No Contribution function scaling specified, setting to 1')
            cscale = 1.0
        if color is None:
            print('No Color for the line plots specified, setting to default')
            color = ['#00CDFF', 'r', 'w', 'w', 'w']
        if linestyle is None:
            print('No Linestyle for the line plots specified, setting to default')
            linestyle = ['-', '-', ':', '--', '-']

        wave_in = self.data.ray.wavelength[self.data.ray.wavelength_indices[0:-1]].data
        vz_in=self.data.atmos.velocity_z[x, y, :].dropna('height').data
        temp = self.data.atmos.temperature[x ,y , :].dropna('height').data
        height_in = self.data.atmos.height_scale[x, y, :].dropna('height').data
        chi_in = self.data.ray.chi[x, y, :, 0:-1].dropna('height').data
        S_in = self.data.ray.source_function[x ,y , :, 0:-1].dropna('height').data
        sint = self.data.ray.intensity[x ,y ,self.data.ray.wavelength_indices[0:-1]].data

        if drop_last_index is False:
            wave_in = self.data.ray.wavelength[self.data.ray.wavelength_indices].data
            chi_in = self.data.ray.chi[x, y, :, :].dropna('height').data
            S_in = self.data.ray.source_function[x ,y , :, :].data
            sint = self.data.ray.intensity[x ,y ,self.data.ray.wavelength_indices].data

        plot_formation_diagram(chi_in, S_in, wave_in, height_in, vz_in, temp, sint,
                               vrange=vrange, wave_ref=wave_ref, zrange=zrange, sbtrange=sbtrange,
                               irange=irange, linestyle=linestyle, color=color, cmap=cmap)

def plot_formation_diagram(chi_in, S_in, wave_in, height_in, vz_in, temp, sint,
                vrange=[-50.0, 50.0], zrange=[0.0, 2.0], wave_ref=5000, cmap='gist_gray', cscale=1.0,
                sbtrange=[3.5, 8.0], irange=[4.7, 5.1], linestyle = ['-', '-', ':', '--', '-'], color= ['#00CDFF', 'r', 'w', 'w', 'w']):
    """ Plots, for the formation diagram, in four panels:

        * chi/tau
        * source function
        * tau*exp(-tau)
        * contribution function

        IN:
        tau_in  : 2D array with tau[wave,height] (in m^-1)
        tau_one : 1D array (function of wavelength) with height of tau=1 [Mm]
        wave    : 1D array with wavelength in nm (air)
        height  : 1D array with height (in m)
        vz      : 1D array with velocities in z (units in m/s)
        wave_ref: reference wavelength for the velocity subtraction
        vrange  : range in velocities for the x axis [km/s]
        zrange  : range in depth for the y axis [Mm]

    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    #plt.figure(figsize=(8, 8))
    plt.figure(figsize=(14, 14))    #### EDITED ####


    #------ Set ticks on the outside ------
    plt.rc('xtick', direction = 'out')
    plt.rc('ytick', direction = 'out')

    #------ Calculate tau ------
    tau = tau_integ(chi_in, height_in)
    tau_one = get_tau_one(tau,height_in)

    #------ Convert to more convenient units ------
    height = height_in/1.e6 # Mm
    vz = vz_in/1e3     # km/s
    vaxis = const.c.value/1e3 * (wave_ref - wave_in) / wave_ref

    if wave_ref > np.max(wave_in) or wave_ref < np.min(wave_in):
        raise ValueError("Reference wavelength not contained in input wave!")

    #------ Get indices range of height and velocity
    wi = np.where((vaxis >= vrange[0]) & (vaxis <= vrange[1]))[0][0]
    wf = np.where((vaxis >= vrange[0]) & (vaxis <= vrange[1]))[0][-1] + 1
    zi = np.where((height >= zrange[0]) & (height <= zrange[1]))[0][0]
    zf = np.where((height >= zrange[0]) & (height <= zrange[1]))[0][-1] + 1

    #------ Slice the arrays to improve the scaling and get only values in certain range ------
    tau = tau[wi:wf,zi:zf].T
    chi = chi_in[zi:zf, wi:wf]
    S = S_in[zi:zf, wi:wf]
    height = height[zi:zf]
    vz = vz[zi:zf]
    T = temp[zi:zf]
    wave = wave_in[wi:wf]
    wave_m = wave*1e-9
    bint = int_to_bt(sint[wi:wf], wave_m)/1e3  # intensity in brightness temp [kK]
    vaxis = vaxis[wi:wf]
    tau_one = tau_one[wi:wf]/1e6

    ###--------------------------
    ###------ Chi/Tau plot ------
    ###--------------------------
    plt.subplot(2,2,1, facecolor='k')
    plt.draw()

    plt.pcolormesh(vaxis, height, chi / tau, cmap=cmap,
                rasterized=True, shading='gouraud',
                vmax=np.percentile(chi[1:] / tau[1:], 99))

    #------ Plot tau = 1 curve ------
    plt.plot(vaxis, tau_one, linestyle[0], color=color[0])

    #------ Velocity curve and v=0 line ------
    plt.axvline(0, color='w') # Set by hand
    plt.plot(vz, height, linestyle[1], color=color[1])

    #------ Set x and y limits ------
    plt.xlim(vrange[1], vrange[0])
    plt.ylim(zrange[0], zrange[1])

    #------ remove ticklabels, set minor ticks -------
    plt.setp(plt.gca(),'xticklabels',[])

    # ------ Plot labels ------
    plt.ylabel('Height [Mm]')
    plt.text(0.03, 0.92, r'$\chi_\nu/\tau_\nu$', transform=plt.gca().transAxes, ha='left', va='center',color='w')

    ax = plt.gca()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(direction='out', which='both')
    plt.minorticks_on()

    ###----------------------------------
    ###------ Source function plot ------
    ###----------------------------------

    plt.subplot(2,2,2, facecolor='k')
    plt.pcolormesh(vaxis,height, S, cmap=cmap, rasterized=True,
            shading='gouraud', vmin=0, vmax=np.percentile(S, 99))

    #------ Plot tau = 1 curve ------
    plt.plot(vaxis, tau_one, linestyle[0], color=color[0])

    # velocity curve
    plt.plot(vz, height, linestyle[1], color=color[1])

    #------ Set x limits -------
    plt.xlim(vrange[1], vrange[0])

    #------ Remove tickmarks ------
    plt.setp(plt.gca(),'xticklabels',[])
    plt.setp(plt.gca(),'yticklabels',[])

    #------ Create second x top and y right axis -------
    ax = plt.gca()
    plt.minorticks_on()
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    #------ Get index of restwavelenght, 0 velocity
    ind   = np.argmin(np.abs(vaxis.data))
    Sline = int_to_bt(S[:,ind],wave_m[ind])  # source function at ind, in bri. temp.

    #------ Second x axis for Temperature and Source functions ------
    par1 = plt.gca().twiny()
    par1.plot(T/1e3, height, linestyle[2], color=color[2])
    par1.plot(Sline/1e3, height, linestyle[3], color=color[3])
    par1.set_xlim(sbtrange[0],sbtrange[1])
    par1.grid(color='w') # Seems like you have to use twice to set grid color ?
    par1.grid(color='w')

    par1.set_xlabel('T [kK]')
    plt.ylim(zrange[0], zrange[1])

    plt.text(0.03,0.92, r'S$_\nu$', transform=plt.gca().transAxes, ha='left', va='center',color='w')

    ###----------------------------------
    ###------ tau * exp(-tau) plot------
    ###----------------------------------

    plt.subplot(2,2,3, facecolor='k')
    plt.pcolormesh(vaxis,height, tau*np.exp(-tau), cmap=cmap, rasterized=True,
            shading='gouraud')

    plt.plot(vaxis,tau_one, linestyle[0], color=color[0])
    plt.plot(vz, height, linestyle[1], color=color[1])
    plt.xlim(vrange[1], vrange[0])
    plt.ylim(zrange[0], zrange[1])

    #------ Plot labels ------
    plt.ylabel('Height [Mm]')
    plt.xlabel('$\Delta$v [km/s]')
    plt.text(0.03,0.92, r'$\tau_\nu\exp(-\tau_\nu)$', transform=plt.gca().transAxes, ha='left', va='center',color='w')

    ax = plt.gca()
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.minorticks_on()

    ###----------------------------------
    ###------ Contribution function plot------
    ###----------------------------------
    plt.subplot(2,2,4, facecolor='k')
    cf = chi * np.exp(-tau) * S
    cf_norm = cf.copy()

    # normalise each column
    cf_norm = cf_norm / np.max(cf, axis=0)
    plt.pcolormesh(vaxis, height, cf_norm, cmap=cmap, vmax=np.max(cf_norm)*cscale,
                rasterized=True, shading='gouraud')

    # tau = 1 curve,  velocity curve
    plt.plot(vz, height, linestyle[1], color=color[1])
    plt.plot(vaxis, tau_one, linestyle[0], color=color[0])
    plt.ylim(zrange[0], zrange[1])
    # remove tickmarks
    plt.setp(plt.gca(),'yticklabels',[])
    plt.xlabel('$\Delta$v [km/s]')

    # Plot second y-axis for brightness Temperature
    par1 = plt.gca().twinx()
    par1.plot(vaxis,bint,linestyle[4], color=color[4])
    par1.set_ylim(irange[0],irange[1])
    par1.set_ylabel(r'I$_\nu$ [kK]')

    plt.xlim(vrange[1], vrange[0])
    ax.xaxis.set_ticks_position('both')

    plt.text(0.02,0.92, r'C${}_\mathsf{I}$',
            transform=plt.gca().transAxes, ha='left', va='center',color='w')
    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.08,hspace=0.08, left=0.1, bottom=0.10,
                        right=0.9, top=0.9)
    plt.tight_layout()


def tau_integ(chi_in, height, masked=True):
    ''' Integrates the opacity to get the optical depth tau '''
    chi = np.transpose(chi_in)
    tau = np.zeros(chi.shape)
    if masked:
        try:
            # if there is a masked array, sum only over used depth points
            mm = np.invert(chi[0].mask)
            if mm.sum() == 0:
                return np.ma.masked_all_like(tau)
            zcut = np.where(mm == 1)[0][0]
            tau[:,zcut+1:] = cumtrapz(chi[:,mm], x=-height[mm])
        except AttributeError:
            tau[:,1:] = cumtrapz(chi, x=-height)
    else:
        tau[:,1:] = cumtrapz(chi, x=-height)
    return tau

def get_tau_one(tau, height):
    ''' Finds the tau = 0 iso curve from a 2D array of tau, and the height
        array. Height should be the last index on the tau array. '''
    from scipy.interpolate import interp1d

    tau_one = np.zeros(tau.shape[0])
    for i in range(tau.shape[0]):
        f = interp1d(tau[i],height)
        tau_one[i] = f(1.0)
    return tau_one

def int_to_bt(inu, wave):
    """
    Converts radiation intensity to brightness temperature.
    Parameters
    ----------
    inu : `Quantity` object (number or sequence)
        Radiation intensity in units of energy per second per area
        per frequency per solid angle.
    wave: `Quantity` object (number or sequence)
        Wavelength in length units
    Returns
    -------
    brightness_temp : `Quantity` object (number or sequence)
        Brightness temperature in SI units of temperature.
    """
    from astropy.constants import c, h, k_B
    import astropy.units as u

    bt = h.value * c.value / (wave * k_B.value * np.log(2 * h.value * c.value / (wave**3 * inu) + 1))
    return bt

if __name__ == '__main__':

    import Jonas_lineform as lf
    import importlib
    importlib.reload(lf)

    path_Heps = '/mn/stornext/d18/RoCS/kiliankr/Hepsilon/cb24bih/Column271/H9merCRD_CaKCRD_ray'
    data = rh15d.Rh15dout(path_Heps)

    RHoutput = lf.Rh15d_calc(data)
    RHoutput.formation_diagram(0,0,wave_ref=397.0078, vrange=[-40,40], sbtrange=[2.8, 7.0], irange=[4.7, 5.1], zrange=[0.1, 2.0])
