from helita.sim import rh15d
import numpy as np
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from astropy import constants as const
from scipy import interpolate


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

    bt = h * c / (wave * k_B * np.log(2 * h * c / (wave**3 * inu * u.rad**2) + 1))
    return bt

def plot_form_diag(chi_in, S_in, wave_in, height_in, vz_in, temp, sint, tle='',
                   vrange=[-59,59], zrange=[-0.11,2], wave_ref=279.552754614,
                   colour=True, cmap='gist_gray', cscale=0.2, sbtrange=[30,2],
                   newfig=True, irange=[2.4, 6], pubgraph=False):
    ''' Plots, for the formation diagram, in four panels:

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

    '''
    import matplotlib.pyplot as plt
    import matplotlib.lines as mplline
    from scipy.interpolate import interp1d
    # for MgII, wave_ref (in the air, nm) can be obtained by:
    #
    # lambda = waveconv( 1e7 / (E_u - E_l) [cm-1] )
    #
    # For Mg II h, wave_ref = 280.27045685193
    # For Mg II k, wave_ref = 279.55275461489
    if newfig:
        if pubgraph:
            plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
            plt.rc('text', usetex=True)
            plt.rc('font',size=12)
            plt.rc('legend', fontsize=7)
            plt.rc('axes', linewidth=0.5)  # for axes thickness
            plt.figure(figsize=(6, 6))
        else:
            plt.figure(figsize=(7.6, 7.423))

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
    wi = max(np.where(vaxis >= vrange[0])[0][0]  - 1, 0)
    wf = min(np.where(vaxis <= vrange[1])[0][-1] + 1, vaxis.shape[0]   - 1)
    zi = max(np.where(height <= zrange[1])[0][0]  - 1, 0)
    zf = min(np.where(height >= zrange[0])[0][-1] + 1, height.shape[0] - 1)

    #------ Slice the arrays to improve the scaling and get only values in certain range ------
    tau = tau[wi:wf+1,zi:zf+1].T
    chi = chi_in[zi:zf+1, wi:wf+1]
    S = S_in[zi:zf+1, wi:wf+1]
    height = height[zi:zf+1]
    vz = vz[zi:zf+1]
    T = temp[zi:zf+1]
    wave = wave_in[wi:wf+1]
    wave_m = wave*1e-9
    bint = int_to_bt(sint[wi:wf+1], wave_m)/1e3  # intensity in brightness temp [kK]
    vaxis = vaxis[wi:wf+1]
    tau_one = tau_one[wi:wf+1]/1e6

    #------ Colours ------
    if colour:
        # tau=1, vel, planck, S
        #ct = ['c-', 'r-', 'w:', 'y--' ]
        ct = ['-', 'r-', 'w:', 'w--' ]
        #tau = 1 colour
        t1c = '#00CDFF'
    else:
        ct = ['0.5', 'w-', 'w:', 'w--']
    z_minortick = 0.1
    v_minortick = 5.

    ###--------------------------
    ###------ Chi/Tau plot ------
    ###--------------------------
    plt.subplot(2,2,1, facecolor='k')
    plt.draw()
    #pcolormesh(vaxis,height[1:], np.transpose(chi[:,1:]/tau[:,1:]), cmap=cmap,
    plt.pcolormesh(vaxis, height, chi / tau, cmap=cmap,
                   rasterized=True, shading='gouraud',
                   vmax=np.percentile(chi[1:] / tau[1:], 99))

    #------ Plot tau = 1 curve ------
    plt.plot(vaxis, tau_one, ct[0], color=t1c)

    #------ Velocity curve and v=0 line ------
    plt.axvline(0, color='w')
    plt.plot(vz, height, ct[1])

    #------ Set x and y limits ------
    plt.xlim(vrange[1], vrange[0])
    plt.ylim(zrange[0], zrange[1])

    #------ remove ticklabels, set minor ticks -------
    plt.setp(plt.gca(),'xticklabels',[])
    #plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(v_minortick))
    #plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(z_minortick))

    #ylabel('Height (Mm)')
    if pubgraph:
        plt.ylabel(r'\textbf{z (Mm)}', fontsize=14)
        plt.text(0.03, 0.92, r'$\chi_\nu/\tau_\nu$', fontsize=12,
             transform=plt.gca().transAxes, ha='left', va='center',color='w')
    else:
        plt.ylabel('Height [Mm]')
        plt.text(0.03, 0.92, r'$\chi_\nu/\tau_\nu$',
             transform=plt.gca().transAxes, ha='left', va='center',color='w')

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
    plt.plot(vaxis,tau_one, ct[0], color=t1c)

    # velocity curve
    plt.plot(vz, height, ct[1])

    #------ Set x limits -------
    plt.xlim(vrange[1], vrange[0])
    #axvline(0,color='w')

    #------ Remove tickmarks ------
    plt.setp(plt.gca(),'xticklabels',[])
    plt.setp(plt.gca(),'yticklabels',[])
    #plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(v_minortick))

    #------ Create second x top and y right axis -------
    ax = plt.gca()
    plt.minorticks_on()
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    # find wavelength point closest to maximum in tau --NOPE. Now using v=0
    #ind   = np.argmax(tau_one)
    ind   = np.argmin(np.abs(vaxis.data))
    Sline = int_to_bt(S[:,ind],wave_m[ind])  # source function at ind, in bri. temp.

    # second x axis for source functions
    par1 = plt.gca().twiny()
    #par1.plot(T/1e3, height, ct[2])
    par1.plot(T/1e3, height, ct[2])
    par1.plot(Sline/1e3, height, ct[3])#, dashes=(10, 9.2))
    par1.set_xlim(sbtrange[0],sbtrange[1])

    if pubgraph:
        par1.set_xlabel(r'\textbf{T (kK)}', fontsize=14)
    else:
        par1.set_xlabel('T (kK)', fontsize=14)

    par1.xaxis.set_minor_locator(plt.MultipleLocator(1.))
    plt.ylim(zrange[0], zrange[1])
    #text(0.03,0.92, r'$\mathrm{S}_\nu$', fontsize=16,
    if pubgraph:
        plt.text(0.03,0.92, r'S$_\nu$', fontsize=14,
             transform=plt.gca().transAxes, ha='left', va='center',color='w')
    else:
        plt.text(0.03,0.92, r'S$_\nu$', fontsize=16,
             transform=plt.gca().transAxes, ha='left', va='center',color='w')

    ###
    ### tau * exp(-tau) plot
    ###
    plt.subplot(2,2,3, facecolor='k')
    #pcolormesh(vaxis,height, np.transpose(tau*np.exp(-tau)), cmap=cmap,
    plt.pcolormesh(vaxis,height, tau*np.exp(-tau), cmap=cmap, rasterized=True,
               shading='gouraud')
    # tau = 1 curve
    plt.plot(vaxis,tau_one, ct[0], color=t1c)
    # velocity curve
    plt.plot(vz, height, ct[1])
    #xlim(vrange[0], vrange[1])
    plt.xlim(vrange[1], vrange[0])
    plt.ylim(zrange[0], zrange[1])
    #axvline(0,color='w')
    #ylabel('Height [Mm]')
    if pubgraph:
        plt.ylabel(r'\textbf{z (Mm)}', fontsize=14)
        plt.xlabel(r'\textbf{$\Delta$v (km~s$^{-1}$)}', fontsize=14)
        plt.text(0.03,0.92, r'$\tau_\nu\:$'+'exp('+r'$-\tau_\nu$)', fontsize=14,
            transform=plt.gca().transAxes, ha='left', va='center',color='w')
    else:
        plt.ylabel('z (Mm)')
        plt.xlabel('$\Delta$v (km/s)')
        plt.text(0.03,0.92, r'$\tau_\nu\exp(-\tau_\nu)$', fontsize=16,
             transform=plt.gca().transAxes, ha='left', va='center',color='w')

    ax = plt.gca()
    #a = [t.set_color('w') for t in ax.xaxis.get_ticklines()[1::2] +
    #     ax.xaxis.get_minorticklines()[1::2]]
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.minorticks_on()

    ###
    ### contribution function plot
    ###
    plt.subplot(2,2,4, facecolor='k')
    #cf = np.transpose(chi*np.exp(-tau)*S)
    cf = chi * np.exp(-tau) * S
    cf_norm = cf.copy()
    # normalise each column
    cf_norm = cf_norm / np.max(cf, axis=0)
    plt.pcolormesh(vaxis, height, cf_norm, cmap=cmap, vmax=np.max(cf_norm)*cscale,
                   rasterized=True, shading='gouraud')
    # velocity curve
    plt.plot(vz, height, ct[1])
    plt.ylim(zrange[0], zrange[1])
    #axvline(0,color='w')
    # remove tickmarks
    plt.setp(plt.gca(),'yticklabels',[])
    if pubgraph:
        plt.xlabel(r'\textbf{$\Delta$v (km~s$^{-1}$)}', fontsize=14)
    else:
        plt.xlabel('$\Delta$v (km/s)')
    # tau = 1 curve
    plt.plot(vaxis,tau_one, ct[0], color=t1c)
    ax = plt.gca()
    #a = [t.set_color('w') for t in ax.xaxis.get_ticklines()[1::2] +
    #     ax.xaxis.get_minorticklines()[1::2]]

    #plot(vaxis,tau_one, 'ro')
    # second y axis for line profile plot
    par1 = plt.gca().twinx()
    par1.plot(vaxis,bint,'w-',lw=1.5)
    par1.set_ylim(irange[0],irange[1])  # was [3,7]
    #par1.set_ylabel(r'$I_\nu$ [kK]',fontsize=14)
    if pubgraph:
        par1.set_ylabel(r'\textbf{I$_\nu$ (kK)}', fontsize=14)
    else:
        par1.set_ylabel(r'I$_\nu$ (kK)',fontsize=14)
    #xlim(vrange[0], vrange[1])
    plt.xlim(vrange[1], vrange[0])
    #par1.yaxis.set_minor_locator(MultipleLocator(.25))
    ax.xaxis.set_ticks_position('both')
    #text(0.02,0.92, r'$\mathrm{C}_\mathrm{I}$', fontsize=16,
    if pubgraph:
        plt.text(0.02,0.92, r'C$_I$', fontsize=12,
             transform=plt.gca().transAxes, ha='left', va='center',color='w')
    else:
        plt.text(0.02,0.92, r'C${}_\mathsf{I}$', fontsize=16,
             transform=plt.gca().transAxes, ha='left', va='center',color='w')
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(v_minortick))
    # subplots
    if newfig:
        plt.subplots_adjust(wspace=0.08,hspace=0.08, left=0.1, bottom=0.10,
                        right=0.9, top=0.9)
    # put ticks back on the inside
    return

if __name__ == '__main__':

    from helita.sim import rh15d
    import Jonas_lineform as sc
    import matplotlib.pyplot as plt
    plt.ion()
    #import importlib #This is only to reload module, because I changed a few things
    #importlib.reload(sc)

    #Path to some data
    path_data = '/mn/stornext/u3/kiliankr/rh/rh15d/run/Projects/H_epsilon/output/FalC_Omega_ray'
    # Read rh data
    data_t = rh15d.Rh15dout(path_data)

    # Extract the different quantities from the rh object you need for calling plot_form_diag
    vz_in=data_t.atmos.velocity_z[0,0,:].dropna('height').data
    wave_ref = 1/(107440.4508 - 82259.1580)*1e7 # Reference wavelenght of spectral line given in nm
    temp = data_t.atmos.temperature[0,0,:].dropna('height').data
    sint = data_t.ray.intensity[0,0,data_t.ray.wavelength_indices[0:-1]] # I drop the last index of wavelenght because it stores 500 Angstrom wvl in my particular dataset.
    chi_in = data_t.ray.chi[0,0,:,0:-1].dropna('height').data
    height_in = data_t.atmos.height_scale[0,0,:].dropna('height').data
    wave_in = data_t.ray.wavelength[data_t.ray.wavelength_indices[0:-1]].data
    S_in = data_t.ray.source_function[0,0,:,0:-1] # I did not add here .data because it seems there is a problem with the int_to_bt function, just dont add .data then you are fine

    # Call plot_form_diag with the quantities from above.
    # I particular tested it on the H epsilon line, so you have to change a few keyword for your line and play around with limits:
    # vrange: Sets the minimum and maximum of plotted doppler velocity, with wave_ref as restwavelength
    # zrange: Sets the minimum and maximum atmospheric height, which you want to plot
    # Color: Set a few option how to plot different line in the subplot, you probably have to change this by hand which colors and linestyles you prefer
    # cmap: just gives the colormap
    # cscale: The Contribution function are normalized to its maximum value per wavelength, this factor just multiplicates the maximum scaling of the contribution function plot. You have to find whats is best for you.
    # sbtrange: Sets the temperature range of the second subplot (Source function), second x-axis at the top of the plot
    # irange: Sets the intensity range in kK of the last subplot (Contrinution function CI), for ploting the lineprofile
    sc.plot_form_diag(chi_in, S_in, wave_in, height_in, vz_in, temp, sint, tle='',
                   vrange=[-40,40], zrange=[0.0, 1,8], wave_ref=wave_ref,
                   colour=True, cmap='gist_gray', cscale=1.0, sbtrange=[3,8],
                   newfig=True, irange=[4.5, 6], pubgraph=False)
    #Pubgraph=True just set a different fontsize and so, I did not try it if it work, generally you can just change the fonsizes yourselves as you like it
    # newfig=True just stars a new plotting figure
