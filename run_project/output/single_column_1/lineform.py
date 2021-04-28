###
### Set of tools to plot and calculate quantities relative to line formation
###
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import scipy.constants as const


def form_diag_rr(sel, xi, yi, wave_ref=None, cscale=0.2, vmax=59, tle=None,
                 zrange=[-0.1,2], newfig=True, colour=True, trange=[30, 2],
                 irange=[2.4, 6], pubgraph=False):
    """
    Calls plot_form_diag from Rh15dout object (here named 'sel').
    """
    # wave indices
    widx = sel.ray.wavelength_indices[:]
    # depth indices for temp, vz # this ONLY WORKS with non memmap arrays!!!!!!
    didx2 = sel.atmos.temperature[xi,yi] < 1.e30

    # protect against completely masked columns
    if not (float(np.sum(didx2)) > 0):
        vz   = np.zeros(sel.atmos.temperature.shape[-1])
        temp = vz.copy()
        height = vz.copy()
    else:
        vz = sel.atmos.velocity_z[xi,yi,didx2]
        temp = sel.atmos.temperature[xi,yi,didx2]
        #height = sel.atmos.height[xi,yi,didx2]       ### EDITED!! ###
        height = sel.atmos.height_scale[xi,yi,didx2]  ### EDITED!! ###


    if 'mask' in dir(sel.ray.intensity[xi,yi,0]):
        if sel.ray.intensity[xi,yi,0].mask:
            print(('Intensity at (%i,%i) is masked! Making empty plot.' % (xi, yi)))
            plot_form_diag_empty(height, vz, temp, vrange=[-vmax,vmax],
                                 cscale=cscale, zrange=zrange, tle=tle,
                                 newfig=newfig, sbtrange=trange, irange=irange,
                                 pubgraph=pubgraph)
            return

    intensity = sel.ray.intensity[xi,yi,widx]
    wave = sel.ray.wavelength[:][widx]

    """
                     ### EDITED below!! ###
    # have chi/source_function or full stuff?
    if hasattr(sel.ray, 'chi') and hasattr(sel.ray, 'source_function'):
        # depth indices for Chi, S
        didx1 = sel.ray.chi[xi,yi,:,0] > 0.
    elif hasattr(sel.ray, 'eta_line') and hasattr(sel.ray, 'chi_line'):
        # depth indices for Chi, S
        didx1 = sel.ray.chi_line[xi,yi,:,0] > 0.
    else:
        raise ValueError("(EEE) form_diag_rr: structure has no suitable "
                         "source function/chi.")
                     ### EDITED above!! ###
    """
    didx1 = np.where(sel.ray.chi[xi,yi,:,0] > 0.)[0]    ### EDITED!! ###
    S   = sel.ray.source_function[xi,yi,didx1,:]
    chi = sel.ray.chi[xi,yi,didx1,:]
    vz  = sel.atmos.velocity_z[xi,yi,didx1]
    #height = sel.atmos.height[xi,yi,didx1]       ### EDITED!! ###
    height = sel.atmos.height_scale[xi,yi,didx1]  ### EDITED!! ###
    temp = sel.atmos.temperature[xi,yi,didx1]
    if wave_ref is None:
        wave_ref = np.mean(sel.ray.wavelength[widx])
    if tle is None:
        tle = '[%i, %i]' % (xi, yi)

    plot_form_diag(chi, S, wave, height, vz, temp, intensity, tle=tle,
                   vrange=[-vmax,vmax],colour=colour, wave_ref=wave_ref,
                   cscale=cscale, zrange=zrange, newfig=newfig,
                   sbtrange=trange, irange=irange, pubgraph=pubgraph)
    return


def tau_integ(chi_in, height, masked=True):
    ''' Integrates the opacity to get the optical depth tau '''
    #chi = np.transpose(chi_in)     ### EDITED!! ###
    chi = chi_in.T                  ### EDITED!! ###
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
    tau_one = np.zeros(tau.shape[0])
    for i in range(tau.shape[0]):
        f = interp1d(tau[i],height)
        tau_one[i] = f(1.0)
    return tau_one


def get_tau_one_veryfast(tau, height):
    ''' Finds the tau = 0 iso curve from a 2D array of tau, and the height
        array. Height should be the last index on the tau array. '''
    return height[np.argmin(np.abs(tau-1.), axis=1)]


def get_tau_one_fast(tau, height):
    ''' Finds the tau = 1 iso curve from a 2D array of tau, and the height
        array. Height should be the last index on the tau array. '''
    idx = np.argmin(np.abs(tau-1.), axis=1)
    tau0 = tau[(np.arange(tau.shape[0]), idx)]
    tau1 = tau[(np.arange(tau.shape[0]), idx+1)]
    height0 = height[idx]
    height1 = height[idx+1]
    # manual linear inter/extrapolation
    theta = (1.-tau0) / (tau1-tau0)
    tau_one = height0 + theta * (height1 - height0)
    return tau_one

def get_tau_fast(tau, height, value=1.):
    ''' Finds the tau = value iso curve from a 2D array of tau, and the height
        array. Height should be the last index on the tau array. '''
    idx = np.argmin(np.abs(tau - value), axis=1)
    tau0 = tau[(np.arange(tau.shape[0]), idx)]
    tau1 = tau[(np.arange(tau.shape[0]), idx+1)]
    height0 = height[idx]
    height1 = height[idx+1]
    # manual linear inter/extrapolation
    theta = (value-tau0) / (tau1-tau0)
    tau_one = height0 + theta * (height1 - height0)
    return tau_one


def get_tau_one_rh(rr, height, verbose=False):
    """
    Calculates the tau = 1 depth from an rh15d instance that has
    chi saved (and not tau_one!)
    """
    from tt.io.shell import progressbar
    nwave = rr.ray.wavelength_indices.shape[0]
    nx, ny = rr.ray.intensity.shape[:2]
    tau_one = np.zeros((nx, ny, nwave), dtype='f')
    for i in range(nx):
        if verbose:
            progressbar(i + 1, nx)
        for j in range(ny):
            tau = tau_integ(rr.ray.chi[i, j], height, masked=False)
            tau_one[i, j] = get_tau_one_fast(tau, height)
    return tau_one


def plot_form_diag(chi_in, S_in, wave_in, height_in, vz_in, temp, sint, tle='',
                   vrange=[-59,59], zrange=[-0.2,2], wave_ref=279.552754614,
                   colour=False, cmap='gist_gray', cscale=0.2, sbtrange=[30,2],
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
    from helita.utils.utilsmath import int_to_bt
    import matplotlib.pyplot as plt
    import matplotlib.lines as mpllines
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
            #plt.figure(figsize=(6, 6))    ### EDITED!! ###
            plt.figure(figsize=(12, 12))   ### EDITED!! ###
        else:
            #plt.figure(figsize=(7.6, 7.423))   ### EDITED!! ###
            plt.figure(figsize=(12, 12))   ### EDITED!! ###
    # set ticks on the outside
    plt.rc('xtick', direction = 'out')
    plt.rc('ytick', direction = 'out')
    # Calculate tau
    tau = tau_integ(chi_in, height_in)
    tau_one = get_tau_one(tau,height_in)
    # Convert to more convenient units
    height = height_in/1.e6 # Mm
    #vz     = -vz_in/1e3     # km/s, opposite positive convention
    vz = vz_in/1e3     # km/s
    vaxis = 299792.45 * (wave_ref - wave_in) / wave_ref
    if wave_ref > np.max(wave_in) or wave_ref < np.min(wave_in):
        raise ValueError("Reference wavelength not contained in input wave!")
    wi = max(np.where(vaxis >= vrange[0])[0][0]  - 1, 0)
    wf = min(np.where(vaxis <= vrange[1])[0][-1] + 1, vaxis.shape[0]   - 1)
    zi = max(np.where(height <= zrange[1])[0][0]  - 1, 0)
    zf = min(np.where(height >= zrange[0])[0][-1] + 1, height.shape[0] - 1)
    # wave_idx = (vaxis  => vrange[0]) & (vaxis  <= vrange[1])
    # z_idx    = (height => zrange[0]) & (height <= zrange[1])
    # slice the arrays to improve the scaling
    tau     = tau[wi:wf+1,zi:zf+1].T
    #chi     = np.transpose(chi_in)[wi:wf+1,zi:zf+1]
    #S       = np.transpose(S_in)[wi:wf+1,zi:zf+1]
    chi     = chi_in[zi:zf+1, wi:wf+1]
    S       = S_in[zi:zf+1, wi:wf+1]
    height  = height[zi:zf+1]
    vz      = vz[zi:zf+1]
    T       = temp[zi:zf+1]
    wave    = wave_in[wi:wf+1]

    #### EDITED!! below #####
    # bint is calculated differently (as *.si does not work)
    #bint    = int_to_bt(sint[wi:wf+1], wave)/1e3  # intensity in brightness temp [kK]
    from astropy.constants import c, h, k_B
    import astropy.units as u
    wave_unit = 1 * u.Unit(wave.units)
    wave_si = 1 * u.m / wave_unit
    bint = h * c / (wave*wave_si * k_B * np.log(2 * h * c / ((wave*wave_si)**3 * sint[wi:wf+1] * u.rad**2) + 1))
    #### EDITED!! above #####

    vaxis   = vaxis[wi:wf+1]
    tau_one = tau_one[wi:wf+1]/1e6
    # colours:
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

    ###
    ### chi/tau plot
    ###
    #plt.subplot(2,2,1, axisbg='k')     ### EDITED!! ###
    plt.subplot(2,2,1)                  ### EDITED!! ###
    plt.draw()
    #pcolormesh(vaxis,height[1:], np.transpose(chi[:,1:]/tau[:,1:]), cmap=cmap,
    plt.pcolormesh(vaxis, height[1:], chi[1:] / tau[1:], cmap=cmap,
                   rasterized=True, shading='gouraud',
                   vmax=np.percentile(chi[1:] / tau[1:], 99))
    # tau = 1 curve
    plt.plot(vaxis, tau_one, ct[0], color=t1c)
    # velocity curve and v=0 line
    plt.axvline(0,color='w')
    plt.plot(vz, height, ct[1])
    #xlim(vrange[0], vrange[1])
    plt.xlim(vrange[1], vrange[0])
    plt.ylim(zrange[0], zrange[1])
    # remove ticklabels, set minor ticks
    plt.setp(plt.gca(),'xticklabels',[])
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(v_minortick))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(z_minortick))
    #ylabel('Height (Mm)')
    if pubgraph:
        plt.ylabel(r'\textbf{z (Mm)}', fontsize=14)
        plt.text(0.03, 0.92, r'$\chi_\nu/\tau_\nu$', fontsize=12,
             transform=plt.gca().transAxes, ha='left', va='center',color='w')
    else:
        plt.ylabel('z (Mm)')
        plt.text(0.03, 0.92, r'$\chi_\nu/\tau_\nu$', fontsize=16,
             transform=plt.gca().transAxes, ha='left', va='center',color='w')
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('none')
    plt.title(tle)

    ###
    ### source function plot
    ###
    #plt.subplot(2,2,2, axisbg='k')     ### EDITED!! ###
    plt.subplot(2,2,2)                  ### EDITED!! ###
    #pcolormesh(vaxis,height, np.transpose(S), cmap=cmap,
    print('Showing sqrt(S), remove!')
    plt.pcolormesh(vaxis,height, np.sqrt(S), cmap=cmap, rasterized=True,
               shading='gouraud', vmin=0, vmax=np.percentile(np.sqrt(S), 99))
    # tau = 1 curve
    plt.plot(vaxis,tau_one, ct[0], color=t1c)

    # velocity curve
    plt.plot(vz, height, ct[1])
    #xlim(vrange[0], vrange[1])
    plt.xlim(vrange[1], vrange[0])
    #axvline(0,color='w')
    # remove tickmarks
    plt.setp(plt.gca(),'xticklabels',[])
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(v_minortick))
    ax = plt.gca()
    plt.minorticks_on()
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_minor_locator(plt.MultipleLocator(z_minortick))
    a = [t.set_color('w') for t in ax.yaxis.get_ticklines()  +
         ax.yaxis.get_minorticklines()]
    # find wavelength point closest to maximum in tau --NOPE. Now using v=0
    #ind   = np.argmax(tau_one)
    ind   = np.argmin(np.abs(vaxis))

    #### EDITED!! below #####
    #Sline = int_to_bt(S[:,ind],wave[ind])  # source function at ind, in bri. temp.
    #print(S.units)
    Sline = h * c / (wave[ind]*wave_si * k_B * np.log(2 * h * c / ((wave[ind]*wave_si)**3 * S[:,ind] * u.rad**2) + 1))
    #print(Sline)
    """
    Sline units of?
    """
    #### EDITED!! above #####

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
    par1.xaxis.set_ticks_position('top')
    ax.xaxis.set_ticks_position('none')
    plt.ylim(zrange[0], zrange[1])
    plt.setp(plt.gca(),'yticklabels',[])
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
    #plt.subplot(2,2,3, axisbg='k')     ### EDITED!! ###
    plt.subplot(2,2,3)                  ### EDITED!! ###
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
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(v_minortick))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(z_minortick))
    ax = plt.gca()
    a = [t.set_color('w') for t in ax.xaxis.get_ticklines()[1::2] +
         ax.xaxis.get_minorticklines()[1::2]]
    ax.yaxis.set_ticks_position('left')

    ###
    ### contribution function plot
    ###
    #plt.subplot(2,2,4, axisbg='k')     ### EDITED!! ###
    plt.subplot(2,2,4)                  ### EDITED!! ###
    #cf = np.transpose(chi*np.exp(-tau)*S)
    cf = chi * np.exp(-tau) * S
    # normalise each column
    cf /= np.max(cf, axis=0)
    plt.pcolormesh(vaxis, height, cf, cmap=cmap, vmax=np.max(cf)*cscale,
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
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(z_minortick))
    # tau = 1 curve
    plt.plot(vaxis,tau_one, ct[0], color=t1c)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(plt.MultipleLocator(v_minortick))
    #a = [t.set_color('w') for t in ax.xaxis.get_ticklines()[1::2] +
    #     ax.xaxis.get_minorticklines()[1::2]]
    a = [t.set_color('w') for t in ax.yaxis.get_ticklines()  +
         ax.yaxis.get_minorticklines()]
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
    par1.xaxis.set_ticks_position('bottom')
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
        plt.subplots_adjust(wspace=0.,hspace=0., left=0.11, bottom=0.10,
                        right=0.89, top=0.9)
    # put ticks back on the inside
    plt.rc('xtick', direction = 'in')
    plt.rc('ytick', direction = 'in')
    return

#-----------------------------------------------------------------------------------------

def plot_form_diag_empty(height_in, vz_in, T, tle='', vrange=[-59,59], zrange=[-0.2,2],
                         cmap='gist_gray', cscale=0.2,
                         sbtrange=[30,2], newfig=True):
    ''' Same as plot_form_diag, but plots an empty diagram (for non-converged columns).'''

    from helita.utils.utilsmath import int_to_bt
    import matplotlib.lines as mpllines
    import matplotlib.pyplot as plt

    # for MgII, wave_ref (in the air, nm) can be obtained by:
    #
    # lambda = waveconv( 1e7 / (E_u - E_l) [cm-1] )
    #
    # For Mg II h, wave_ref = 280.27045685193
    # For Mg II k, wave_ref = 279.55275461489

    if newfig:
        plt.figure(figsize=(10,8.5))

    # Convert to more convenient units
    height = height_in.copy()/1.e6 # Mm
    vz     = -vz_in.copy()/1e3     # km/s, opposite positive convention

    ###
    ### chi/tau plot
    ###
    plt.subplot(2,2,1, axisbg='k')
    plt.draw()

    # velocity curve and v=0 line
    plt.axvline(0,color='w')
    plt.plot(vz, height, 'w-')

    plt.xlim(vrange[0], vrange[1])
    plt.ylim(zrange[0], zrange[1])

    # remove ticklabels, set minor ticks
    plt.setp(plt.gca(),'xticklabels',[])
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(2.5))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(.1))

    plt.ylabel('Height [Mm]')
    plt.text(0.03,0.92, r'$\chi_\nu/\tau_\nu$', fontsize=16,
         transform=plt.gca().transAxes, ha='left', va='center',color='w')

    plt.title(tle)

    ###
    ### source function plot
    ###
    plt.subplot(2,2,2, axisbg='k')

    # velocity curve
    plt.plot(vz, height, 'w-')
    plt.xlim(vrange[0], vrange[1])

    # remove tickmarks
    plt.setp(plt.gca(),'xticklabels',[])
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(2.5))

    # second x axis for source functions
    par1 = plt.gca().twiny()
    par1.plot(T/1e3, height, 'w:')
    par1.set_xlim(sbtrange[0],sbtrange[1])
    par1.set_xlabel('T [kK]', fontsize=14)
    plt.ylim(zrange[0], zrange[1])
    par1.xaxis.set_minor_locator(plt.MultipleLocator(.5))

    plt.setp(plt.gca(),'yticklabels',[])
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(.1))

    plt.text(0.03,0.92, r'$\mathrm{S}_\nu$', fontsize=16,
         transform = plt.gca().transAxes, ha='left', va='center',color='w')

    ###
    ### tau * exp(-tau) plot
    ###
    plt.subplot(2,2,3, axisbg='k')

    # velocity curve
    plt.plot(vz, height, 'w-')
    plt.xlim(vrange[0], vrange[1])
    plt.ylim(zrange[0], zrange[1])

    plt.ylabel('Height [Mm]')
    plt.xlabel('$\Delta$v [km/s]')
    plt.text(0.03,0.92, r'$\tau_\nu\exp(-\tau_\nu)$', fontsize=16,
         transform=plt.gca().transAxes, ha='left', va='center',color='w')

    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(2.5))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(.1))

    ###
    ### contribution function plot
    ###
    plt.subplot(2,2,4, axisbg='k')

    # velocity curve
    plt.plot(vz, height, 'w-')
    plt.ylim(zrange[0], zrange[1])

    # remove tickmarks
    plt.setp(plt.gca(),'yticklabels',[])

    plt.xlabel('$\Delta$v [km/s]')
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(.1))

    # second y axis for line profile plot
    par1 = plt.gca().twinx()
    par1.set_ylim(3,7)
    par1.set_ylabel(r'$I_\nu$ [kK]',fontsize=14)
    plt.xlim(vrange[0], vrange[1])
    par1.yaxis.set_minor_locator(plt.MultipleLocator(.25))

    plt.text(0.02,0.92, r'$\mathrm{C}_\mathrm{I}$', fontsize=16,
         transform=plt.gca().transAxes, ha='left', va='center',color='w')
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(2.5))

    # subplots
    if newfig:
        plt.subplots_adjust(wspace=0.03,hspace=0.03, left=0.07, bottom=0.07,
                            right=0.93, top=0.93)

    # put ticks back on the inside
    plt.rc('xtick', direction = 'in')
    plt.rc('ytick', direction = 'in')
    return
