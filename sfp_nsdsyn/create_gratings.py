import numpy as np
import matplotlib.pyplot as plt


## copied from Broderick et al. (2023) 
## https://github.com/billbrod/spatial-frequency-preferences/blob/main/sfp/stimuli.py

def mkR(size, exponent=1, origin=None):
    '''make distance-from-origin (r) matrix

    Compute a matrix of dimension SIZE (a [Y X] list/tuple, or a scalar)
    containing samples of a radial ramp function, raised to power EXPONENT
    (default = 1), with given ORIGIN (default = (size+1)//2, (0, 0) = upper left).

    NOTE: the origin is not rounded to the nearest int
    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    xramp, yramp = np.meshgrid(np.arange(1, size[1]+1)-origin[1],
                               np.arange(1, size[0]+1)-origin[0])

    if exponent <= 0:
        # zero to a negative exponent raises:
        # ZeroDivisionError: 0.0 cannot be raised to a negative power
        r = xramp ** 2 + yramp ** 2
        res = np.power(r, exponent / 2.0, where=(r != 0))
    else:
        res = (xramp ** 2 + yramp ** 2) ** (exponent / 2.0)
    return res


def mkAngle(size, phase=0, origin=None):
    '''make polar angle matrix (in radians)

    Compute a matrix of dimension SIZE (a [Y X] list/tuple, or a scalar) containing
    samples of the polar angle (in radians, increasing counter-clockwise from the right
    horizontal meridian, ranging from -pi to pi), relative to angle PHASE (default = 0),
    about ORIGIN pixel (default = (size+1)/2).

    Note that setting phase effectively changes where angle=0 lies (e.g., setting
    angle=np.pi/2 puts angle=0 on the upper vertical meridian)

    NOTE: the origin is not rounded to the nearest int

    '''

    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    xramp, yramp = np.meshgrid(np.arange(1, size[1]+1)-origin[1],
                               np.arange(1, size[0]+1)-origin[0])
    # xramp and yramp are both in "Cartesian coordinates", so that xramp increases as
    # you go from left to right and yramp increases as you go from top to bottom
    xramp = np.array(xramp)
    # in order to get the proper angle array (0 at right horizontal meridian, increasing
    # counter-clockwise), yramp needs to increase from bottom to top.
    yramp = np.flip(np.array(yramp), 0)

    res = np.arctan2(yramp, xramp)

    # shift the phase but preserve the range
    res = ((res+(np.pi-phase)) % (2*np.pi)) - np.pi

    return res


def log_polar_grating(size, w_r=0, w_a=0, phi=0, ampl=1, origin=None, scale_factor=1):
    """Make a sinusoidal grating in logPolar space.

    this allows for the easy creation of stimuli whose spatial frequency decreases with
    eccentricity, as the peak spatial frequency of neurons in the early visual cortex does.

    Examples
    ============

    radial: `log_polar_grating(512, 4, 10)`

    angular: `log_polar_grating(512, 4, w_a=10)`

    spiral: `log_polar_grating(512, 4, 10, 10)`

    plaid: `log_polar_grating(512, 4, 10) + log_polar_grating(512, 4, w_a=10)`


    Parameters
    =============

    size: scalar. size of the image (only square images permitted).

    w_r: int, logRadial frequency.  Units are matched to those of the angular frequency (`w_a`).

    w_a: int, angular frequency.  Units are cycles per revolution around the origin.

    phi: int, phase (in radians).

    ampl: int, amplitude

    origin: 2-tuple of floats, the origin of the image, from which all distances will be measured
    and angles will be relative to. By default, the center of the image

    scale_factor: int or float. how to scale the distance from the origin before computing the
    grating. this is most often done for checking aliasing; e.g., set size_2 = 100*size_1 and
    scale_factor_2 = 100*scale_factor_1. then the two gratings will have the same pattern, just
    sampled differently
    """
    assert not hasattr(size, '__iter__'), "Only square images permitted, size must be a scalar!"
    rad = mkR(size, origin=origin)/scale_factor
    # if the origin is set such that it lies directly on a pixel, then one of the pixels will have
    # distance 0, that means we'll have a -inf out of np.log2 and thus a nan from the cosine. this
    # little hack avoids that issue.
    if 0 in rad:
        rad += 1e-12
    lrad = np.log2(rad**2)
    theta = mkAngle(size, origin=origin)

    # in the paper, we simplify this to np.cos(w_r * log(r) + w_a * theta +
    # phi), where log is the natural logarithm. They're equivalent
    return ampl * np.cos(((w_r * np.log(2))/2) * lrad + w_a * theta + phi)
    
def find_wr_wa_based_on_magnitude_and_angle(magnitudes, angle_deg=None):
    """
    Generate (w_r, w_a) pairs from magnitudes, placing each point on a circle.
    
    Parameters:
        magnitudes (array-like): List of magnitude values (radius)
        angles_deg (array-like or None): If provided, angles in degrees to place on each circle.
                                         If None, uses 45 degrees.
    
    Returns:
        w_r: np.array of radial components
        w_a: np.array of angular components
    """
    magnitudes = np.asarray(magnitudes)
    if angle_deg is None:
        angle_deg = np.full_like(magnitudes, 45)  # default 45°
    
    angle_rad = np.deg2rad(angle_deg)
    w_r = np.round(magnitudes * np.cos(angle_rad), 2)
    w_a = np.round(magnitudes * np.sin(angle_rad), 2)
    return w_r, w_a

def generate_wr_wa_grid(magnitudes=np.round(np.linspace(2,128,10)), 
                        angles=[0,30,60,-30,-60,-90]):
    """
    Generate a grid of (w_r, w_a) pairs based on specified magnitudes and angles.
    
    Parameters:
        magnitudes (array-like, optional): List of magnitude values. If None, uses np.linspace(2,128,10)
        angles (array-like, optional): List of angles in degrees. If None, uses [0,30,60,-30,-60,-90]
    
    Returns:
        tuple: (w_r, w_a) arrays containing the radial and angular components
    """
    magnitudes = np.round(magnitudes)
    w_r = []
    w_a = []
    for angle in angles:
        w_r_tmp, w_a_tmp = np.round(find_wr_wa_based_on_magnitude_and_angle(magnitudes, angle))
        w_r.append(w_r_tmp)
        w_a.append(w_a_tmp)
    
    return np.concatenate(w_r), np.concatenate(w_a)

def plot_wr_wa_grid(magnitudes, w_r, w_a, ax=None):
    # Create a figure
    if ax is None:
        ax = plt.gca()
    # Plot the circles
    for radius in magnitudes:
        half_circle = plt.matplotlib.patches.Arc((0, 0), 2*radius, 2*radius, 
                                            theta1=-90, theta2=90, 
                                            fill=False, color='gray', 
                                            linestyle='--', alpha=0.3)
        ax.add_patch(half_circle)  # Removed .gca() since ax is already an axes object

    ax.axis('scaled')  # Ensure equal aspect ratio
    # Plot the points
    ax.scatter(w_r, w_a, color='red', s=50, zorder=5, alpha=0.5)
    ax.set_xlabel('w_r')
    ax.set_ylabel('w_a')
    ax.set_xlim(-10 , max(magnitudes)+10)
    ax.set_ylim(-max(magnitudes)-10, max(magnitudes)+10)
    return ax

def check_all_stims(w_a, w_r, magnitudes, angles, scale_factor=1):
    fig, axes = plt.subplots(len(magnitudes), len(angles), 
                             figsize=(len(angles)*scale_factor,len(magnitudes)*scale_factor))
    axes = axes.flatten()
    for i, (tmp_w_r, tmp_w_a) in enumerate(zip(w_r, w_a)):
        im = log_polar_grating(256, tmp_w_r, tmp_w_a)
        axes[i].imshow(im, cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()

def create_scaled_gratings(w_r_list, w_a_list, phi_list, imsize=512, save_path=None):
    file = np.zeros((len(w_r_list),imsize, imsize))
    for i, (tmp_w_r, tmp_w_a, tmp_phi) in enumerate(zip(w_r_list, w_a_list, phi_list)):
        im = log_polar_grating(imsize, tmp_w_r, tmp_w_a, tmp_phi, ampl=1, origin=None, scale_factor=1)
        file[i] = im
    if save_path is not None:
        np.save(save_path, file)
    return file


def generate_sinusoidal_grating(image_size, cycles, theta=0, phase=0):
    """
    Generate a sinusoidal grating pattern.

    Parameters:
        image_size (int): Size of the image (e.g., 512x512).
        cycles (float): Spatial frequency (cycles per image).
        theta (float): Orientation in radians. This means that the variation of the grating is along the direction of theta.
        phase (float): Phase offset in radians.

    Returns:
        np.ndarray: 2D sinusoidal grating image.
    """
    frequency = cycles / image_size  # Convert cycles per image to cycles per pixel
    x = np.linspace(-image_size//2, image_size//2, image_size)
    y = np.linspace(-image_size//2, image_size//2, image_size)
    X, Y = np.meshgrid(x, y)
    # Rotate coordinates
    X_theta = X * np.cos(theta) - Y * np.sin(theta)
    # Generate sinusoidal wave
    grating = np.sin(2 * np.pi * frequency * X_theta + phase)

    return grating

import numpy as np
import matplotlib.pyplot as plt

def make_img(w_r, w_a, phi, imsize=512, r_max=4.2, epsilon=1e-6, return_r_theta=False):
    """
    Generate Img(r, theta) on a 2D grid.

    Parameters
    ----------
    w_r : float
        Weight for ln(r).
    w_a : float
        Additive constant (default) or multiplier for theta (if use_theta_term=True).
    phi : float
        Phase offset (radians).
    size : int, optional
        Output image size (size x size). Default 512.
    r_max : float, optional
        Visual field radius mapped to half-width of the image. Default 4.2.
    epsilon : float, optional
        Small value to avoid ln(0). Default 1e-6.
    use_theta_term : bool, optional
        If True, use w_a * theta instead of + w_a.

    Returns
    -------
    img : np.ndarray
        Generated image (size x size).
    r : np.ndarray
        Eccentricity grid.
    theta : np.ndarray
        Polar angle grid (radians).
    """
    # coordinate grid
    ax = np.linspace(-r_max, r_max, imsize)
    X, Y = np.meshgrid(ax, ax)
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    # avoid log(0) at the center
    r_safe = np.maximum(r, epsilon)
    # argument for cosine
    arg = w_r * np.log(r_safe) + w_a * theta + phi
    img = np.cos(arg)
    if return_r_theta:
        return img, r, theta
    else:
        return img

