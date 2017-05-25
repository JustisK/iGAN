import cv2
import iGAN_predict
import math
import numpy as np
import requests

from lib import utils
from lib.theano_utils import floatX, sharedX
from matplotlib.pyplot import imshow
from PIL import Image
from pydoc import locate
from StringIO import StringIO

def get_image(url):
    """Take a URL and return the image at that URL.  A typical source will be a
    handbag or shoe image from Amazon."""
    r = requests.get(url)
    return Image.open(StringIO(r.content))

def display_image(img):
    imshow(np.asarray(img))

def lerp(z0, z1, p):
    """Return the vector linearly interpolating between z0 and z1, with
    parameter p: p=0 corresponds to z0 and p=1 to z1.

    """
    return (1-p)*z0+p*z1

def serp(z0, z1, p):
    r0 = np.linalg.norm(z0)
    n0 = z0/r0
    r1 = np.linalg.norm(z1)
    n1 = z1/r1
    theta = math.acos(np.inner(n0, n1)) # angle between z0 and z1
    perp = n1-np.inner(n0, n1)*n0 # vector perp to n0 and in the n0, n1 plane
    nperp = perp/np.linalg.norm(perp) # n0 and nperp are orthonormal
                                      # and span the n0, n1 plane
    angle = p*theta
    n = math.cos(angle)*n0+math.sin(angle)*nperp # spherically interpolated unit vector
    # next interpolate the radii
    r = (1-p)*r0 + p*r1
    return r*n

class Network(object):

    def __init__(self, model_name):
        self.model_name = model_name
        self.model_class = locate('model_def.dcgan_theano')
        model_file = './models/'+model_name+'.dcgan_theano'
        self.model = self.model_class.Model(
            model_name=model_name, model_file=model_file)
        self.gen_model = self.model_class.Model(
            model_name=model_name, model_file=model_file, use_predict=True)
        self.invert_models = iGAN_predict.def_invert_models(
            self.gen_model, layer='conv4', alpha=0.002)

    def get_lv(self, img):
        """Return the latent vector z associated to img."""
        [h, w] = img.size
        npx = self.gen_model.npx
        img = img.resize((npx, npx))
        img = np.array(img)
        img_pre = img[np.newaxis, :, :, :]
        _, _, z  = iGAN_predict.invert_images_CNN_opt(
            self.invert_models, img_pre, solver="cnn_opt")
        return z

    def interpolate_full(self, img0, img1, interp=serp, x0=-0.5, x1=1.5, delta=1/20.0):
        """Return a visualization of an interpolation between img0 and img1,
        using interpolation method interp.  The interpolation starts
        with parameter x0 and goes to x1, in increments of delta.
        Note that img0 corresponds to parameter x0=0 and img1 to
        parameter x1=1.  The default is to start outside that range,
        and so we do some extrapolation.

        """
        z0 = self.get_lv(img0).reshape((100,))
        z1 = self.get_lv(img1).reshape((100,))
        ps = np.arange(x0, x1-0.000001, delta)
        n = ps.size
        arrays = [lerp(z0, z1, p) for p in ps]
        z = np.stack(arrays); print z.shape
        zmb = floatX(z[0 : n, :]); print zmb.shape
        xmb = self.model._gen(zmb); print xmb.shape
        samples = [xmb]
        samples = np.concatenate(samples, axis=0)
        samples = self.model.inverse_transform(
            samples, npx=self.model.npx, nc=self.model.nc)
        samples = (samples * 255).astype(np.uint8)
        m = math.ceil(math.sqrt(n))
        img_vis = utils.grid_vis(samples, m, m)
        return img_vis
    # img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

    def imagify(self, z):
        """Return an image corresponding to the latent vector z."""
        z = np.stack([z.reshape((100,))])
        zmb = floatX(z[0 : 1, :]);
        xmb = self.model._gen(zmb);
        samples = np.concatenate([xmb], axis=0)
        samples = self.model.inverse_transform(
            samples, npx=self.model.npx, nc=self.model.nc)
        samples = (samples * 255).astype(np.uint8)
        img_vis = utils.grid_vis(samples, 1, 1)
        return img_vis

    def analogize(self, z0, z1, z2):
        "Return the vector z3 so that z0 : z1 as z2 : z3"
        return z1-z0+z2
