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

    def get_latent_vector(self, img):
        """Return the latent vector z associated to img."""
        [h, w] = img.size
        npx = self.gen_model.npx
        img = img.resize((npx, npx))
        img = np.array(img)
        img_pre = img[np.newaxis, :, :, :]
        _, _, z  = iGAN_predict.invert_images_CNN_opt(
            self.invert_models, img_pre, solver="cnn_opt")
        return z

    def interpolate(self, img0, img1, x0=-0.5, x1=1.5, delta=1/20.0):
        """Return a visualization of an interpolation between img0 and img1,
        starting with parameter x0 and going to x1, in increments of
        delta.  Note that img0 corresponds to parameter x0=0 and img1
        to parameter x1=1.  The default is to start outside that
        range, and so we do some extrapolation.
        """
        z0 = self.get_latent_vector(img0).reshape((100,))
        z1 = self.get_latent_vector(img1).reshape((100,))
        ps = np.arange(x0, x1-0.000001, delta)
        n = ps.size
        arrays = [(1-p)*z0+p*z1 for p in ps]
        z = np.stack(arrays)
        zmb = floatX(z[0 : n, :])
        xmb = self.model._gen(zmb)
        samples = [xmb]
        samples = np.concatenate(samples, axis=0)
        samples = self.model.inverse_transform(
            samples, npx=self.model.npx, nc=self.model.nc)
        samples = (samples * 255).astype(np.uint8)
        m = math.ceil(math.sqrt(n))
        img_vis = utils.grid_vis(samples, m, m)
        return img_vis
    # img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

# def lerp(img0, img1, p, model=None, gen_model=None, model_name="handbag_64"):
#     if  not model:
#         model_class = locate('model_def.dcgan_theano')
#         model_file = './models/'+model_name+'.dcgan_theano'
#         model = model_class.Model(
#             model_name=model_name, model_file=model_file)
#     if not gen_model:
#         model_class = locate('model_def.dcgan_theano')
#         model_file = './models/'+model_name+'.dcgan_theano'
#         gen_model = model_class.Model(
#             model_name=model_name, model_file=model_file, use_predict=True)
#     arrays = [p*z0+(1-p)*z1]
#     z = np.stack(arrays)
#     zmb = floatX(z[0 : 64, :])
#     xmb = model._gen(zmb)
#     samples = [xmb]
#     samples = np.concatenate(samples, axis=0)
#     samples = model.inverse_transform(samples, npx=model.npx, nc=model.nc)
#     samples = (samples * 255).astype(np.uint8)
#     return samples[0]
