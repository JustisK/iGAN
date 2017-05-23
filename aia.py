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

def get_latent_vector(img, gen_model):
    """Return the latent vector z associated to img, with the appropriate Theano 
    model."""
    [h, w] = img.size
    invert_models = iGAN_predict.def_invert_models(gen_model, layer='conv4', alpha=0.002)
    npx = gen_model.npx
    img = img.resize((npx, npx))
    img = np.array(img)
    img_pre = img[np.newaxis, :, :, :]
    _, _, z  = iGAN_predict.invert_images_CNN_opt(invert_models, img_pre, solver="cnn_opt")
    return z

def lerp(img0, img1, p, model=None, gen_model=None, model_name="handbag_64"):
    if  not model:
        model_class = locate('model_def.dcgan_theano')
        model_file = './models/'+model_name+'.dcgan_theano'
        model = model_class.Model(
            model_name=model_name, model_file=model_file)
    if not gen_model:
        model_class = locate('model_def.dcgan_theano')
        model_file = './models/'+model_name+'.dcgan_theano'
        gen_model = model_class.Model(
            model_name=model_name, model_file=model_file, use_predict=True)
    arrays = [p*z0+(1-p)*z1]
    z = np.stack(arrays)
    zmb = floatX(z[0 : 64, :])
    xmb = model._gen(zmb)
    samples = [xmb]
    samples = np.concatenate(samples, axis=0)
    samples = model.inverse_transform(samples, npx=model.npx, nc=model.nc)
    samples = (samples * 255).astype(np.uint8)
    return samples[0]

def interpolate(img0, img1, model_name="handbag_64", x0=-0.5, x1=1.5, delta=1/32.0):
    model_class = locate('model_def.dcgan_theano')
    model_file = './models/'+model_name+'.dcgan_theano'
    model = model_class.Model(
        model_name=model_name, model_file=model_file)
    gen_model = model_class.Model(
        model_name=model_name, model_file=model_file, use_predict=True)
    z0 = get_latent_vector(img0, gen_model=gen_model).reshape((100,))
    z1 = get_latent_vector(img1, gen_model=gen_model).reshape((100,))
    ps = np.arange(x0, x1-0.000001, delta)
    n = ps.size
    arrays = [p*z0+(1-p)*z1 for p in ps]
    z = np.stack(arrays)
    zmb = floatX(z[0 : 64, :])
    xmb = model._gen(zmb)
    samples = [xmb]
    samples = np.concatenate(samples, axis=0)
    samples = model.inverse_transform(samples, npx=model.npx, nc=model.nc)
    samples = (samples * 255).astype(np.uint8)
    m = math.ceil(math.sqrt(n))
    img_vis = utils.grid_vis(samples, m, m)
    return img_vis
    # img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

