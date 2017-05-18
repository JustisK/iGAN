from __future__ import print_function
import argparse, iGAN_predict
from pydoc import locate
from lib import utils
from lib.rng import np_rng
import cv2
import numpy as np
from lib.theano_utils import floatX
import requests
from PIL import Image
from StringIO import StringIO

def parse_args():
    parser = argparse.ArgumentParser(description='generated random samples (dcgan_theano)')
    parser.add_argument('--model_name', dest='model_name', help='the model name', default='outdoor_64', type=str)
    parser.add_argument('--model_type', dest='model_type', help='the generative models and its deep learning framework', default='dcgan_theano', type=str)
    parser.add_argument('--framework', dest='framework', help='deep learning framework', default='theano')
    parser.add_argument('--model_file', dest='model_file', help='the file that stores the generative model', type=str, default=None)
    parser.add_argument('--output_image', dest='output_image', help='the name of output image', type=str, default=None)

    args = parser.parse_args()
    return args

def interpolate(url0, url1, output_image):
    model_class = locate('model_def.dcgan_theano')
    model_file = './models/handbag_64.dcgan_theano'
    model = model_class.Model(
        model_name="handbag_64", model_file=model_file)
    # save images
    for j, url in enumerate([url0, url1]):
        r = requests.get(url)
        i = Image.open(StringIO(r.content))
        i.save("pics/url"+str(j)+".jpg")
    z0 = iGAN_predict.find_latent(url=url0).reshape((100,))
    z1 = iGAN_predict.find_latent(url=url1).reshape((100,))
    delta = 1.0/32.0
    arrays = [p*z0+(1-p)*z1 for p in np.arange(-16*delta, 1+16*delta-0.0001, delta)]
    z = np.stack(arrays)
    print(z.shape)
    zmb = floatX(z[0 : 64, :])
    xmb = model._gen(zmb)
    samples = [xmb]
    samples = np.concatenate(samples, axis=0)
    print(samples.shape)
    samples = model.inverse_transform(samples, npx=model.npx, nc=model.nc)
    samples = (samples * 255).astype(np.uint8)
    # generate grid visualization
    im_vis = utils.grid_vis(samples, 8, 8)
    # write to the disk
    im_vis = cv2.cvtColor(im_vis, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_image, im_vis)

if __name__ == '__main__':
    args = parse_args()
    if not args.model_file:  #if model directory is not specified
        args.model_file = './models/%s.%s' % (args.model_name, args.model_type)

    if not args.output_image:
        args.output_image = '%s_%s_samples.png' % (args.model_name, args.model_type)

    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))

    # initialize model and constrained optimization problem
    model_class = locate('model_def.%s' % args.model_type)
    model = model_class.Model(model_name=args.model_name, model_file=args.model_file)
    # generate samples

        #def gen_samples(self, z0=None, n=32, batch_size=32, use_transform=True):
    samples = []
    n = 32
    batch_size = 32
    z0 = np_rng.uniform(-1., 1., size=(n, model.nz))
    n_batches = int(np.ceil(n/float(batch_size)))
    for i in range(n_batches):
        zmb = floatX(z0[batch_size * i:min(n, batch_size * (i + 1)), :])
        xmb = model._gen(zmb)
        samples.append(xmb)
    samples = np.concatenate(samples, axis=0)
    samples = model.inverse_transform(samples, npx=model.npx, nc=model.nc)
    samples = (samples * 255).astype(np.uint8)
    #samples = model.gen_samples(z0=None, n=196, batch_size=49, use_transform=True)
    # generate grid visualization
    im_vis = utils.grid_vis(samples, 14, 14)
    # write to the disk
    im_vis = cv2.cvtColor(im_vis, cv2.COLOR_BGR2RGB)
    cv2.imwrite(args.output_image, im_vis)
    print('samples_shape', samples.shape)
    print('save image to %s' % args.output_image)


