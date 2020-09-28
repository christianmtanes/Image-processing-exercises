import numpy as np
from scipy.misc import imread as imread, imsave as imsave
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from numpy.linalg import inv

RGB2YIQ_MATRIX = np.matrix('0.299 0.587 0.114; 0.569 -0.275 -0.321; 0.212 -0.523 0.311')
GRAY_SCALE_IMAGE = 2
MAX_GRAY_LEVEL = 255
def read_image(filename, representaion):
    im = imread(filename, mode = 'RGB').astype(np.float64)
    if(representaion == 1):
        return rgb2gray(im)/MAX_GRAY_LEVEL
    return im/MAX_GRAY_LEVEL

def imdisplay(filename, representaion):
    plt.figure()
    if(representaion == 1):
            plt.imshow(read_image(filename, representaion), cmap='gray')
    else:
            plt.imshow(read_image(filename, representaion))
    plt.axis('off')
    plt.show()

"""
this function help transform an image according to a transform matrix.
"""
def imtransform(image, transform_matrix):
    im_new = image.copy()
    for i in range(3):
        im_new[:,:,i] = transform_matrix[i,0]*image[:,:,0] + transform_matrix[i,1]*image[:,:,1] + transform_matrix[i,2]*image[:,:,2]
    return im_new

def rgb2yiq(imRGB):
    return imtransform(imRGB, RGB2YIQ_MATRIX)

    
def yiq2rgb(imYIQ):
    return imtransform(imYIQ, inv(RGB2YIQ_MATRIX))

def histogram_equalize(im_orig):
    if len(im_orig.shape) == GRAY_SCALE_IMAGE:
        hist_orig, bins = np.histogram(im_orig,MAX_GRAY_LEVEL + 1)
        imageCumSum = np.cumsum(hist_orig)
        NormalizedImageCumSum = imageCumSum/im_orig.size
        MultipliedNormalizedImageCumSum = NormalizedImageCumSum*MAX_GRAY_LEVEL
        im_normalized = MultipliedNormalizedImageCumSum[(im_orig * MAX_GRAY_LEVEL).astype(np.uint8)]
        im_eq = np.round((im_normalized - im_normalized.min()) * MAX_GRAY_LEVEL/(im_normalized.max() -im_normalized.min()))
        hist_eq = np.histogram(im_eq,MAX_GRAY_LEVEL + 1)[0]
        return [im_eq/MAX_GRAY_LEVEL, hist_orig, hist_eq]
    else:
        yiqImage = rgb2yiq(im_orig)
        yiqImage[:,:,0],hist_orig, hist_eq = histogram_equalize(yiqImage[:,:,0])
        im_eq = np.clip(yiq2rgb(yiqImage),0,1)
        return [im_eq, hist_orig, hist_eq]

def quantize(im_orig, n_quant, n_iter):
    if len(im_orig.shape) == GRAY_SCALE_IMAGE:
        hist_orig, bins = np.histogram(im_orig,MAX_GRAY_LEVEL + 1)
        imageCumSum = np.cumsum(hist_orig)
        pixelsPerSection = im_orig.size//n_quant
        z = [np.where(imageCumSum >= pixelsPerSection*k)[0][0] for k in range(n_quant+1) ]
        z[0] = 0
        z[n_quant] = MAX_GRAY_LEVEL
        q = [0]*n_quant
        error = []
        grayLevels = np.arange(MAX_GRAY_LEVEL + 1) 
        for j in range(n_iter):
            prevZ = z.copy()
            for i in range(n_quant):
                q[i] = np.dot(hist_orig[z[i]:z[i+1]+1], grayLevels[z[i]:z[i+1]+1])//np.sum(hist_orig[z[i]:z[i+1]+1])
            for i in range(1,n_quant):
                z[i] = (q[i-1] + q[i])//2
            if prevZ == z :
                break
            err = 0
            for i in range(n_quant):
                square_diff = np.power(q[i] - grayLevels[z[i]: z[i+1] + 1],2)
                err += np.dot(square_diff, hist_orig[z[i]:z[i+1]+1])
            error.append(err)
        im_orig = im_orig*MAX_GRAY_LEVEL
        im_quant = im_orig.copy()
        for i in range(n_quant):
            lower_bound = z[i] <= im_orig
            upper_bound = z[i+1] + 1 > im_orig
            mask = np.logical_and(lower_bound, upper_bound)
            im_quant[mask] = q[i]/MAX_GRAY_LEVEL
        return im_quant , error
    else:
        yiqImage = rgb2yiq(im_orig)
        yiqImage[:,:,0], error = quantize(yiqImage[:,:,0], n_quant, n_iter)
        im_quant = yiq2rgb(yiqImage)
        im_quant = (im_quant - im_quant.min())/(im_quant.max() -im_quant.min()) #normalize to [0,1]
        return im_quant, error



