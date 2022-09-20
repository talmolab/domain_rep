import os
import PIL
import torchvision
import math
import matplotlib
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import glob
import argparse
import seaborn as sns
import pandas as pd
from tqdm.notebook import tqdm
from scipy.signal import welch
from scipy.optimize import curve_fit

def plot_power_spectra(images, kvals, Abins):
    n_images = len(images)
    fig,axs = plt.subplots(nrows=len(images),ncols=2,facecolor='white',figsize=(10,20))
    if n_images == 1:
        p,a = curve_fit(power_law,kvals[0],Abins[0],p0=[2,1.3e6])[0]
        axs[0].imshow(images[0],cmap='gray')
        axs[0].set_axis_off()
        # axs[1].plot(kvals[0],Abins[0],label='Raw Power Spectrum')
        # axs[1].plot(kvals[0], power_law(kvals[0],p,a),ls='--',label=r'${}/f^{{{}}}$'.format(np.format_float_scientific(a,3),np.round(p,3)))
        axs[1].loglog(kvals[0],Abins[0],label='Raw Power Spectrum')
        axs[1].loglog(kvals[0], power_law(kvals[0],p,a),ls='--',label=r'${}/f^{{{}}}$'.format(np.format_float_scientific(a,3),np.round(p,3)))
        axs[1].set_xlabel("$log_{10} f$ (cycles/image)")
        axs[1].set_ylabel("$log_{10} p$")
        axs[1].yaxis.set_label_position("right")
        axs[1].yaxis.tick_right() 
        axs[1].legend()
    else:
        for row in range(0,n_images):
            p,a = curve_fit(power_law,kvals[row],Abins[row],p0=[2,1.3e6])[0]
            axs[row,0].imshow(images[row],cmap='gray')
            axs[row,0].set_axis_off()
            # axs[row,1].plot(kvals[row],Abins[row],label='Raw Power Spectrum')
            # axs[row,1].plot(kvals[row], power_law(kvals[row],p,a),ls='--',label=r'${}/f^{{{}}}$'.format(np.format_float_scientific(a,3),np.round(p,3)))
            axs[row,1].loglog(kvals[row],Abins[row],label='Raw Power Spectrum')
            axs[row,1].loglog(kvals[row], power_law(kvals[row],p,a),ls='--',label=r'${}/f^{{{}}}$'.format(np.format_float_scientific(a,3),np.round(p,3)))
            axs[row,1].set_xlabel("$log_{10} f$ (cycles/image)")
            axs[row,1].set_ylabel("$log_{10} p$")
            axs[row,1].yaxis.set_label_position("right")
            axs[row,1].yaxis.tick_right()
            axs[row,1].legend()
            fig.tight_layout()
    return fig
def power_spectrum(im_path,mode='jpeg',plot=True,transform=True,verbose = False):
    if type(im_path) == str or type(im_path) == np.str_:
        if mode == 'npy':
            image = PIL.Image.fromarray(np.load(im_path).squeeze())
        else:
            image = PIL.Image.open(im_path)
        if transform:
            image = torchvision.transforms.functional.center_crop(image,(256,256))
            image = np.array(torchvision.transforms.functional.rgb_to_grayscale(image))
        # print(image.shape)
        else:
            image = np.array(image)
    else:
        image = im_path
    if verbose:
        print(np.array(image).shape)
    
    npix = image.shape[0]

    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
    # Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    if plot:
        plot_power_spectra([image],[kvals],[Abins])
    return kvals,Abins
def power_law(x,p,a):
    return (a/x)**p
def power_fitting_lmfit(params,x,y):
    p = params['p']
    a = params['a']
    y_fit = (a/x)**p
    return y_fit-y
def r_squared(xdata, ydata, p, a):
    residuals = ydata - power_law(xdata, p, a) #np.log10(ydata) - np.log10(power_law(xdata, p, a))  
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2) #np.sum((np.log10(ydata)-np.mean(np.log10(ydata)))**2)   
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared
def fit_power_law(kvals,Abins):
    p,a = curve_fit(power_law,kvals,Abins,p0=[2,1.3e6])[0]
    r2 = r_squared(kvals,Abins,p,a)
    return p,r2

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_paths',nargs='+',action='store')
    parser.add_argument('-n','--n_samples',action='store',type=int,default=27750)
    parser.add_argument('-o','--output_path',default='power_spectrum.parquet')
    args = parser.parse_args()
    domain_p2 = {str(i):[] for i in range(len(args.dataset_paths))}
    domain_r2 ={str(i):[] for i in range(len(args.dataset_paths))}
    for i,dataset in enumerate(args.dataset_paths):
        n_samples = args.n_samples
        print(f'Gathering Images for dataset')
        imgs = glob.glob(dataset,recursive=True)
        if n_samples > len(imgs):
            print("Total Number of Samples is greater than size of dataset. Calculating power spectrum of entire dataset.")
            n_samples = len(imgs)
        print(f'Choosing {n_samples} samples from {len(imgs)} images')
        samples = np.random.choice(imgs,n_samples,replace=False)
        print(f'{len(samples)} Samples Chosen. Fitting Curves')
        for sample in tqdm(samples,desc=f'{i}'):
            ps = power_spectrum(sample,plot=False)
            try:
                p,r=fit_power_law(*ps)
            except RuntimeError as e:
                print(e)
                domain_p2[str(i)].append(np.nan)
                domain_r2[str(i)].append(np.nan)
                continue
            domain_p2[str(i)].append(p)
            domain_r2[str(i)].append(r)
        print(len(domain_p2[str(i)]))
    domain_p2=pd.DataFrame(domain_p2)
    domain_p2.to_parquet(args.output_path)
    