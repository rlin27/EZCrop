# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:57:23 2020

@author: Rui LIN
"""
import torch
import math

def torch_fftshift(real, imag):
    '''
    Input:
        - real: a matrix of size [h, w], which is the real number part of the feature map slice in frequency domain.
        - imag: a matrix of size [h, w], which is the imaginary number part of the feature map slice in frequency domain.
            
    Output:
        - real: a matrix of size [h, w], which is the real number part of the feature map slice in frequency domain 
        after shift operation.
        - imag: a matrix of size [h, w], which is the imaginary number part of the feature map slice in frequency domain
        after shift operation.
    '''
    for dim in range(0, len(real.size())):
        real = torch.roll(real, dims=dim, shifts=real.size(dim)//2)
        imag = torch.roll(imag, dims=dim, shifts=imag.size(dim)//2)
    return real, imag

def StepDecision(h, w, alpha):
    '''
    Input:
        - h: a scalar, which is the height of the given feature map slice.
        - w: a scalar, which is the width of the given feature map slice.
        - alpha: a scalar between 0 and 1, which determines the size of selcted area.
    
    Output:
        - step: a scalar, which decides the selected area of the given feature map in frequency domain.
    '''
    if h % 2 == 0 and w % 2 == 0:
        xc = h / 2
        yc = w / 2
    else:
        xc = (h - 1) / 2
        yc = (w - 1) / 2
    max_h = h - (xc + 1)
    max_w = w - (yc + 1)
    if xc - 1 == 0 or yc - 1 == 0:
        step = 0
    else:
        step = min(int(math.ceil(max_h * alpha)),int(math.ceil(max_w * alpha)))
    return step

def EnergyRatio(fm_slice, alpha=1/4):
    '''
    Input:
        - fm_slice: a matrix of size [h, w], which is a slice of a given feature map in spatial domain.
    
    Output:
        - ratio: a scalar, which is the ratio of the energy of the unselected area of the feature map 
        and the total energy of the feature map (both in frequency domain).
    '''
    FFT_fm_slice = torch.rfft(fm_slice, signal_ndim=2, onesided=False)
    shift_real, shift_imag = torch_fftshift(FFT_fm_slice[:,:,0], FFT_fm_slice[:,:,1])
    FFTshift_fm_slice = (shift_real**2 + shift_imag**2)**(1/2)
    FFTshift_fm_slice = torch.log(FFTshift_fm_slice+1)
    h, w = FFTshift_fm_slice.shape
    step = StepDecision(h, w, alpha)
    if h % 2 == 0 and w % 2 == 0:
        xc = h / 2
        yc = w / 2
    else:
        xc = (h - 1) / 2
        yc = (w - 1) / 2
    E = sum(sum(FFTshift_fm_slice))
    select_FFTshift_fm_slice = FFTshift_fm_slice[int(xc-step):int(xc+step+1), int(yc-step):int(yc+step+1)]
    select_E = sum(sum(select_FFTshift_fm_slice))
    ratio = 1 - select_E / E
    if ratio != ratio:
        ratio = torch.zeros(1)
    return ratio