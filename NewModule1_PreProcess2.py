import preprocess2 as pp2
from preprocess3 import PyCNN
import easygui as eg
import cv2
import adaptive_median as amf
import average_filter as af
import BilateralFilter as bf
import NLMFilter as NLMf
import wiener
import gaussianFilter
import findParams as fp
import matplotlib.pyplot as plt
import numpy as np
import os

algos = []
psnrs_amf = []
psnrs_af = []
psnrs_weiner = []
psnrs_gaussian = []
psnrs_bf = []
psnrs_nlm = []
psnrs_hist = []
psnrs_medblur = []
directory = eg.diropenbox();

arr = os.listdir(directory)

for count in range(0,len(arr)) :
    try :
        file_name = arr[count]
        file = directory + "\\" + file_name
        
        amf.adaptiveMF(file, 'images/output_amf.png')
        
        p = fp.findParams(file, 'images/output_amf.png', 'AMF')
        algos.append('AMF')
        psnrs_amf.append(p)
        
        af.avgFilter(file, 'images/output_af.png')
        p = fp.findParams(file, 'images/output_af.png', 'Avg Filter')
        algos.append('Avg Filter')
        psnrs_af.append(p)
        
        wiener.applyAdaptiveWiener(file, 'images/output_wiener.png')
        p = fp.findParams(file, 'images/output_wiener.png', 'Wiener')
        algos.append('Wiener')
        psnrs_weiner.append(p)
        
        gaussianFilter.applyGaussianFilter(file, 'images/output_gaussian.png')
        p = fp.findParams(file, 'images/output_gaussian.png', 'Gaussian')
        algos.append('Gaussian')
        psnrs_gaussian.append(p)
        
        bf.BilateralFilter(file, 'images/output_bf.png')
        p = fp.findParams(file, 'images/output_bf.png', 'Bilateral Filter')
        algos.append('Bilateral Filter')
        psnrs_bf.append(p)
        
        NLMf.NLMFilter(file, 'images/output_NLMf.png')
        p = fp.findParams(file, 'images/output_NLMf.png', 'NLM Filter')
        algos.append('NLM Filter')
        psnrs_nlm.append(p)
        
        img = cv2.imread(file,0)
        equ = cv2.equalizeHist(img)
        res = np.hstack((img,equ)) #stacking images side-by-side
        cv2.imwrite('images/hist_eq.png',res)
        p = fp.findParams(file, 'images/hist_eq.png', 'Histogram Equalization')
        algos.append('Hist Eq.')
        psnrs_hist.append(p)
        
        gray_orig = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        gray = cv2.bitwise_not(gray_orig)
        proc5 = pp2.median_blur(gray)
        cv2.imwrite("images/median_blur.png", proc5)
        p = fp.findParams(file, 'images/median_blur.png', 'Median Blur Detection')
        algos.append('Blur')
        psnrs_medblur.append(p)        
    except :
        print('Not processed:' + arr[count])
        
psnrs = []
algos = ['AMF', 'AF', 'Weiner', 'Gaussian', 'BF', 'NLM', 'Hist', 'Med Blur']
psnrs.append(np.mean(psnrs_amf))
psnrs.append(np.mean(psnrs_af))
psnrs.append(np.mean(psnrs_weiner))
psnrs.append(np.mean(psnrs_gaussian))
psnrs.append(np.mean(psnrs_bf))
psnrs.append(np.mean(psnrs_nlm))
psnrs.append(np.mean(psnrs_hist))
psnrs.append(np.mean(psnrs_medblur))

print(psnrs)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(algos,psnrs)
plt.show()
