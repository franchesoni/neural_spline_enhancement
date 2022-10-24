
# An Image-to-video Model for Real-Time Video Enhancement

## Summary
They approximate a 3D LUT by 3 2D LUTs similartly than in [zeng] and use the method for video.

## Dataset
They use the dataset from [zeng]

## Results
SSIM 0.929 (vs. 0.922 [zeng]) (480p)

# Learning Image-adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time

## Summary
They jointly learn basis 3DLUTs and their combination weights in a paired or unpaired manner. They also standardize the dataset processing.

## Dataset
[zeng] itself

## Results
SSIM 0.922


# 4D LUT: Learnable Context-Aware 4D Lookup Table for Image Enhancement

## Summary
They do the same as in [zeng] but 1) the basis LUTs are not learned and 2) they add context information as another channel 

## Dataset
MIT-Adobe-5K-UPE 
MIT-Adobe-5K-DPE 

## Results
MIT-Adobe-5K-UPE 
SSIM 0.924 (vs. 0.911 [zeng])

MIT-Adobe-5K-DPE 
SSIM 0.918 (vs. 0.910 [zeng])

# Real-time Image Enhancer via Learnable Spatial-aware 3D Lookup Tables

## Summary
They achieve good performance and fast speed by replicating [zeng] but using T basic sets of M LUTs each. Note that [zeng] is the case with M=1. The T basic sets are combined using T image specifc weights, and the M LUTs in the resulting set are combined with spatial-specific weights. 

## Dataset
customized [zeng]

## Results
SSIM 0.8904 (vs. 0.8864 [zeng]) (480p)

# AdaInt: Learning Adaptive Intervals for 3D Lookup Tables on Real-Time Image Enhancement

## Summary
They do as [zeng] but also estimate the lattice in which the resulting 3DLUT lives, all trained via gradient descent

## Dataset
customized [zeng] (different splits)

## Results
SSIM 0.926 (vs. 0.923 [zeng]) (480p)

# MAXIM: Multi-Axis MLP for Image Processing

## Summary
New image processing architecture achieving state-of-the-art results over different benchmarks

## Dataset
FiveK as in [Towards unsupervised deep image enhancement with generative adversarial network] 

## Results
SSIM 0.945 (vs. 0.929 UEGAN, 0.922 DPE, 0.861 Exposure)


# Learning Tone Curves for Local Image Enhancement

## Summary
Similar to [Personalized] but applying curves locally and without using splines (just trivial interpolation)

## Dataset
FiveK as in [Towards unsupervised deep image enhancement with generative adversarial network] 

## Results
SSIM 0.913 (vs. 0.875 UPE)

# Distilling Style from Image Pairs for Global Forward and Inverse Tone Mapping

## Summary
They use normalizing flows to get a "style" vector for each image.
The results are not relevant because comparisons are not fair.

# Neural Color Operators for Sequential Image Retouching

## Summary
They learn and estimate the strenght of 3 color operations to be applied sequentially

## Dataset
500p version of [Hu exposure]

## Results
SSIM 0.907 (vs. 0.874 [zeng])

# StarEnhancer: Learning Real-Time and Style-Aware Image Enhancement 

## Summary
They stimate knots of cubic interpolators for an input space (r, g, b, x, y) that considers the position.

## Dataset
MIT-Adobe-5KUPE

## Results
SSIM 0.948 (vs. 0.934 [zeng])

# CURL: Neural Curve Layers for Global Image Enhancement

## Summary
They have an encoder-decoder architecture followed by knot-parameterized piecewise linear operations in different color spaces

## Dataset
MIT-Adobe5k-DPE

## Results
Suboptimal

# CLUT-Net: Learning Adaptively Compressed Representations of 3DLUTs for Lightweight Image Enhancement
Max can you read this?
## Summary
Like [zeng] but with compressed 3D LUTs

## Dataset
[zeng] like, without much details

## Results
0.927 (vs. 0.912 [zeng] and 0.922 [zeng] with 20 LUTs)

# Learning Diverse Tone Styles for Image Retouching

## Summary
Normalizing flows for style encoding and conditional retouch net for retouching

## Dataset
Not much details, similar to StarEnhancer

## Results
SSIM 0.944 (vs. 0.948 StarEnhancer and 0.934 [zeng])

# Controllable Image Enhancement
## Summary

## Dataset
reportedly [zeng] 

## Results
SSIM 0.924 (vs. 0.886 [zeng])

# Personalized Image Enhancement Using Neural Spline Color Transforms
## Summary
Estimate knots of per channel splines

## Dataset
Custom

# inccesible
Performance comparison of image enhancers with and without deep learning