import torch, gin
import math
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

@gin.configurable
def remove_outliers(points, n_devs=3, already_centered=False, take_biggest_std=False, center=None):
    """
    Removes points from the point cloud that are beyond three standard deviations from the mean.

    Parameters:
    points (ndarray): A numpy array of shape (n, d) where n is the number of points and d is the dimensionality.

    Returns:
    ndarray: A numpy array containing the filtered points.
    """
    if take_biggest_std:
        assert already_centered
    if not already_centered:
        if center is None:
            mean = torch.mean(points, axis=0)
            std_dev = torch.std(points, axis=0)
        else:
            mean = center
            sigma2  = torch.mean((points - center)**2, axis=0) #3,
            std_dev = torch.sqrt(sigma2)
    else: #already centered (0,0,0)
        mean = torch.tensor([0, 0, 0], device=points.device)
        sigma2  = torch.mean(points**2, axis=0) #3,
        std_dev = torch.sqrt(sigma2)
    if take_biggest_std:
        std_dev = torch.max(std_dev).repeat(3)
    lower_bound = mean - n_devs * std_dev
    upper_bound = mean + n_devs * std_dev

    # Apply the filter
    filtered_mask = torch.all((points >= lower_bound) & (points <= upper_bound), axis=1)
    filtered_points = points[filtered_mask]
    return filtered_points, filtered_mask

@gin.configurable
class MinMaxScaler:
    def __init__(self, feature_range=(0, 1), preserve_ratio=True, already_centered=True, already_scaled=True):
        self.min = None
        self.max = None
        self.scale_ = None
        self.min_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.feature_range = feature_range
        self.preserve_ratio = preserve_ratio
        self.already_centered = already_centered
        self.already_scaled = already_scaled
        if self.already_scaled:
            assert self.already_centered
        assert self.preserve_ratio
        


    def fit_transform(self, X):
        if not self.already_centered and not self.already_scaled:
            self.data_min_ = torch.min(X, dim=0)[0]
            self.data_max_ = torch.max(X, dim=0)[0]
            self.data_range_ = self.data_max_ - self.data_min_

            self.min, self.max = self.feature_range
            self.center = (self.min + self.max) / 2
            self.scale_ = (self.max - self.min) / self.data_range_
            if self.preserve_ratio:
                self.scale_ = torch.min(self.scale_)
            self.min_ = self.min - self.data_min_ * self.scale_

            scaled_X = X*self.scale_ 
            scaled_X_mid = (scaled_X.min(dim=0)[0] + scaled_X.max(dim=0)[0]) / 2
            self.trans_ = self.center - scaled_X_mid #translate the mean to the center
        else: #already centered [-1,1] -> [0,1]
            assert self.feature_range == (0, 1)
            self.center = torch.tensor([0.5, 0.5, 0.5], device=X.device)
            self.trans_ = torch.tensor([0.5, 0.5, 0.5], device=X.device)
            if not self.already_scaled:
                self.scale_ = 0.5/torch.abs(X).max()
            else:
                # we only need to scale [-1,1] to [-0.5,0.5]
                self.scale_ = torch.tensor(0.5, device=X.device)
            scaled_X = X*self.scale_ #Now [-0.5, 0.5]^3

        return scaled_X + self.trans_
    
    def transform(self, X):
        return X*self.scale_ + self.trans_

    def inverse_transform(self, X_scaled_translated):
        X_scaled = X_scaled_translated - self.trans_
        return X_scaled / self.scale_