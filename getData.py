#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:55:54 2018

@author: dhingratul
"""

from __future__ import print_function
import numpy as np
import json
import os
import cv2


def imbalance(x_train, x_test, y_train, y_test):
    """
    Removes train-test data imbalance by removing classes that are not present 
    in both training and testing sets
    Input: {x_train, x_test, y_train, y_test}
    Output: {x_train, x_test, y_train, y_test} class-balanced across training
    and testing set
    """
    y_c = list(set(np.unique(y_train)).difference(set(np.unique(y_test))))
    for i in range(len(y_c)):
        # Train
        mask = y_train != y_c[i]
        mask = mask.reshape(mask.shape[0],)
        y_train = y_train[mask]
        x_train = x_train[mask]
        # Test
        mask2 = y_test != y_c[i]
        mask2 = mask2.reshape(mask2.shape[0],)
        y_test = y_test[mask2]
        x_test = x_test[mask2]
    
    return x_train, x_test, y_train, y_test


def load_data(mdir, n_h, n_w): 
    """
    Helper function to load training and testing data in a numpy array format
    Input: {mdir, n_h, n_w}
    -- mdir = Directory where images/ and targets/ are stored, 
    -- n_h = Desired output height of the images,
    -- n_w = Desired output width of the images
    
    Output: {images, targets, images_o} 
    -- images = A numpy array with images in (num_samples, n_h, n_w) format
    -- targets = "Relative Temperature" values
    -- images_0 = Original images in a list format (for visualazation)
    """
    f_images = mdir +  'images/'
    f_targets = mdir + 'targets/'
    f_i = os.listdir(f_images)
    f_i.sort()
    f_t = os.listdir(f_targets)
    f_t.sort()  
    images, images_o = img_transform(f_images, f_i, n_h, n_w)
    targets = np.zeros((len(f_t), 1))
    for j in range(len(f_t)):
        data = json.load(open(f_targets + f_t[j]))
        targets[j] = data['relative_reading']
    return images, targets, images_o



def img_transform(f_images, f_i, n_h=225, n_w=25):
    """
    Helper function to transform images to get ROI 
    Note: Please run from within load_data, not standalone
    Input: {f_images, f_i, n_h=225, n_w=25}
    -- f_images = Directory to images/ folder, 
    -- f_i = Sorted list of files in the folder f_images
    -- n_h = Desired output height of the images,
    -- n_w = Desired output width of the images
    
    Output: {images, targets, images_o} 
    -- images = A numpy array with images in (num_samples, n_h, n_w) format
    -- images_0 = Original images in a list format (for visualazation)
    """
    images = np.zeros((len(f_i) , n_h, n_w))  #500, 145
    images_o = []
    for i in range(len(f_i)):
        img = cv2.imread(f_images + f_i[i], 1)
        images_o.append(img)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img2 = hsv_img[:,:,1]
        gray_image = cv2.convertScaleAbs(img2)
        blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        # Erode/Dilate
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts[1]) == 1:
            c = cnts[1][0]
            (x, y, w, h) = cv2.boundingRect(c)
            crop = thresh[0:y+h, x:x+w]
            crop2 = cv2.threshold(crop, 10, 255, cv2.THRESH_BINARY)[1]
            resized_img = cv2.resize(crop2, (n_w, n_h)) 
            images[i, :, :] = resized_img  
        else:
            for j in range(len(cnts[1])):	
                maxJ = 0
                maxVal = cnts[1][0].shape[0]
                if cnts[1][j].shape[0] > maxVal:
                    maxJ = j
                
            c = cnts[1][maxJ]
            (x, y, w, h) = cv2.boundingRect(c)
            crop = thresh[0:y+h, x:x+w]   
            crop2 = cv2.threshold(crop, 10, 255, cv2.THRESH_BINARY)[1]
            resized_img = cv2.resize(crop2, (n_w, n_h)) 
            images[i, :, :] = resized_img

    return images, images_o


def metric(model, x, y, test_cases, le):
    """
    Helper function to get +1/-1 metric which is defined as follows:
    A prediction is assumed correct if the prediction is within the range of 
    [-1, +1] around the ground truth
    example: Prediction of (46, 47, 48) is acceptable in case where ground truth is 47
    
    Input: {model, x, y, test_cases, le}
    -- model = keras model, 
    -- x = X matrix
    -- y = y matrix,
    -- test_cases = Number of samples from 0 index to produce metric on
    -- le = LabelEncoder() object used to transform y matrix
    
    Output: {gnd, pred, acc} 
    -- gnd = Label encoder transformed ground truth labels
    -- pred = Label encoder transformed predicted labels 
    -- acc * 100 = Percent accuracy on metric
    """
    y_pred = model.predict(x[0:test_cases,:])
    prediction = y_pred.argmax(axis=1)
    # Metric
    gnd = le.inverse_transform(np.argmax(y[0:test_cases], axis =1))
    pred = le.inverse_transform(prediction[0:test_cases])
    k = 1
    a = (gnd == pred) | (gnd == (pred - k)) | (gnd == (pred + k))
    acc = a.sum()/float(len(a))
    
    return gnd, pred, acc * 100
