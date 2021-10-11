from PIL import Image, ImageTk
from keras.preprocessing import image
import os, sys, cv2, glob
import re
import numpy as np
import keras
from draw_pupil import draw_pupil
# from matplotlib import pyplot as plt
from keras import backend as K
import tensorflow as tf

def predict_eye(img_path, model, fd):
     # with session.as_default():
     #     with session.graph.as_default():
    print("Entering the function ...")
    img = image.load_img(img_path)
    input_img = image.img_to_array(img)
    print('Without Array')
    print(input_img)
    print("After Conversion to INT")
    input_img = input_img.astype('uint8')
    print(input_img)
    print(input_img.shape)
    # with session.as_default():
    #     with session.graph.as_default():
    face, lms = fd.detect_face(input_img) # assuming there is only one face in input image
    assert len(face) >= 1, "No face detected"

    #detecting eyes
    left_eye_xy = np.array([lms[6], lms[1]])
    right_eye_xy = np.array([lms[5], lms[0]])

    dist_eyes = np.linalg.norm(left_eye_xy - right_eye_xy)
    eye_bbox_w = (dist_eyes / 1.25)
    eye_bbox_h = (eye_bbox_w *0.6)

    left_eye_im = input_img[
        int(left_eye_xy[0]-eye_bbox_h//2):int(left_eye_xy[0]+eye_bbox_h//2),
        int(left_eye_xy[1]-eye_bbox_w//2):int(left_eye_xy[1]+eye_bbox_w//2), :]
    #left_eye_im = left_eye_im[:,::-1,:] # No need for flipping left eye for iris detection
    right_eye_im = input_img[
        int(right_eye_xy[0]-eye_bbox_h//2):int(right_eye_xy[0]+eye_bbox_h//2),
        int(right_eye_xy[1]-eye_bbox_w//2):int(right_eye_xy[1]+eye_bbox_w//2), :]

    draw = input_img.copy()
    for i, lm in enumerate([left_eye_xy, right_eye_xy]):
        draw = cv2.circle(draw, (int(lm[1]), int(lm[0])), 10, (255*i,255*(1-i),0), -1)

    inp_left = cv2.cvtColor(left_eye_im, cv2.COLOR_RGB2GRAY)
    print(inp_left)
    inp_left = cv2.equalizeHist(inp_left)
    # inp_left = cv2.resize(inp_left, (inp_left.shape[1],inp_left.shape[0]))[np.newaxis, ..., np.newaxis]
    inp_left = cv2.resize(inp_left, (360,216))[np.newaxis, ..., np.newaxis]
    print(inp_left.shape)

    inp_right = cv2.cvtColor(right_eye_im, cv2.COLOR_RGB2GRAY)
    inp_right = cv2.equalizeHist(inp_right)
    # inp_right = cv2.resize(inp_right, (inp_right.shape[1],inp_right.shape[0]))[np.newaxis, ..., np.newaxis]
    inp_right = cv2.resize(inp_right, (360,216))[np.newaxis, ..., np.newaxis]
    print(inp_right.shape)

    input_array = np.concatenate([inp_left, inp_right], axis=0)
    pred_left, pred_right = model.net.predict(input_array/255 * 2 - 1)

    plt.figure(figsize=(15,4))
    plt.subplot(1,3,1)
    plt.title("Left eye")
    lms_left = model._calculate_landmarks(pred_left)
    result_left = draw_pupil(left_eye_im, inp_left, lms_left)
    plt.axis('off')
    plt.imshow(result_left)
    plt.subplot(1,3,2)
    plt.title("Right eye")
    lms_right = model._calculate_landmarks(pred_right)
    # print('lms_right',lms_right.shape)
    # print('inp_right',inp_right.shape)
    # print('right_eye_im',right_eye_im.shape)
    result_right = draw_pupil(right_eye_im, inp_right, lms_right)
    plt.imshow(result_right)
    draw2 = input_img.copy()

    slice_h = slice(int(left_eye_xy[0]-eye_bbox_h//2), int(left_eye_xy[0]+eye_bbox_h//2))
    slice_w = slice(int(left_eye_xy[1]-eye_bbox_w//2), int(left_eye_xy[1]+eye_bbox_w//2))
    im_shape = left_eye_im.shape[::-1]

    draw2[slice_h, slice_w, :] = cv2.resize(result_left, im_shape[1:])

    slice_h = slice(int(right_eye_xy[0]-eye_bbox_h//2), int(right_eye_xy[0]+eye_bbox_h//2))
    slice_w = slice(int(right_eye_xy[1]-eye_bbox_w//2), int(right_eye_xy[1]+eye_bbox_w//2))
    im_shape = right_eye_im.shape[::-1]

    draw2[slice_h, slice_w, :] = cv2.resize(result_right, im_shape[1:])
    print(draw2.shape)
    plt.savefig('./Result/finaloutput_py_app.png')
    return draw2
