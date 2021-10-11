import warnings
warnings.filterwarnings('ignore')
import os, sys, cv2, glob
import re
import numpy as np
# from matplotlib import pyplot as plt
from keras import backend as K
import tensorflow as tf
import pandas as pd
# tf.reset_default_graph()
from face_detector import MTCNNFaceDetector
from model import KerasELG
from PIL import Image, ImageTk
import keras
from draw_pupil import draw_pupil
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify, json
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)

keras.backend.set_session(session)

global model
global fd

config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

mtcnn_weights_dir = "./detection_weights/"
# tf.reset_default_graph()
fd = MTCNNFaceDetector(sess=K.get_session(), model_path=mtcnn_weights_dir)

print('Face Detector Model loaded. Start serving...')

# session = tf.Session(config=config)
# keras.backend.set_session(session)


model = KerasELG()
model.net.load_weights("weights.h5")

print('IRIS Model loaded. Start serving...')

# def draw_pupil(im, inp_im, lms):
#     draw = im.copy()
# #     print(im.shape)
# #     print(inp_im.shape)
#     draw = cv2.resize(draw, (inp_im.shape[2], inp_im.shape[1]))
#     pupil_center = np.zeros((2,))
#     pnts_outerline = []
#     pnts_innerline = []
#     stroke = inp_im.shape[1] // 20 + 1
#     for i, lm in enumerate(np.squeeze(lms)):
#         #print(lm)
#         y, x = int(lm[0]*3), int(lm[1]*3)
#
#         if i < 8:
#             #draw = cv2.circle(draw, (y, x), stroke, (125,255,125), 1)
#             pnts_outerline.append([y, x])
#         elif i < 16:
#             #draw = cv2.circle(draw, (y, x), stroke, (125,125,255), 1)
#             pnts_innerline.append([y, x])
#             pupil_center += (y,x)
#         elif i < 17:
#             pass
#             #draw = cv2.drawMarker(draw, (y, x), (255,200,200), markerType=cv2.MARKER_CROSS, markerSize=1, thickness=1, line_type=cv2.LINE_AA)
#         else:
#             pass
#             #draw = cv2.drawMarker(draw, (y, x), (255,125,125), markerType=cv2.MARKER_CROSS, markerSize=1, thickness=1, line_type=cv2.LINE_AA)
#     pupil_center = (pupil_center/8).astype(np.int32)
# #     print(np.array(pnts_outerline).shape)
# #     print(np.array(pnts_innerline).shape)
#     draw = cv2.cv2.circle(draw, (pupil_center[0], pupil_center[1]), stroke, (255,0,0), 1)
#     draw = cv2.polylines(draw, [np.array(pnts_outerline).reshape(-1,1,2)], isClosed=True, color=(125,0,125), thickness=1)
#     draw = cv2.polylines(draw, [np.array(pnts_innerline).reshape(-1,1,2)], isClosed=True, color=(0,0,255), thickness=1)
#     return draw, pnts_outerline, pnts_innerline, pupil_center

def predict_strabismus(right_eye_im, left_eye_im, inLineRight, inLineLeft):
    new_img = np.hstack((right_eye_im, left_eye_im))
    rEyeWid = right_eye_im.shape[1]
    rEyeHei = right_eye_im.shape[0]
    lEyeWid = left_eye_im.shape[1]
    lEyeHei = left_eye_im.shape[0]

    itxr = inLineRight[0][0]  * (rEyeWid / 180)
    ityr = inLineRight[2][1]  * (rEyeHei / 108)

    hr = inLineRight[6][1]* (rEyeHei / 108) - inLineRight[2][1]* (rEyeHei / 108)
    wr = inLineRight[4][0]* (rEyeWid / 180) - inLineRight[0][0]* (rEyeWid / 180)

    itxl = inLineLeft[0][0]*  (lEyeWid / 180) + rEyeWid
    ityl = inLineLeft[2][1]*  (lEyeHei / 108)

    hl = inLineLeft[6][1]*  (lEyeHei / 108) - inLineLeft[2][1]*  (lEyeHei / 108)
    wl = inLineLeft[4][0]*  (lEyeWid / 180) - inLineLeft[0][0]*  (lEyeWid / 180)

    # itx	ity	iw	ih
    d = {'itx': [itxr, itxl], 'ity': [ityr, ityl], 'iw' : [wr , wl], 'ih' : [hr, hl]}

    df = pd.DataFrame(data=d)
    df = df.astype(int)
    print(df)

    gray_level_thr = 150    ## for detection of corneal reflection  # give option to play
    min_refl_area = 5      ## minimum size of the corneal reflection # give option to play
    max_refl_area = 70     ## maximum size of the corneal reflection # give option to play
    refl_ar_diff = 5       ## max diff in area of corneal reflections in two eyes # option to play
    hr_const = 222.3        ## const. val for strabismus calculation  ## No change
    xbuffer = 0.1           ## horz. boundary for exlcuding corneal reflections # maychange if want so give an option
    ybuffer = 0.3

    ####---- obtain the iris and pupil center and radius for both eyes ---
    rirad = [(df["iw"][0]/2), (df["ih"][0]/2)]
    lirad = [(df["iw"][1]/2), (df["ih"][1]/2)]
    rictr = [df["itx"][0] + rirad[0], df["ity"][0] + rirad[1]]
    lictr = [df["itx"][1] + lirad[0], df["ity"][1] + lirad[1]]

    rgbimg = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB) ## for display
    grayimg = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2GRAY) ## for thresholding


    ####---- obtain ROIs for left and right eyes from the location data ---
    roi_re_gray = grayimg[df["ity"][0]:df["ity"][0]+df["ih"][0], df["itx"][0]:df["itx"][0]+df["iw"][0]]
    roi_le_gray = grayimg[df["ity"][1]:df["ity"][1]+df["ih"][1], df["itx"][1]:df["itx"][1]+df["iw"][1]]

    ####---- perform binary thresholding to obtain corneal reflection regions ---
    thrval, rethrimg = cv2.threshold(roi_re_gray, gray_level_thr, 255, cv2.THRESH_BINARY)
    thrval, lethrimg = cv2.threshold(roi_le_gray, gray_level_thr, 255, cv2.THRESH_BINARY)

    ####---- connected components to obtain region properties
    reccout = cv2.connectedComponentsWithStats(rethrimg, 8, cv2.CV_32S)
    leccout = cv2.connectedComponentsWithStats(lethrimg, 8, cv2.CV_32S)

    ####---- eliminate thresholded reflections that are at periphery of the ROIs ---
    ####---- generate list of indices that are close to the center for both eyes
    reccoutindx = []
    buffx = xbuffer*df["iw"][0]
    buffy = ybuffer*df["ih"][0]
    for i in range(len(reccout[3])):
        if i!= 0:
            if (reccout[3][i][0] >= buffx) and (reccout[3][i][0] <= df["iw"][0] - buffx) and (reccout[3][i][1] >= buffy) and (reccout[3][i][1] <= df["ih"][0] - buffy):
                reccoutindx.append(i)

    leccoutindx = []
    buffx = xbuffer*df["iw"][1]
    buffy = ybuffer*df["ih"][1]
    for i in range(len(leccout[3])):
        if i!= 0:
            if (leccout[3][i][0] >= buffx) and (leccout[3][i][0] <= df["iw"][1] - buffx) and (leccout[3][i][1] >= buffy) and (leccout[3][i][1] <= df["ih"][1] - buffy):
                leccoutindx.append(i)
    print(reccout)
    ####---- identify the correction region by matching corresponding corneal reflections in both eyes ----
    ####---- this is based on properties of the detected regions ---------
    reflidx = []
    for i in range(len(reccoutindx)):
        for j in range(len(leccoutindx)):
            if abs(reccout[2][reccoutindx[i]][4] - leccout[2][leccoutindx[j]][4]) < refl_ar_diff and reccout[2][reccoutindx[i]][4] > min_refl_area and leccout[2][leccoutindx[j]][4] > min_refl_area and reccout[2][reccoutindx[i]][4] < max_refl_area and leccout[2][leccoutindx[j]][4] < max_refl_area:
                reflidx.append([reccoutindx[i],leccoutindx[j]])

    print(reccoutindx)

    rreflx = df["itx"][0] + reccout[3][reflidx[0][0]][0]
    rrefly= df["ity"][0] + reccout[3][reflidx[0][0]][1]
    lreflx = df["itx"][1] + leccout[3][reflidx[0][1]][0]
    lrefly = df["ity"][1] + leccout[3][reflidx[0][1]][1]

    redx = (rictr[0] - rreflx)
    ledx = (lictr[0] - lreflx)
    strab = hr_const*((redx/df["iw"][0]) + (ledx/df["iw"][1])) ## deviation

    strabis_angle = 10
    return strabis_angle

def predict_eye(img_path, model, fd):
     # with session.as_default():
     #     with session.graph.as_default():
    print("Entering the function ...")
    img = image.load_img(img_path)
    input_img = image.img_to_array(img)
    # print('Without Array')
    # print(input_img)
    # print("After Conversion to INT")
    input_img = input_img.astype('uint8')
    # print(input_img)
    # print(input_img.shape)
    with session.as_default():
        with session.graph.as_default():
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
            # print(inp_left)
            inp_left = cv2.equalizeHist(inp_left)
            # inp_left = cv2.resize(inp_left, (inp_left.shape[1],inp_left.shape[0]))[np.newaxis, ..., np.newaxis]
            inp_left = cv2.resize(inp_left, (180,108))[np.newaxis, ..., np.newaxis]
            # print(inp_left.shape)

            inp_right = cv2.cvtColor(right_eye_im, cv2.COLOR_RGB2GRAY)
            inp_right = cv2.equalizeHist(inp_right)
            # inp_right = cv2.resize(inp_right, (inp_right.shape[1],inp_right.shape[0]))[np.newaxis, ..., np.newaxis]
            inp_right = cv2.resize(inp_right, (180,108))[np.newaxis, ..., np.newaxis]
            # print(inp_right.shape)

            input_array = np.concatenate([inp_left, inp_right], axis=0)
            pred_left, pred_right = model.net.predict(input_array/255 * 2 - 1)

            # plt.figure(figsize=(15,4))
            # plt.subplot(1,3,1)
            # plt.title("Left eye")
            lms_left = model._calculate_landmarks(pred_left)
            result_left, inLineLeft  = draw_pupil(left_eye_im, inp_left, lms_left)
            # plt.axis('off')
            # plt.imshow(result_left)
            # plt.subplot(1,3,2)
            # plt.title("Right eye")
            lms_right = model._calculate_landmarks(pred_right)
            # print('lms_right',lms_right.shape)
            # print('inp_right',inp_right.shape)
            # print('right_eye_im',right_eye_im.shape)
            result_right, inLineRight = draw_pupil(right_eye_im, inp_right, lms_right)
            # plt.imshow(result_right)
            draw2 = input_img.copy()

            slice_h = slice(int(left_eye_xy[0]-eye_bbox_h//2), int(left_eye_xy[0]+eye_bbox_h//2))
            slice_w = slice(int(left_eye_xy[1]-eye_bbox_w//2), int(left_eye_xy[1]+eye_bbox_w//2))
            im_shape = left_eye_im.shape[::-1]

            draw2[slice_h, slice_w, :] = cv2.resize(result_left, im_shape[1:])

            slice_h = slice(int(right_eye_xy[0]-eye_bbox_h//2), int(right_eye_xy[0]+eye_bbox_h//2))
            slice_w = slice(int(right_eye_xy[1]-eye_bbox_w//2), int(right_eye_xy[1]+eye_bbox_w//2))
            im_shape = right_eye_im.shape[::-1]

            draw2[slice_h, slice_w, :] = cv2.resize(result_right, im_shape[1:])
            # print(draw2.shape)
            # plt.savefig('./Result/finaloutput_py_app.png')
            # result_right, outLineRight, inLineRight, pupLineRight = draw_pupil(right_eye_im, inp_right, lms_right)
            # result_left, outLineLeft, inLineLeft, pupLineLeft = draw_pupil(left_eye_im, inp_left, lms_left)

            # strabis_angle = predict_strabismus(right_eye_im, left_eye_im, inLineRight, inLineLeft)
            # print(strabis_angle)
            return result_left, result_right


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        print("make Prediction")
        predsLeft, predsRight = predict_eye(file_path, model, fd)
        # preds = model_predict(file_path, model)
        predsLeft = predsLeft.tolist()
        predsRight = predsRight.tolist()
        # print(len(predsRight))
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = "succesful run ..."
        # result = str(pred_class[0][0][1])               # Convert to string

        # image_data = {'data':preds}
        return json.dumps({'predsL': predsLeft,'predsR': predsRight})

    return None


if __name__ == '__main__':
    app.run(debug=True)
