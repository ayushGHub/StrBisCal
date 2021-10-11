import os, sys, cv2
import numpy as np
def draw_pupil(im, inp_im, lms):
    draw = im.copy()
    draw = cv2.resize(draw, (inp_im.shape[2], inp_im.shape[1]))
#     print(draw.shape)
    pupil_center = np.zeros((2,))
    pnts_outerline = []
    pnts_innerline = []
    stroke = inp_im.shape[1] // 20 + 1
    for i, lm in enumerate(np.squeeze(lms)):
        #print(lm)
        y, x = int(lm[0]*3), int(lm[1]*3)

        if i < 8:
            #draw = cv2.circle(draw, (y, x), stroke, (125,255,125), 1)
            pnts_outerline.append([y, x])
        elif i < 16:
            #draw = cv2.circle(draw, (y, x), stroke, (125,125,255), 1)
            pnts_innerline.append([y, x])
            pupil_center += (y,x)
        elif i < 17:
            pass
            #draw = cv2.drawMarker(draw, (y, x), (255,200,200), markerType=cv2.MARKER_CROSS, markerSize=1, thickness=1, line_type=cv2.LINE_AA)
        else:
            pass
            #draw = cv2.drawMarker(draw, (y, x), (255,125,125), markerType=cv2.MARKER_CROSS, markerSize=1, thickness=1, line_type=cv2.LINE_AA)
    pupil_center = (pupil_center/8).astype(np.int32)
    # print(np.array(pnts_outerline).shape)
    # print(np.array(pnts_innerline).shape)
    draw = cv2.circle(draw, (pupil_center[0], pupil_center[1]), stroke, (255,0,0), 2)
    draw = cv2.polylines(draw, [np.array(pnts_outerline).reshape(-1,1,2)], isClosed=True, color=(0,255,125), thickness=2)
    draw = cv2.polylines(draw, [np.array(pnts_innerline).reshape(-1,1,2)], isClosed=True, color=(0,0,255), thickness=2)
    return draw, pnts_innerline
