import numpy as np
import cv2
import sys

bgr_color = 40, 30, 145 #tuple like list but immutable and can be accessed like array
color_threshold = 60
hsv_color = cv2.cvtColor( np.uint8([[bgr_color]] ), cv2.COLOR_BGR2HSV)[0][0] #numpy nd array  &&uint8 ==> unsigned integer (0,255)
HSV_lower = np.array([hsv_color[0] - color_threshold, hsv_color[1] - color_threshold, hsv_color[2] - color_threshold]) #Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].
HSV_upper = np.array([hsv_color[0] + color_threshold, hsv_color[1] + color_threshold, hsv_color[2] + color_threshold])


def detect_ball(frame): #take each frame captured
    x, y, radius = -1, -1, -1

    try:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert BGR to HSV
    except:
        return

    mask = cv2.inRange(hsv_frame, HSV_lower, HSV_upper)# (src,lower_boud,upper_bound) ==> create a mask
    mask = cv2.erode(mask, None, iterations=0) #
    mask = cv2.dilate(mask, None, iterations=4)

    im2, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = (-1, -1)

    # only proceed if at least one contour was found
    if len(contours) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(mask)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # check that the radius is larger than some threshold
        if radius > 10:
            #outline ball
            cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
            #show ball center
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

    return center[1]

def pixTOmeter(val): #[-484,-5] ==>[0,1]
    pos=(((-val - (-484.0)) * (1.0 - 0.0)) / ((-5.0) - (-484.0))) + 0.0 #NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    #value are hardcoded from 
    return pos

class PIDController:

    def __init__(self, target_pos):
        self.target_pos = target_pos
        self.Kp =3900.0#3800.0
        self.Ki =2400.0#2200.0
        self.Kd = 220.0#250.0
        self.bias = 0.0
        self.integral=0.0
        self.error_prior=0.0
        self.dt=1.0/60.0 #60 fps
        return

    def reset(self):
        return


    def get_fan_rpm(self, image_frame):
        output = 0.0
        vertical_ball_position =detect_ball(image_frame)
        vertical_ball_position_meter=pixTOmeter(vertical_ball_position)
        error=self.target_pos-vertical_ball_position_meter
        self.integral+=error*self.dt
        derivative=(error-self.error_prior)/self.dt
        output=self.Kp*error+self.Ki*self.integral+self.Kd*derivative+self.bias
        self.error_prior=error
        return output, vertical_ball_position_meter
