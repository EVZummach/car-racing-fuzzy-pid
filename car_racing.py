import gym
import cv2
import math
import time

import numpy as np
import matplotlib.pyplot as plt

from PID import pid_control, pid_fuzzy
from gym.envs.box2d import CarRacing
from skimage.morphology import skeletonize
from fuzzy import acceleration_fuzzy, breaking_fuzzy, steering_fuzzy

x_min, x_max = 65, 125
y_min, y_max = 115, 168

x_car, y_car_min, y_car_max = 95, 135, 145

vel_min, vel_max = 0, 6

def show_img(image, name='Image', scale=200):
    img = image.copy()

    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)
    cv2.imshow(name, img)
    cv2.waitKey(1)

def get_car(image):
    width = image.shape[1]
    height = image.shape[0]
    car_line = np.zeros((height, width), np.uint8)
    cv2.line(car_line, (x_car, y_car_min), (x_car, y_car_max), (255, 255, 255), 1)

    return car_line

def pre_process(image):
    original = image.copy()

    image = image[:84,:]
    image = cv2.medianBlur(image, 5)
    width = int(image.shape[1] * 200 / 100)
    height = int(image.shape[0] * 200 / 100)
    image = cv2.resize(image, (width, height), cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thr = cv2.threshold(gray, 150, 1, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    thr = cv2.erode(thr, kernel, iterations=2)


    skeleton = skeletonize(thr, method='lee').astype(np.uint8)*255

    velocity = cv2.cvtColor(original[84:96, 11:16], cv2.COLOR_RGB2GRAY)
    _, thr_vel = cv2.threshold(velocity, 10, 1, cv2.THRESH_BINARY)
    sum_vel = np.sum(thr_vel)/3

    return original, image, thr, skeleton, sum_vel

def track_angle(skeleton):
    indices_y = np.where(np.any(skeleton == 255, axis=1))
    up_y, dn_y = np.min(indices_y), np.max(indices_y) if np.max(indices_y) < 35 else 35
    up_x = np.argmax(skeleton[up_y,:])
    dn_x = np.argmax(skeleton[dn_y,:])

    sk = cv2.cvtColor(skeleton.copy(), cv2.COLOR_GRAY2RGB)

    #cv2.circle(sk, (up_x, up_y), 0, (255, 0 ,0), 2)
    #cv2.circle(sk, (dn_x, dn_y), 0, (255, 0 ,0), 2)
    #cv2.line(sk, (dn_x, dn_y), (up_x, up_y), (0, 255, 0), 1)

    up_y = 44-up_y
    dn_y = 44-dn_y

    #print(f'Up:({up_x}, {up_y})')
    #print(f'Down:({dn_x}, {dn_y})')

    d_x = up_x-dn_x
    d_y = up_y-dn_y
    angle = np.arctan2(d_y, d_x)*180/np.pi

    #show_img(sk, 'Angles', 500)
    return angle

def car_distance(image):
    img = image.copy()
    x,y,w,h = cv2.boundingRect(img[:,:,0])
    car_center = (x+w//2, y+h//2)



    x,y,w,h = cv2.boundingRect(img[:35,:,1])
    track_center = (x+w//2, y+h//2)

    #cv2.circle(img, car_center, 0, (0, 0, 255), 2)
    #cv2.circle(img, track_center, 0, (255, 0 ,0), 2)
    #show_img(img, 'Center', 500)

    distance = track_center[0]-car_center[0]
    return distance

def get_info(image):
    original, image, thr, skeleton, vel = pre_process(image)
    car_line = get_car(image)
    car_roi = car_line[y_min:y_max, x_min:x_max]
    sk_roi = skeleton[y_min:y_max, x_min:x_max]

    img = np.zeros((y_max-y_min, x_max-x_min, 3), np.uint8)
    img[:, :, 0] = car_roi
    img[:, :, 1] = sk_roi

    angle_diff = 90-track_angle(sk_roi)
    distance = car_distance(img)
    image[np.where(skeleton==255)] = 0

    return angle_diff, distance, vel

def pid(error,previous_error, delta):
    Kp = 0.02
    Ki = 0.04
    Kd = 0.03

    steering = Kp * error + Ki * (error + previous_error)/delta + Kd * (error - previous_error)*delta

    return steering

def pid_test(error,previous_error, delta):
    Kp = 0.02
    Ki = 0.00
    Kd = 0.00

    steering = Kp * error + Ki * (error + previous_error)*delta + Kd * (error - previous_error)/delta

    return steering


env = gym.make("CarRacing-v2", render_mode="human")
env.action_space.seed(42)

observation, info = env.reset(seed=42, return_info=True)
prev_angle = prev_dist = prev_vel = 0
acceleration = steering = breaking = 0

start_delay = 1

diff_list = []
angles = []
run = 0

begin = time.time()

for _ in range(10000):
    diff = time.time()-begin

    observation, reward, done, info = env.step([steering, acceleration, breaking])

    if done:
        observation, info = env.reset(return_info=True)
        acceleration = breaking = steering = 0
        begin = time.time()
        diff = time.time()-begin
        np.savetxt(f'results/run_{run}.txt', np.array(diff_list), fmt='%2i')
        run += 1

    if diff > start_delay:
        angle, distance, vel = get_info(observation)
        steering = pid_control(0.005, 0.04, 0.03, distance, prev_dist, 1)#+pid_control(0.00002, 0.01, 0.02, angle, prev_angle, 1)
        #steering = pid_fuzzy([0.00, 0.1], [0, 0.00001], [0, 0.00001], distance, prev_dist, angle, prev_angle, vel, 1)
        #steering = steering_fuzzy(angle, distance, 0)
        #print(f'Angle:{angle:0.2f} Steering: {steering:.2f} Accel:{acceleration:.2f} Breaking:{breaking:.2f} Velocity:{vel} Distance:{distance}')
        breaking = breaking_fuzzy(angle, vel)
        acceleration = acceleration_fuzzy(angle, vel)

        diff_list.append([distance, angle, steering])
        prev_prev_dist = prev_dist
        prev_dist = distance

        prev_prev_angle = prev_angle
        prev_angle = angle

        prev_prev_vel = prev_vel
        prev_vel = vel

env.close()
