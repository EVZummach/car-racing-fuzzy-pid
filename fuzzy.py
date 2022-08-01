import numpy as np
import skfuzzy as fuzz

import matplotlib.pyplot as plt

def angle_fuzzy(angle, plot=0):
    a = np.arange(-91, 91, 1)
    a_hl = fuzz.trapmf(a, [-90, -80, -25, -15])
    a_l = fuzz.trimf(a, [-20, -10, -5])
    a_n = fuzz.trimf(a, [-10, 0, 10])
    a_r = fuzz.trimf(a, [5, 10, 20])
    a_hr = fuzz.trapmf(a, [15, 25, 80, 90])

    a_lvl_hl = fuzz.interp_membership(a, a_hl, angle)
    a_lvl_l  = fuzz.interp_membership(a, a_l, angle)
    a_lvl_n = fuzz.interp_membership(a, a_n, angle)
    a_lvl_r  = fuzz.interp_membership(a, a_r, angle)
    a_lvl_hr = fuzz.interp_membership(a, a_hr, angle)

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(a, a_hl, 'r', label='Hard Left')
        plt.plot(a, a_l, 'g', label='Left')
        plt.plot(a, a_n, 'b', label='Straight')
        plt.plot(a, a_r, 'c', label='Right')
        plt.plot(a, a_hr, 'm', label='Hard Right')
        plt.legend()
        plt.title('Fuzzy Logic: Angle')
        plt.show()

    return [a_lvl_hl, a_lvl_l, a_lvl_n, a_lvl_r, a_lvl_hr]

def velocity_fuzzy(velocity, plot=0):
    v = np.arange(0, 7, 0.5)
    v_s = fuzz.trapmf(v, [-1, 0, 1, 2])
    v_m = fuzz.trimf(v, [1, 2, 3])
    v_f = fuzz.trapmf(v, [2, 4, 6, 8])

    v_lvl_s = fuzz.interp_membership(v, v_s, velocity)
    v_lvl_m  = fuzz.interp_membership(v, v_m, velocity)
    v_lvl_f = fuzz.interp_membership(v, v_f, velocity)

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(v, v_s, 'r', label='Slow')
        plt.plot(v, v_m, 'g', label='Medium')
        plt.plot(v, v_f, 'b', label='Fast')
        plt.legend()
        plt.title('Fuzzy Logic: Fast')
        plt.show()

    return [v_lvl_s, v_lvl_m, v_lvl_f]

def distance_fuzzy(distance, plot = 0):
    d = np.arange(-30, 30, 1)

    d_hhl = fuzz.trapmf(d, [-30, -25, -18, -10])
    d_hl = fuzz.trimf(d, [-18, -12, -6])
    d_l = fuzz.trimf(d, [-12, -6, 0])
    d_n = fuzz.trimf(d, [-6, 0, 6])
    d_r = fuzz.trimf(d, [0, 6, 12])
    d_hr = fuzz.trimf(d, [6, 12, 18])
    d_hhr = fuzz.trapmf(d, [10, 18, 25, 30])

    d_lvl_hhl = fuzz.interp_membership(d, d_hhl, distance)
    d_lvl_hl = fuzz.interp_membership(d, d_hl, distance)
    d_lvl_l  = fuzz.interp_membership(d, d_l, distance)
    d_lvl_n = fuzz.interp_membership(d, d_n, distance)
    d_lvl_r = fuzz.interp_membership(d, d_r, distance)
    d_lvl_hr = fuzz.interp_membership(d, d_hr, distance)
    d_lvl_hhr = fuzz.interp_membership(d, d_hhr, distance)

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(d, d_hhl, 'r', label='Excessivo Direita')
        plt.plot(d, d_hl, 'g', label='Muito Direita')
        plt.plot(d, d_l, 'b', label='Direita')
        plt.plot(d, d_n, 'k', label='Reta')
        plt.plot(d, d_r, 'c', label='Esquerda')
        plt.plot(d, d_hr, 'm', label='Muito Esquerda')
        plt.plot(d, d_hhr, 'b', label='Excessivo Esquerda')
        plt.legend()
        plt.title('Fuzzy Logic: Distance')
        plt.show()

    return [d_lvl_hhl, d_lvl_hl, d_lvl_l, d_lvl_n, d_lvl_r, d_lvl_hr, d_lvl_hhr]

def acceleration_fuzzy(angle, velocity, plot=0):
    a = np.arange(0, 1, 0.001)

    a_s = 1-fuzz.smf(a, 0.0, 0.5)
    a_m = fuzz.gaussmf(a, 0.45, 0.1)
    a_h = fuzz.smf(a, 0.5, 0.6)

    hard_left, left, straight, right, hard_right = angle_fuzzy(angle)
    slow, medium, fast = velocity_fuzzy(velocity)

    medium_angle = np.fmax(left, right)
    high_angle = np.fmax(hard_left, hard_right)


    active_rule1 = np.fmax(high_angle, medium_angle) #If angle high
    active_rule2 = np.fmin(straight, fast) # If angle medium
    active_rule3 = np.fmin(straight, np.fmax(medium, slow)) # If angle low
    #print(f'Angle Activations: {hard_left}, {left}, {straight}, {right}, {hard_right}')
    #print(f'Velocity Activations: {slow}, {medium}, {fast}\n')
    a_activ_s = np.fmin(active_rule1, a_s)
    a_activ_m = np.fmin(active_rule2, a_m)
    a_activ_h = np.fmin(active_rule3, a_h)

    aggregated = np.fmax(a_activ_s, np.fmax(a_activ_m, a_activ_h))
    acceleration = fuzz.defuzz(a, aggregated, 'centroid')

    if plot:
        print(f'Angle Activations: {hard_left}, {left}, {straight}, {right}, {hard_right}')
        print(f'Velocity Activations: {slow}, {medium}, {fast}')
        print(acceleration)

        plt.figure(figsize=(10, 5))
        plt.plot(a, a_s, 'r', label='Small')
        plt.plot(a, a_m, 'g', label='Medium')
        plt.plot(a, a_h, 'b', label='High')
        plt.legend()
        plt.title('Fuzzy Logic: Acceleration')
        plt.show()

    return acceleration

def breaking_fuzzy(angle, velocity, plot=0):
    b = np.arange(0, 1, 0.001)

    b_n = fuzz.zmf(b, -0.1, 0.01)
    b_s = fuzz.trapmf(b, [-1, 0, 0.08, 0.1])
    b_m = fuzz.trapmf(b, [0.08, 0.1, 0.2, 0.6])
    b_h = fuzz.smf(b, 0.6, 0.8)

    hard_left, left, straight, right, hard_right = angle_fuzzy(angle)
    slow, medium, fast = velocity_fuzzy(velocity)

    medium_angle = np.fmax(left, right)
    high_angle = np.fmax(hard_left, hard_right)
    #print(f'Angle Activations: {hard_left}, {left}, {straight}, {right}, {hard_right}')
    #print(f'Velocity Activations: {slow}, {medium}, {fast}')

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(b, b_n, 'r', label='Não freia')
        plt.plot(b, b_s, 'g', label='Freia Pouco')
        plt.plot(b, b_m, 'b', label='Freia Médio')
        plt.plot(b, b_h, 'c', label='Freia Excessivo')
        plt.legend()
        plt.title('Fuzzy Logic: Breaking')
        plt.show()


    #print(straight, medium_angle, high_angle)
    #print(slow, medium, fast)

    active_rule1 = np.fmax(straight, np.fmin(np.fmax(high_angle, medium_angle), slow))
    active_rule2 = np.fmin(medium_angle, medium) # If angle low and speed is fast
    active_rule3 = np.fmax(np.fmin(medium_angle, fast), np.fmin(high_angle, medium))
    active_rule4 = np.fmin(high_angle, fast) #If angle high and velocity high or medium

    #print(active_rule1, active_rule2, active_rule3, active_rule4)

    b_activ_n = np.fmin(active_rule1, b_n)
    b_activ_s = np.fmin(active_rule2, b_s)
    b_activ_m = np.fmin(active_rule3, b_m)
    b_activ_h = np.fmin(active_rule4, b_h)

    aggregated = np.fmax(np.fmax(b_activ_s, np.fmax(b_activ_m, b_activ_h)), b_activ_n)
    breaking = fuzz.defuzz(b, aggregated, 'centroid')

    return breaking

def steering_fuzzy(angle, distance, plot=0):
    s = np.arange(-1.1, 1.1, 0.001)

    s_hhl = fuzz.trapmf(s, [-1.1, -1, -0.8, -0.6])
    s_hl = fuzz.trimf(s, [-0.8, -0.5, -0.2])
    s_l = fuzz.trimf(s, [-0.4, -0.2, 0])
    s_s = fuzz.trimf(s, [-0.1, 0, 0.1])
    s_r = fuzz.trimf(s, [0, 0.2, 0.4])
    s_hr = fuzz.trimf(s, [0.2, 0.5, 0.8])
    s_hhr = fuzz.trapmf(s, [0.6, 0.8, 1, 1.1])

    hard_left, left, straight, right, hard_right = angle_fuzzy(angle)
    dhh_left, dhard_left, dleft, dstraight, dright, dhard_right, dhh_right = distance_fuzzy(distance)

    medium_angle = np.fmax(left, right)
    high_angle = np.fmax(hard_left, hard_right)

    far = np.fmax(dhard_left, dhard_right)
    close = np.fmax(dleft, dright)
    #print(f'Angle Activations: {hard_left}, {left}, {straight}, {right}, {hard_right}')
    #print(f'Velocity Activations: {slow}, {medium}, {fast}')

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(s, s_hhl, 'r', label='Hard Hard Left')
        plt.plot(s, s_hl, 'r', label='Hard Left')
        plt.plot(s, s_l, 'g', label='Left')
        plt.plot(s, s_s, 'b', label='Straight')
        plt.plot(s, s_r, 'c', label='Right')
        plt.plot(s, s_hr, 'm', label='Hard Right')
        plt.plot(s, s_hhr, 'm', label='Hard Right')
        plt.legend()
        plt.title('Fuzzy Logic: Distance')
        plt.show()

    #print(straight, medium_angle, high_angle)
    #print(slow, medium, fast)

    active_rule1 = np.fmax(dhh_left, hard_left)
    active_rule2 = np.fmax(dhard_left, left)
    active_rule3 = np.fmax(dleft, left)
    active_rule4 = np.fmax(dstraight, straight)
    active_rule5 = np.fmax(dright, right)
    active_rule6 = np.fmax(dhard_right, right)
    active_rule7 = np.fmax(dhh_right, hard_right)

    #print(distance)
    #print(active_rule1, active_rule2, active_rule3, active_rule4, active_rule5, active_rule6, active_rule7)

    s_activ_hhl = np.fmin(active_rule1, s_hhl)
    s_activ_hl = np.fmin(active_rule2, s_hl)
    s_activ_l = np.fmin(active_rule3, s_l)
    s_activ_s = np.fmin(active_rule4, s_s)
    s_activ_r = np.fmin(active_rule5, s_r)
    s_activ_hr = np.fmin(active_rule6, s_hr)
    s_activ_hhr = np.fmin(active_rule7, s_hhr)

    activ_1 = np.fmax(s_activ_hl, s_activ_hr)
    activ_2 = np.fmax(s_activ_hhl, s_activ_hhr)
    activ_3 = np.fmax(s_activ_l, s_activ_r)

    aggregated = np.fmax(activ_1, np.fmax(activ_2, np.fmax(activ_3, s_activ_s)))
    steering = fuzz.defuzz(s, aggregated, 'centroid')

    return steering

def pid_fuzzy(angle, velocity, distance):
    return 0
