import numpy as np
import matplotlib.pyplot as plt

from fuzzy import distance_fuzzy, angle_fuzzy, velocity_fuzzy

import skfuzzy as fuzz

def pid_control(Kp, Ki, Kd, error, previous_error, delta):

    value = Kp * error + Ki * (error + previous_error)*delta + Kd * (error - previous_error)/delta

    return value

def pid_fuzzy(Kp, Ki, Kd, dist, prev_dist, angle, prev_angle, velocity, delta):
    kp = k_fuzzy(Kp[0], Kp[1], 4, dist, angle, velocity)
    ki = k_fuzzy(Ki[0], Ki[1], 4, dist, angle, velocity)
    kd = k_fuzzy(Kd[0], Kd[1], 4, dist, angle, velocity)
    #print(f'Ki={ki}')
    val = pid_control(kp, ki, kd, dist, prev_dist, delta)
    #print(error)
    #print(val)
    return val

def k_fuzzy(k_min, k_max, qtd, distance, angle, velocity):
    k = np.arange(k_min, k_max, 0.001)
    k_total = k_max - k_min
    step = k_total/3

    ks = []
    k_ns = []

    for i in range(0, qtd+1):
        k_n = fuzzy_trimf(k, step, i)
        k_ns.append(k_n)
        #k_lvl = fuzz.interp_membership(k, k_n, distance)
        #ks.append(k_lvl)

    k_ns = np.array(k_ns)

    #plt.figure(figsize=(10, 5))
    #for i, k_j in enumerate(k_ns):
    #    plt.plot(k, k_j, 'r', label=f'K{i}')
    #plt.legend()
    #plt.title('Fuzzy Logic: Distance')
    #plt.show()

    dhh_left, dhard_left, dleft, dstraight, dright, dhard_right, dhh_right = distance_fuzzy(distance)
    very_far = np.fmax(dhh_left, dhh_right)
    far = np.fmax(dhard_left, dhard_right)
    close = np.fmax(dleft, dright)

    hard_left, left, straight, right, hard_right = angle_fuzzy(angle)
    high_angle = np.fmax(hard_left, hard_right)
    medium_angle = np.fmax(left, right)

    slow, medium, fast = velocity_fuzzy(velocity)


    far_straight = np.fmin(np.fmax(far, very_far), straight)
    close_straight = np.fmin(np.fmax(close, far), straight)
    straight_straight = np.fmin(np.fmax(dstraight, close), straight)

    active_rule1 = np.fmin(np.fmax(np.fmax(dstraight, close), straight), np.fmax(fast, medium))
    active_rule2 = np.fmin(np.fmax(close, medium_angle), np.fmax(slow, medium))
    active_rule3 = np.fmin(np.fmax(np.fmax(close, far), np.fmax(medium_angle, high_angle)), np.fmax(slow, medium))
    active_rule4 = np.fmin(np.fmax(np.fmax(far, very_far), np.fmax(medium_angle, high_angle)), np.fmax(slow, medium))

    #active_rule1 = np.fmax(np.fmin(dstraight, straight), np.fmax(fast, medium))

    #active_rule2 = np.fmax(close, medium_angle)

    #active_rule3 = np.fmin(np.fmax(far, medium_angle), np.fmin(medium, fast))

    #active_rule4 = np.fmin(np.fmax(very_far, high_angle), slow)

    #print(k_ns[0].shape)
    #print(k.shape)
    #print(k_ns[0])

    k_activ_0 = np.fmin(active_rule1, k_ns[0])
    k_activ_1 = np.fmin(active_rule2, k_ns[1])
    k_activ_2 = np.fmin(active_rule3, k_ns[2])
    k_activ_3 = np.fmin(active_rule4, k_ns[3])

    aggregated = np.fmax(k_activ_0, np.fmax(k_activ_1, np.fmax(k_activ_2, k_activ_3)))

    kp = fuzz.defuzz(k, aggregated, 'centroid')
    #print(kp)

    return kp

def fuzzy_trimf(k, step, mult):
    return fuzz.trimf(k, [step*mult-step, step*mult, step*mult+step])

#pid_fuzzy([0, 0.3], [0, 0], [0, 0], 1, 0, 0)
