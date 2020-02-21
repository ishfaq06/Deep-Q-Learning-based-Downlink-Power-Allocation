# library

import sys
import math
import random
import math as mt
import numpy as np
import time
from keras import backend as K
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt


# function definition

def circle(x, y,r):
    ang=np.arange(0, 2*mt.pi, 0.01)
    xp=r*np.cos(ang)
    yp=r*np.sin(ang)
    plt.plot(x+xp,y+yp)
    return

def random_beam():
    x=[]
    x.append(60)
    x.append((60+120)%360)
    x.append((60-120)%360)
    return x

def point_d(thita,x1,y1,l):
    x2=[]
    y2=[]
    if thita==90:
        x2.append(x1)
        x2.append(x1)
        y2.append(y1+l)
        y2.append(y1-l)
    if (thita>180 and thita<270) or (thita<=180 and thita>90):
        x2.append(l*np.cos(np.deg2rad(thita)) + x1)
        y2.append(l * np.sin(np.deg2rad(thita)) + y1)
        x2.append(-l * np.cos(np.deg2rad(thita)) + x1)
        y2.append(-l * np.sin(np.deg2rad(thita)) + y1)
    else:
        x2.append(-l*np.cos(np.deg2rad(thita)) + x1)
        y2.append(-l * np.sin(np.deg2rad(thita)) + y1)
        x2.append(l * np.cos(np.deg2rad(thita)) + x1)
        y2.append(l * np.sin(np.deg2rad(thita)) + y1)


    return x2, y2



def random_allocation(c_x,c_y,angl,d,rng,num):
    X=[]
    Y=[]
    x=np.random.randint(d,rng+1,num)
    y=np.random.randint(0,angl,num)
    for i in range(num) :
        [a,b]=point_d(y[i],c_x,c_y,x[i])
        if y[i]<90 or y[i]>270 :
            X.append(a[1])
            Y.append(b[1])
        else:
            X.append(a[0])
            Y.append(b[0])
    return X,Y



def find_points_in_angle(x2,y2,x3,y3,angl,q,w):
    result_x=[]
    result_y=[]
    ang=[]
    global U_neighbor
    global U_neighbor_sector
    for i in range(len(x3)):
        if (x3[i]-x2)==0:
            if y3[i]>=y2:
                ang.append(90)
            else:
                ang.append(270)
        else:
            ang.append(np.rad2deg(np.arctan2((y3[i]-y2),(x3[i]-x2)))%360)
        angl1=angl+60
        angl2=angl-60
        if angl1>=360 and ang[i]>=0 and ang[i]<90:
            ang[i]=ang[i]+360
        if angl2<0:
            angl2=angl2%360
        if angl2>angl1:
            angl1=angl1+360
            if ang[i]>=0 and ang[i]<180:
                ang[i]=ang[i]+360
        r=ang[i]
        if angl2<=r and r<=angl1:
            t=1
        else:
            t=0
        if t==1:
            result_x.append(x3[i])
            result_y.append(y3[i])
            U_neighbor[i].extend([q])
            U_neighbor_sector[i].extend([w])
            t=0

    return result_x, result_y

def find_dis(x1,y1,x0,y0):
    return np.sqrt((x0-x1)**2+(y0-y1)**2)

def find_angl(x2,y2,x3,y3,angl):
    z=[]
    if (x3-x2)==0:
        if y3>=y2:
            ang=90
            z=ang
        else:
            ang=270
            z=ang
    else:
        ang=np.rad2deg(np.arctan2((y3-y2),(x3-x2)))%360
        if angl>=0 and angl<90 and ang>270 and ang < 360 :
            z=360-(ang-angl)
        elif ang >= 0 and ang < 90 and angl > 270 and angl < 360:
            z=360-(angl-ang)
        else:
            z=np.abs(angl-ang)

    return z

def power_watt_dbm(power):
    return 10*np.log10(power*1000)

def transmit_power(thita,thita_3db,Am, power_dbm):
    x=-np.min([12*(thita/thita_3db)**2, Am])
    g_x=10**(x/10)
    return (10**(power_dbm/10))/1000*g_x

def pw_m_hata(p,d):
    pl=128.1+37.6*np.log10(d/1000)
    G=10**(-pl/10)
    return G*p



def profile_power(subbands,slot,p_setting,power):
    ln=len(p_setting)

    pset=[]
    for i in range(len(slot)):

        p=power+1
        t=0
        k=[]
        while (p>=power):
            p=0
            t=t+1
            l=[]
            for j in range(len(subbands)):

                l.append(np.random.randint(0,ln,1))
                p=p+(subbands[j]*p_setting[l[j][0]])
            k.append(l)

        for j in range(len(subbands)):
            pset.append(p_setting[k[t - 1][j][0]])

    return pset


def convert_subband_power(sub,x):
    z=[]
    for i in range(len(x)) :
        z.append(sub[i]*x[i])

    return z


def generate_power_matrix_macro():
    global thita
    global U_x
    global num_M
    global N_subband
    global P_macro_subband
    global thita_3db
    global A_m
    global U_received_power_subband
    U_received_power_subband=[]

    for i in range(len(U_x)):
        U_received_power_subband.append([])

        for j in range(num_M):
            U_received_power_subband[i].append([])

            for a in range(N_subband):
                pw=P_macro_subband[j][a]
                pw_R= transmit_power(thita[i][j], thita_3db, A_m, power_watt_dbm(pw))
                U_received_power_subband[i][j].append(pw_m_hata(pw_R,U_macro_distance[i][j]))

    return

def sinr_db(p,pn):

    return 10* np.log10(p/pn)

def db_sinr(val):

    return 10**(val/10)

def sinr_to_cqi(i):

    if i<=-5:
        out=1

    elif i <=-2.5 and i> -5:
        out=2

    elif i <=0 and i> -2.5:
        out=3

    elif i <=2.4 and i> 0:
        out=4

    elif i <=4 and i>2.4:
        out=5

    elif i <=6 and i> 4:
        out=6

    elif i <=8 and i> 6:
        out=7

    elif i <=10 and i> 8:
        out=8

    elif i <=13 and i> 10:
        out=9

    elif i <=16 and i> 13:
        out=10

    elif i <=18 and i> 16:
        out=11

    elif i <=20.5 and i> 18:
        out=12

    elif i <=24 and i> 20.5:
        out=13

    elif i <=26.4 and i> 24:
        out=14

    elif i> 26.4:
        out=15

    return  out

def shanon_formula(p,pn):
    alpha=1
    del_f=12*15*10**3
    N0=(10**(-174/10))/1000
    z1=p/(N0*del_f+pn)

    return del_f*np.log2(1+alpha*z1)


def throughput():
    global U_received_power_subband
    global U_throughput_subband
    global U_SINR_subband
    global U_CQI_subband
    global P_own
    global P_intr
    global sub
    global U_x
    global U_association_macro
    global N_subband
    global num_M

    U_throughput_subband=[]
    U_SINR_subband=[]
    U_CQI_subband=[]

    for i in range(len(U_x)):
        U_throughput_subband.append([])
        U_SINR_subband.append([])
        U_CQI_subband.append([])
        q=U_association_macro[i]

        if q>0:
            id_cell=q-1
            P_own=[]
            P_intr=[]

            for X in range(N_subband):
                P_own.append(U_received_power_subband[i][id_cell][X])
                P_intr.append(0)

                for H in range(num_M):

                    if H !=id_cell:
                        P_intr[X] = P_intr[X] + U_received_power_subband[i][H][X]

                U_SINR_subband[i].append(sinr_db(P_own[X], P_intr[X]))
                U_CQI_subband[i].append(sinr_to_cqi(U_SINR_subband[i][X]))
                U_throughput_subband[i].append(sub[X] * shanon_formula(P_own[X], P_intr[X]) / (1024 * 1024))

    return

def optimize_resource_wmmse():

    #v_k_0=np.random.rand(0,np.sqrt(0.8),num_M)
    global P_macro_subband
    global N_subband
    global num_U
    global num_M
    global ln_fl_comb
    global final_comb
    global k_m_n
    global v_k_old
    global u_k_old
    global w_k_old


    global v_k_new
    global u_k_new
    global w_k_new

    del_f = 12 * 15 * 10 ** 3
    N0 = (10 ** (-174 / 10)) / 1000


    for i in range(num_M):
        P_macro_subband[i] = final_comb[np.random.randint(0, ln_fl_comb)]

    v_k_old=np.sqrt(P_macro_subband).tolist()

    generate_power_matrix_macro()
    throughput()

    K_macro_no = []
    k_m_n = []

    # for calculating user association

    for i in range(len(M_x)):
        [a, b] = calc_max(i * num_U, ((i + 1) * num_U))
        # K_macro_no.append(a)
        k_m_n.append(b)
    #print(k_m_n)

    # for u_k_old
    u_k_old=[]

    for i in range(len(M_x)):      # u_k
        u_k_old.append([])

        for X in range(N_subband):   # u_k^n

            sum_den = 0
            for j in range(len(M_x)):    # j=1

                sum_den = sum_den + (calculate_gain(j, k_m_n[i][X]) * calculate_power(j, i, k_m_n[i][X], X) )

            sum_den=sum_den+(N0*del_f)

            u_k_old[i].append(np.sqrt(calculate_gain(i,k_m_n[i][X])) * np.sqrt(calculate_power(i, i, k_m_n[i][X], X))/ sum_den)

    # for u_k_old
    w_k_old=[]

    for i in range(len(M_x)):      # w_k
        w_k_old.append([])

        for X in range(N_subband):   # w_k^n

            w_k_old[i].append(1/ (u_k_old[i][X] * np.sqrt(calculate_gain(i,k_m_n[i][X])) *  np.sqrt(calculate_power(i, i, k_m_n[i][X], X))))
    for itr_wmmse in range(40):

        update_v_k()
        update_u_k()
        update_w_k()
        if u_k_old==u_k_new:
            break
        u_k_old=u_k_new
        w_k_old=w_k_new

    generate_power_matrix_macro()
    throughput()

    O_macro_no = []
    O_macro_pos = []

    for i in range(len(M_x)):
        [a, b] = calc_max(i * num_U, ((i + 1) * num_U))
        O_macro_no.append(a)
        O_macro_pos.append(b)



    print('Final   :',P_macro_subband)



    return P_macro_subband, np.sum(O_macro_no)   # action , throughput

def update_v_k():
    global P_macro_subband
    global v_k_new
    global u_k_old
    global w_k_old
    global k_m_n
    # for v_k_new
    v_k_new = []

    for i in range(len(M_x)):  # v_k
        v_k_new.append([])

        for X in range(N_subband):  # v_k^n

            sum_den = 0
            for j in range(len(M_x)):  # j=1

                sum_den = sum_den + (calculate_gain(j, k_m_n[i][X]) * (( u_k_old[j][X])**2) * w_k_old[j][X])

            #sum_den = sum_den + 1

            v_k_new[i].append(
                np.sqrt(calculate_gain(i, k_m_n[i][X])) * w_k_old[i][X] * u_k_old[i][X] / sum_den)

    P_macro_subband=np.square(v_k_new).tolist()

    normalized_check_max()

    generate_power_matrix_macro()
    throughput()

    k_m_n = []

    # for calculating user association

    for i in range(len(M_x)):
        [a, b] = calc_max(i * num_U, ((i + 1) * num_U))
        # K_macro_no.append(a)
        k_m_n.append(b)

    #print(np.square(v_k_new).tolist())
    #print(P_macro_subband)

    return

def update_u_k():
    global u_k_new

    del_f = 12 * 15 * 10 ** 3
    N0 = (10 ** (-174 / 10)) / 1000
    u_k_new = []

    for i in range(len(M_x)):  # u_k
        u_k_new.append([])

        for X in range(N_subband):  # u_k^n

            sum_den = 0
            for j in range(len(M_x)):  # j=1

                sum_den = sum_den + (calculate_gain(j, k_m_n[i][X]) * calculate_power(j, i, k_m_n[i][X], X))

            sum_den = sum_den + (N0 * del_f)

            u_k_new[i].append(
                np.sqrt(calculate_gain(i, k_m_n[i][X])) * np.sqrt(calculate_power(i, i, k_m_n[i][X], X)) / sum_den)

    #print(u_k_new)

    return

def update_w_k():
    global w_k_new
    w_k_new = []

    for i in range(len(M_x)):  # w_k
        w_k_new.append([])

        for X in range(N_subband):  # w_k^n

            w_k_new[i].append(1 / (u_k_new[i][X] * np.sqrt(calculate_gain(i, k_m_n[i][X])) * np.sqrt(
                calculate_power(i, i, k_m_n[i][X], X))))

    return

def normalized_check_max():


    for i in range(len(M_x)):  #

        for X in range(N_subband):  #
            P_macro_subband[i][X]= normalized_power(P_macro_subband[i][X])

        if np.sum(P_macro_subband[i][0:N_subband]) >2.41:
            P_macro_subband[i]=(0.8*np.ones(N_subband)).tolist()

    return


def normalized_power(inpt):

    if inpt <(MC_power[0]+MC_power[1])/2:
        output=MC_power[0]
    elif (MC_power[0]+MC_power[1])/2 <= inpt < (MC_power[1]+MC_power[2])/2:
        output=MC_power[1]
    elif (MC_power[1]+MC_power[2])/2 <= inpt < (MC_power[2]+MC_power[3])/2:
        output=MC_power[2]
    elif (MC_power[2]+MC_power[3])/2 <= inpt < (MC_power[3]+MC_power[4])/2:
        output=MC_power[3]
    elif (MC_power[3]+MC_power[4])/2 <= inpt :
        output=MC_power[4]

    return output

def optimize_resource_iwf():

    global P_macro_subband
    global N_subband
    global num_U
    global num_M
    global ln_fl_comb
    global final_comb
    global k_m_n

    for i in range(num_M):
        P_macro_subband[i] = final_comb[np.random.randint(0, ln_fl_comb)]

    generate_power_matrix_macro()
    throughput()

    K_macro_no = []
    k_m_n = []

# for calculating user association

    for i in range(len(M_x)):
        [a, b] = calc_max(i * num_U, ((i + 1) * num_U))
        #K_macro_no.append(a)
        k_m_n.append(b)
    print(k_m_n)
    #g = np.sum(K_macro_no)


# for calulating t_m
    t_m=[]

    for i in range(len(M_x)):      # t_m
        t_m.append([])

        for X in range(N_subband):   # t_m^n

            sum_total = 0
            for j in range(len(M_x)):    # j=1, j=/m

                if j !=i:

                    sum_den = 0
                    for l in range(len(M_x)):     #l=1
                        sum_den = sum_den + (calculate_power(l, j, k_m_n[j][X], X) * calculate_gain(l, k_m_n[j][X]))

                    sum_den = sum_den + 1

                    sum_total=sum_total+(calculate_gain(i,k_m_n[j][X]) * calculate_sinr(j,k_m_n[j][X],X) )/sum_den
            t_m[i].append(sum_total)

    print(t_m)

    # for updating the power

    gamma_m=10
    p_m=[]

    for i in range(len(M_x)):      # p_m
        p_m.append([])

        for X in range(N_subband):   # p_m^n

            sum_num = 0
            for j in range(len(M_x)):    # j=1, j=/m

                if j !=i:

                    sum_num = sum_num + (calculate_power(j, j, k_m_n[j][X], X) * calculate_gain(j, k_m_n[j][X]))

            sum_num=sum_num+1

            p_m[i].append(1/((gamma_m* np.log(2))+t_m[i][X]) - sum_num/calculate_gain(i,k_m_n[i][X]))


    print("p_m   ", p_m)

    return

def calculate_sinr(cell_m,user_i,subband_x):

    return db_sinr(U_SINR_subband[(num_U*cell_m)+user_i][subband_x])


def calculate_gain(cell_m,user_i):

    d=U_macro_distance[(num_U*cell_m)+user_i][cell_m]

    pl=128.1+37.6*np.log10(d/1000)
    G=10**(-pl/10)
    #print('G    :',G)
    return G

def calculate_power(cell_m,cell_n,user_i,subband_x):

    pw = P_macro_subband[cell_m][subband_x]
    pw_R = transmit_power(thita[(num_U*cell_n)+user_i][subband_x], thita_3db, A_m, power_watt_dbm(pw))
    #print(pw,pw_R)
    return pw_R



def optimize_resource_ga():
    global ga_comb
    global ga_comb2
    global ga_fit_comb
    global P_macro_subband
    global N_subband
    global O_macro_pos
    global O_macro_no
    global O_macro_sum
    global num_M
    global num_U
    global sub
    global slots_slots
    global MC_power
    global P_max_macro

    O_macro_sum=[]
    ga_comb_temp=[]

    for itr1 in range(6000):
        x=[]

        for i in range(num_M):
           x.extend(profile_power(sub,slots_slots,MC_power,P_max_macro))

        ga_comb_temp.append(x)

    ga_comb=np.unique(ga_comb_temp,axis=0).tolist()


    for itr1 in range(len(ga_comb)):

        for i in range(len(M_x)):
            P_macro_subband[i]=ga_comb[itr1][i*N_subband: (i+1)*N_subband]

        #print(ga_comb_temp[itr1],P_macro_subband)
        generate_power_matrix_macro()
        throughput()

        O_macro_no = []
        O_macro_pos = []

        for i in range(len(M_x)):
            [a,b]=calc_max(i*num_U,((i+1)*num_U))
            O_macro_no.append(a)
            O_macro_pos.append(b)

        O_macro_sum.append(np.sum(O_macro_no))

    q=np.sort(O_macro_sum).tolist()
    Q = np.sort(O_macro_sum).tolist()
    w = np.argsort(O_macro_sum).tolist()
    W = np.argsort(O_macro_sum).tolist()

    O_macro_no=[]
    O_macro_pos=[]
    O_macro_sum=[]
    ga_fit_comb=[[],[],[],[],[],[],[],[],[],[],[],[]]
    ga_comb2=ga_comb

    LEN=len(ga_comb2)

    for uu in range(3000):
        O_macro_no = []
        O_macro_pos = []
        O_macro_sum=[]
        #prev_config=[]

        ga_fit_comb[0]=ga_comb[w[-1]]
        ga_fit_comb[1] = ga_comb[w[-2]]
        ga_fit_comb[2] = ga_comb[w[-3]]
        ga_fit_comb[3] = ga_comb[w[-4]]

        ga_comb=[]
        w=[]
        q=[]

        [ga_fit_comb[4],ga_fit_comb[5]]=split_merge_ga(ga_fit_comb[0],ga_fit_comb[1])
        ga_fit_comb[4]=mutation_ga(ga_fit_comb[4])
        ga_fit_comb[5] = mutation_ga(ga_fit_comb[5])

        [ga_fit_comb[6], ga_fit_comb[7]] = split_merge_ga(ga_fit_comb[0], ga_fit_comb[2])
        ga_fit_comb[6] = mutation_ga(ga_fit_comb[6])
        ga_fit_comb[7] = mutation_ga(ga_fit_comb[7])

        [ga_fit_comb[8], ga_fit_comb[9]] = split_merge_ga(ga_fit_comb[0], ga_fit_comb[3])
        ga_fit_comb[8] = mutation_ga(ga_fit_comb[8])
        ga_fit_comb[9] = mutation_ga(ga_fit_comb[9])


        [ga_fit_comb[2], ga_fit_comb[3]] = split_merge_ga(ga_comb2[W[np.random.randint(LEN-10, LEN,1)[0]]], ga_comb2[W[np.random.randint(LEN-10, LEN,1)[0]]])
        ga_fit_comb[2] = mutation_ga(ga_fit_comb[2])
        ga_fit_comb[3] = mutation_ga(ga_fit_comb[3])

        [ga_fit_comb[10], ga_fit_comb[11]] = split_merge_ga(ga_comb2[W[np.random.randint(LEN - 50, LEN, 1)[0]]],ga_comb2[W[np.random.randint(LEN - 50, LEN, 1)[0]]])
        ga_fit_comb[10] = mutation_ga(ga_fit_comb[10])
        ga_fit_comb[11] = mutation_ga(ga_fit_comb[11])

        ga_comb = np.unique(ga_fit_comb, axis=0).tolist()

        OM_position=[]

        for itr1 in range(len(ga_comb)):

            for i in range(len(M_x)):
                P_macro_subband[i] = ga_comb[itr1][i * N_subband: (i + 1) * N_subband]

            # print(ga_comb_temp[itr1],P_macro_subband)
            generate_power_matrix_macro()
            throughput()

            O_macro_no = []
            O_macro_pos = []

            for i in range(len(M_x)):
                [a, b] = calc_max(i * num_U, ((i + 1) * num_U))
                O_macro_no.extend(a)
                O_macro_pos.extend(b)

            #print(O_macro_pos)
            OM_position.append(O_macro_pos)
            O_macro_sum.append(np.sum(O_macro_no))

        #print(O_macro_sum)

        q = np.sort(O_macro_sum).tolist()
        w = np.argsort(O_macro_sum).tolist()

        current_config = ga_comb[w[-1]]

        if uu==0:
            prev_config= ga_comb[w[-1]]
            cont_itr=0

        if current_config==prev_config :
            cont_itr=cont_itr+1
        else:
            cont_itr=0

        prev_config=ga_comb[w[-1]]


        if cont_itr >500:
            break

    power_config=ga_comb[w[-1]]
    ff=np.digitize(power_config,MC_power).tolist()
    ff.extend(OM_position[w[-1]])
    #print("power_config",ff)
    #g_ga_th.append(q[-1])
    #print("throughput_ga :", q[-1])
    return power_config,ff,q[-1]

def calc_max(xi,xe):
    global U_throughput_subband

    return np.amax(U_throughput_subband[xi:xe],axis=0).tolist() , np.argmax(U_throughput_subband[xi:xe],axis=0).tolist()


def split_merge_ga(v1,v2):

    z1=[]
    z2=[]
    a=np.random.randint(1,num_M+1,1)
    b = np.random.randint(0, num_M+1 - a, 1)

    temp_v11=v1[0:(a[0]-1)*N_subband]
    temp_v12 = v1[(a[0]-1) * N_subband: (a[0]+b[0])*N_subband]
    temp_v13 = v1[(a[0] + b[0]) * N_subband:]

    temp_v21 = v2[0:(a[0] - 1) * N_subband]
    temp_v22 = v2[(a[0] - 1) * N_subband: (a[0] + b[0]) * N_subband]
    temp_v23 = v2[(a[0] + b[0]) * N_subband:]

    z1.extend(temp_v11)
    z1.extend(temp_v22)
    z1.extend(temp_v13)

    z2.extend(temp_v21)
    z2.extend(temp_v12)
    z2.extend(temp_v23)

    return z1, z2

def mutation_ga(v):

    global MC_power
    z=[]

    temp=v[:]

    a=np.random.randint(0,7,1)
    if a[0]==1:
        ln=len(v)
        p1=np.random.randint(0,ln,1)
        q1=np.random.randint(0,len(MC_power),1)

        p2 = np.random.randint(0, ln, 1)
        q2 = np.random.randint(0, len(MC_power), 1)

        temp[p1[0]]=MC_power[q1[0]]
        temp[p2[0]] = MC_power[q2[0]]

        right=check_max(temp)

        if right <0.5:
            z=temp
        else:
            z=v

    elif a[0] == 2:
        ln = len(v)
        p1 = np.random.randint(0, ln, 1)
        q1 = np.random.randint(0, len(MC_power), 1)

        p2 = np.random.randint(0, ln, 1)
        q2 = np.random.randint(0, len(MC_power), 1)

        p3 = np.random.randint(0, ln, 1)
        q3 = np.random.randint(0, len(MC_power), 1)

        temp[p1[0]] = MC_power[q1[0]]
        temp[p2[0]] = MC_power[q2[0]]
        temp[p3[0]] = MC_power[q3[0]]

        right = check_max(temp)

        if right < 0.5:
            z = temp
        else:
            z = v

    elif a[0] == 3:
        ln = len(v)
        p1 = np.random.randint(0, ln, 1)
        q1 = np.random.randint(0, len(MC_power), 1)

        p2 = np.random.randint(0, ln, 1)
        q2 = np.random.randint(0, len(MC_power), 1)

        p3 = np.random.randint(0, ln, 1)
        q3 = np.random.randint(0, len(MC_power), 1)

        p4 = np.random.randint(0, ln, 1)
        q4 = np.random.randint(0, len(MC_power), 1)

        temp[p1[0]] = MC_power[q1[0]]
        temp[p2[0]] = MC_power[q2[0]]
        temp[p3[0]] = MC_power[q3[0]]
        temp[p4[0]] = MC_power[q4[0]]

        right = check_max(temp)

        if right < 0.5:
            z = temp
        else:
            z = v

    elif a[0] == 4:
        ln = len(v)
        p1 = np.random.randint(0, ln, 1)
        q1 = np.random.randint(0, len(MC_power), 1)

        p2 = np.random.randint(0, ln, 1)
        q2 = np.random.randint(0, len(MC_power), 1)

        p3 = np.random.randint(0, ln, 1)
        q3 = np.random.randint(0, len(MC_power), 1)

        p4 = np.random.randint(0, ln, 1)
        q4 = np.random.randint(0, len(MC_power), 1)

        p5 = np.random.randint(0, ln, 1)
        q5 = np.random.randint(0, len(MC_power), 1)

        temp[p1[0]] = MC_power[q1[0]]
        temp[p2[0]] = MC_power[q2[0]]
        temp[p3[0]] = MC_power[q3[0]]
        temp[p4[0]] = MC_power[q4[0]]
        temp[p5[0]] = MC_power[q5[0]]

        right = check_max(temp)

        if right < 0.5:
            z = temp
        else:
            z = v


    else :
        ln = len(v)
        p = np.random.randint(0, ln, 1)
        q = np.random.randint(0, len(MC_power), 1)

        temp[p[0]] = MC_power[q[0]]

        right = check_max(temp)

        if right < 0.5:
            z = temp
        else:
            z = v


    return z


def check_max(v):

    z=0
    for irr in range(num_M):

        if np.sum(v[irr*N_subband:(irr+1)*N_subband]) <= 2.5 :
            z=(z | 0)
        else:
            z= (z | 1)

    return z

def check_ga_with_drl(action):
    global g_total_call
    global g_total_corrected
    global final_comb
    global M_x
    global ln_fl_comb
    action_drl=[]
    if (g_cntr-1) < g_data_len:
        #print("running ---------------")
        action_ga=data_set['arr_2'][g_cntr-1].tolist()
        ga_th=data_set['arr_3'][g_cntr-1].tolist()
        action_wmmse = data_set['arr_4'][g_cntr - 1].tolist()
        wmmse_th = data_set['arr_5'][g_cntr - 1].tolist()

    else:

        action_ga,_,ga_th=optimize_resource_ga()
        g_data_action_ga.append(action_ga)
        g_data_ga_th.append(ga_th)
        action_wmmse, wmmse_th = optimize_resource_wmmse()
        g_data_action_wmmse.append(action_wmmse)
        g_data_wmmse_th.append(wmmse_th)


    g_ga_th.append(ga_th)
    g_wmmse_th.append(wmmse_th)

    #print("throughput_ga :", ga_th)

    for i in range(len(M_x)):
        action_drl.extend(final_comb[action[i]-i*ln_fl_comb])
    answer=np.double(action_drl==action_ga)
    print(answer,action_drl,action_ga)
    g_total_call=g_total_call+1
    g_total_corrected=g_total_corrected+answer
    g_drl_th.append(g_drl_th_val)

    g_eql_th.append(g_eql_th_val)
    g_rnd_th.append(g_rnd_th_val)



    print("Throughput GA: ",ga_th,' DRL: ', g_drl_th_val,' EQL: ', g_eql_th_val,' RND: ',g_rnd_th_val,' WMMSE: ',wmmse_th)
    np.savez("result_5c_3h_5p.npz", g_drl_th, g_ga_th,g_eql_th,g_rnd_th,g_wmmse_th)
    np.save("counter_v2_5c_3h_5p_lr1.npy",g_cntr)
    if g_cntr > g_data_len:
        np.savez("data_set_5c.npz",g_data_x,g_data_y,g_data_action_ga,g_data_ga_th,g_data_action_wmmse,g_data_wmmse_th)


    return


########################################################################################################################

# functions related to DQN
def self_env_step(a):
    global ln_fl_comb
    global M_x
    global P_max_macro
    global num_U
    global num_M
    global g_counter
    global O_macro_no
    global O_macro_pos
    global final_comb
    global current_throughput
    global previous_throughput

    g_counter=g_counter+1

    for i in range(len(M_x)):
        P_macro_subband[i] = final_comb[a[i]-i*ln_fl_comb]

    generate_power_matrix_macro()
    throughput()

    O_macro_no = []
    O_macro_pos = []

    for i in range(len(M_x)):
        [a, b] = calc_max(i * num_U, ((i + 1) * num_U))
        O_macro_no.append(a)
        O_macro_pos.append(b)

    current_throughput=np.sum(O_macro_no)

    if current_throughput> previous_throughput and g_counter<=10:
        if current_throughput==previous_throughput:
            reward=1
        else:
            reward = 1
        done=False

        previous_throughput=current_throughput

        temp_var = []     # temp_var here means next state

        for i in range(num_M * num_U):
            temp_var.extend(U_CQI_subband[i])
            temp_var.append(np.double(U_macro_distance[i][U_association_macro[i] - 1] / R_m >= 0.5))
    else:
        if g_counter>10 and current_throughput>=previous_throughput:
            reward=1
        else:
            reward=1
        temp_var=None
        done=True



    return np.array(temp_var), reward, done, current_throughput




# ----------
HUBER_LOSS_DELTA = 1.0
LEARNING_RATE = 0.00025

# ----------
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)


# -------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel()

    def _createModel(self):
        model = Sequential()

        model.add(Dense(units=actionCnt*4, activation='relu', input_dim=stateCnt))
        model.add(Dense(units=actionCnt * 3, activation='relu'))
        model.add(Dense(units=actionCnt*2,activation='relu'))
        model.add(Dense(units=actionCnt, activation='linear'))

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=opt)
        #model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=64*1, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):

        return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()
        #return self.predict([s], target=target)

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())



# -------------------- MEMORY --------------------------
class Memory:  # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def isFull(self):
        return len(self.samples) >= self.capacity



# -------------------- AGENT ---------------------------
MEMORY_CAPACITY = 50000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001  # speed of decay

UPDATE_TARGET_FREQUENCY = 1000


class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        global num_M
        global ln_fl_comb

        temp_arg= self.brain.predictOne(s)
        #print("temp_arg",len(temp_arg))
        z=[]
        for i in range(num_M):
            z.append(i*ln_fl_comb+np.argmax(temp_arg[i*ln_fl_comb:(i+1)*ln_fl_comb]))

            if random.random() < self.epsilon:
                z[i]=np.random.randint(i * ln_fl_comb, (i + 1) * ln_fl_comb)
                #print("yes", i,z[i],self.epsilon)

        return z


    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()
            np.save("data_5c_3h_5p.npy", agent.memory.samples)
            np.save("step_5c_3h_5p.npy",agent.steps)
            agent.brain.model.save("dqn_5c_3h_5p.h5")

        # debug the Q function in poin S
        if self.steps % 100 == 0:
            S = init_env
            pred = agent.brain.predictOne(S)
            print(pred)
            sys.stdout.flush()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = np.zeros(self.stateCnt)

        states = np.array([o[0] for o in batch])
        #print("states", states.shape)
        states_ = np.array([(no_state if o[3] is None else o[3]) for o in batch])
        #print("states-----",states_)
        p = self.brain.predict(states)
        p_ = self.brain.predict(states_, target=True)

        x = np.zeros((batchLen, self.stateCnt))
        y = np.zeros((batchLen, self.actionCnt))

        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            #print("a    =",a)

            t = p[i]
            #print("before   :",t)
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * np.amax(p_[i])
            #t[a] = r + GAMMA * np.amax(p_[i])
            #print("after   :",t)
            x[i] = s
            y[i] = t

        self.brain.train(x, y)


class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)
    #memory = Memory(10000)

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt

    def act(self, s):
        global num_M
        global ln_fl_comb
        z=[]
        for i in range(num_M):
            z.append(np.random.randint(i*ln_fl_comb,(i+1)*ln_fl_comb))
        return z

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

    def replay(self):
        pass



# -------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem
        #self.env = gym.make(problem)

    def run(self, agent):
        global g_action
        global g_drl_th_val
        global ln_fl_comb
        global M_x
        #s = self.env.reset()
        #s_ini,s=self.env_reset(agent)
        s,s_ini = self.env_reset(agent)
        #print(s_ini)
        R = 0
        #g_action = [0 + ln_fl_comb * i for i in range(len(M_x))]

        while True:
            #self.env.render()

            a = agent.act(s)
            #print(a)

            #s_, r, done, info = self.env.step(a)
            s_,r,done, t_th = self_env_step(a)
            #print("type s_--",type(s_))

            if done:  # terminal state
                s_ = None

            agent.observe((s, a, r, s_))
            agent.replay()

            s = s_
            R += r

            if done:
                break

            g_action=a
            g_drl_th_val=t_th

        print("Total reward:", R)
        action_drl = []
        for i in range(len(M_x)):
            action_drl.extend(final_comb[g_action[i] - i * ln_fl_comb])
        print(action_drl)
        return R, g_action

    def env_reset(self, agent):

        global thita
        global g_drl_th_val
        global g_eql_th_val
        global g_rnd_th_val
        global thita_3db
        global P_max_macro
        global MC_power
        global slots_slots
        global sub
        global A_m
        global R_m
        global num_U
        global num_M
        global M_x
        global M_y
        global M_cell_beam
        global M_cell_associated_user_id
        global U_x
        global U_y
        global U_association_macro
        global U_neighbor
        global U_neighbor_sector
        global M_cell_region_x
        global M_cell_region_y
        global U_macro_distance
        global U_macro_power
        global M_cell_txblock_power
        global P_macro_subband
        global U_received_power_subband
        global U_throughput_subband
        global U_SINR_subband
        global U_CQI_subband
        global g_action
        global g_cntr

        global current_throughput
        global previous_throughput

        global g_counter

        global k_m_n
        global v_k_old
        global u_k_old
        global w_k_old

        global v_k_new
        global u_k_new
        global w_k_new

        k_m_n=[]
        v_k_old=[]
        u_k_old=[]
        w_k_old=[]

        v_k_new=[]
        u_k_new=[]
        w_k_new=[]


        g_counter = 0
        current_throughput = []
        previous_throughput = 0
        g_drl_th_val = 0
        g_eql_th_val = 0
        g_rnd_th_val=0

        M_cell_beam = []
        M_cell_associated_user_id = []
        M_cell_region_x = []
        M_cell_region_y = []
        M_cell_txblock_power = []



        U_x = []
        U_y = []
        U_association_macro = []
        U_neighbor = []
        U_neighbor_sector = []
        U_macro_distance = []
        U_macro_power = []
        U_throughputt = []

        P_macro_subband = []
        U_received_power_subband = []
        U_throughput_subband = []

        U_SINR_subband = []
        U_CQI_subband = []

        itr = 0
        for i in range(len(M_x)):
            M_cell_beam.append(random_beam())
            M_cell_associated_user_id.append([])

            for j in range(num_U):
                M_cell_associated_user_id[i].extend([itr + 1])
                U_association_macro.append(i + 1)
                itr = itr + 1

            #[x, y] = random_allocation(M_x[i], M_y[i], 360, 0.2 * R_m, R_m, num_U)
            #U_x.extend(x)
            #U_y.extend(y)


# ***********************************************************************************************************************

        #g_data_x.append(U_x)
        #g_data_y.append(U_y)
        if g_cntr<g_data_len:
            U_x=data_set['arr_0'][g_cntr].tolist()
            #print(U_x)
            U_y = data_set['arr_1'][g_cntr].tolist()

        else:
            for i in range(len(M_x)):

                [x, y] = random_allocation(M_x[i], M_y[i], 360, 0.2 * R_m, R_m, num_U)
                U_x.extend(x)
                U_y.extend(y)

            g_data_x.append(U_x)
            g_data_y.append(U_y)

        g_cntr=g_cntr+1


# ***********************************************************************************************************************
        for i in range(len(U_x)):
            U_neighbor.append([])
            U_neighbor_sector.append([])

        for i in range(len(M_x)):
            M_cell_region_x.append([[], [], []])
            M_cell_region_y.append([[], [], []])

            for j in range(3):
                [a, b] = find_points_in_angle(M_x[i], M_y[i], U_x, U_y, M_cell_beam[i][j], i + 1, j + 1)
                M_cell_region_x[i][j].extend(a)
                M_cell_region_y[i][j].extend(b)

        thita = np.zeros((len(U_x), len(M_x)))

        for i in range(len(U_x)):
            U_macro_distance.append([])
            U_macro_power.append([])

            for j in range(len(M_x)):
                U_macro_distance[i].append(find_dis(U_x[i], U_y[i], M_x[j], M_y[j]))
                angl = M_cell_beam[j][U_neighbor_sector[i][j] - 1]

                thita[i][j] = find_angl(M_x[j], M_y[j], U_x[i], U_y[i], angl)
                a = transmit_power(thita[i][j], thita_3db, A_m, power_watt_dbm(P_max_macro))
                U_macro_power[i].append(pw_m_hata(a, U_macro_distance[i][j]))

        for i in range(len(M_x)):
            M_cell_txblock_power.append(profile_power(sub, slots_slots, MC_power, P_max_macro))
            P_macro_subband.append(M_cell_txblock_power[i])

        generate_power_matrix_macro()

        throughput()

        temp_var = []

        for i in range(num_M * num_U):
            temp_var.extend(U_CQI_subband[i])
            temp_var.append(np.double(U_macro_distance[i][U_association_macro[i] - 1] / R_m >= 0.5))

        #a = agent.act(np.array(temp_var))
        #g_action = a

        g_action = [0 + ln_fl_comb * i for i in range(len(M_x))]

        #for i in range(len(M_x)):
            #P_macro_subband[i] = final_comb[a[i] - i * ln_fl_comb]

        for i in range(len(M_x)):
            P_macro_subband[i] = final_comb[0]

        generate_power_matrix_macro()
        throughput()

        temp_var2 = []

        for i in range(num_M * num_U):
            temp_var2.extend(U_CQI_subband[i])
            temp_var2.append(np.double(U_macro_distance[i][U_association_macro[i] - 1] / R_m >= 0.5))

        O_macro_no = []
        O_macro_pos = []

        for i in range(len(M_x)):
            [a, b] = calc_max(i * num_U, ((i + 1) * num_U))
            O_macro_no.append(a)
            O_macro_pos.append(b)

        previous_throughput = np.sum(O_macro_no)
        g_drl_th_val = previous_throughput



# Maximum power allocation  ##############################################

        for i in range(len(M_x)):
            P_macro_subband[i] = 0.8*np.ones(N_subband)

        generate_power_matrix_macro()
        throughput()

        O_macro_no = []
        O_macro_pos = []

        for i in range(len(M_x)):
            [a, b] = calc_max(i * num_U, ((i + 1) * num_U))
            O_macro_no.append(a)
            O_macro_pos.append(b)

        g_eql_th_val = np.sum(O_macro_no)

# Random power allocation  ##############################################
        for i in range(num_M):
            P_macro_subband[i] = final_comb[np.random.randint(0,ln_fl_comb)]

        generate_power_matrix_macro()
        throughput()

        O_macro_no = []
        O_macro_pos = []

        for i in range(len(M_x)):
            [a, b] = calc_max(i * num_U, ((i + 1) * num_U))
            O_macro_no.append(a)
            O_macro_pos.append(b)

        g_rnd_th_val = np.sum(O_macro_no)

        return np.array(temp_var), np.array(temp_var2)

########################################################################################################################

# variable definition

current_throughput=[]
previous_throughput=0

g_counter=0
g_action=[]
g_total_call=0
g_total_corrected=0
g_ga_th=[]
g_wmmse_th=[]
g_drl_th_val=0
g_drl_th=[]
g_eql_th_val=0
g_rnd_th_val=0
g_eql_th=[]
g_rnd_th=[]

g_cntr=0
g_data_len=0

g_data_x=[]
g_data_y=[]

g_data_action_ga=[]
g_data_ga_th=[]

g_data_action_wmmse=[]
g_data_wmmse_th=[]



M_cell_beam=[]
M_cell_associated_user_id=[]
M_cell_cor_x=[]
M_cell_cor_y=[]
M_cell_region_x=[]
M_cell_region_y=[]
M_cell_txblock_power=[]

U_x=[]
U_y=[]
U_association_macro=[]
U_neighbor=[]
U_neighbor_sector=[]
U_macro_distance=[]
U_macro_power=[]
U_throughputt=[]

P_macro_subband=[]

U_received_power_subband=[]
U_throughput_subband=[]
U_SINR_subband=[]
U_CQI_subband=[]

ga_comb=[]
ga_comb2=[]
ga_fit_comb=[]

O_macro_no=[]
O_macro_pos=[]
O_macro_sum=[]

thita=[]


thita_3db=70
A_m=35
P_max_macro=40
alpha=25
r=1000
R_m=500
min_RM=600
num_M=5
num_U=5
N=9000
N_subband=3

MC_power_1=[0.4, 0.6, 0.8,1.0,1.2]
MC_power=[0.4,0.6, 0.8,1.0,1.2]

N_RB=48
N_slot=1
N_subframes=1
sub=[]
slots_slots=[1]
sub.append(mt.floor(N_RB/N_subband))
sub.append(mt.floor(N_RB/N_subband))

# for 5 cell system
M_x=[272, -270, -72, 490, -804]
M_y=[668, 136, -839, -100, -405]

# for 10 cell system
#M_x=[-507.325029580438,1002.33404665068,145.312062677023,200.78357265330,1638.02723651541,1373.51667496699,-505.567176719993,89.0624746033709,888.768341569969,700.98746226480]
#M_y=[622.439530990585,1204.09269092070,-857.441644095846,0.776230721301,-266.682269021143,409.885301601770,-276.654621800917,971.99443198228,-404.073417200223,500.42153804114]

# for 15 cell system
#M_x=[-507.325029580438,1002.33404665068,145.312062677023,1008.94157032003,200.78357265330,1638.02723651541,1373.51667496699,-505.567176719993,89.0624746033709,208.3844792471699,-1205.45088276488,-1203.86653509258,888.768341569969,-623.017861398548,700.98746226480]
#M_y=[622.439530990585,1204.09269092070,-857.441644095846,-1254.30978634305,0.776230721301,-566.682269021143,309.885301601770,-276.654621800917,971.99443198228,-1708.16795562791,281.640997912069,-609.705714626437,-404.073417200223,-1209.60615767022,500.42153804114]


sub.append(N_RB-sub[N_subband-2]*(N_subband-1))
#print(sub)
final_comb=[]
P_own=[]
P_intr=[]
P_SINR=[]


# final_comb : combination of all the actions for a particular cell

power_comb = list(map(list, itertools.product(MC_power_1, repeat=N_subband)))

for i in power_comb:
    temp_sum=0.0
    for j in range(N_subband):
        temp_sum=temp_sum+ i[j]*sub[j]
    if temp_sum <=40 :
        final_comb.append(i)

ln_fl_comb=len(final_comb)

#print(final_comb[12])

plt.figure(1)
plt.plot(M_x,M_y,'b*')

for i in range(len(M_x)):
    circle(M_x[i],M_y[i],R_m)
#plt.show()  # to plot the figure





# -------------------- MAIN ----------------------------
PROBLEM = 'CartPole-v0'
env = Environment(PROBLEM)

#stateCnt = env.env.observation_space.shape[0]
stateCnt=num_M*num_U*(3+1)
#actionCnt = env.env.action_space.n
actionCnt=ln_fl_comb*num_M
print("actions    :",actionCnt)
agent = Agent(stateCnt, actionCnt)
########################################################################################################################
#agent.brain.model.load_weights("dqn_5c_3h_5p.h5")
#agent.brain.model_.load_weights("dqn_5c_3h_5p.h5")
#RandomAgent.memory.samples = np.load("data_5c_3h_5p.npy").tolist()
#agent.steps=np.load("step_5c_3h_5p.npy").tolist()
########################################################################################################################


########################################################################################################################
#drl=np.load("result_5c_3h_5p.npz")
#g_drl_th=drl['arr_0'].tolist()
#g_ga_th=drl['arr_1'].tolist()
#g_eql_th=drl['arr_2'].tolist()
#g_rnd_th=drl['arr_3'].tolist()
#g_wmmse_th=drl['arr_4'].tolist()

########################################################################################################################
randomAgent = RandomAgent(actionCnt)

init_env,no_use=env.env_reset(agent)

try:
    while randomAgent.memory.isFull() == False:
        env.run(randomAgent)


    agent.memory.samples = randomAgent.memory.samples
    randomAgent = None

    g_cntr=0
    g_data_x=[]
    g_data_y=[]

    ########################################################################################################################
    #g_cntr=np.load("counter_v2_5c_3h_5p_lr1.npy").tolist()
    #data_set = np.load("data_set_5c.npz")
    g_data_len = len(data_set['arr_0'])
    g_data_x = data_set['arr_0'].tolist()
    g_data_y = data_set['arr_1'].tolist()
    g_data_action_ga = data_set['arr_2'].tolist()
    g_data_ga_th = data_set['arr_3'].tolist()
    g_data_action_wmmse = data_set['arr_4'].tolist()
    g_data_wmmse_th = data_set['arr_5'].tolist()

    print(g_data_len,g_cntr)

    ########################################################################################################################

    while True:
        rward,actn= env.run(agent)
        sys.stdout.flush()
        check_ga_with_drl(actn)
        #print("total call vs corrected answer :", g_total_call,g_total_corrected)
finally:
    agent.brain.model.save("dqn_5c_3h_5p.h5")
    np.save("data_5c_3h_5p.npy", agent.memory.samples)

    print("total call vs corrected answer :", g_total_call,g_total_corrected)
