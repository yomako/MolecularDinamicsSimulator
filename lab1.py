import numpy as np
import matplotlib.pyplot as plt
import sys
from pytictoc import TicToc
t = TicToc()


def set_r(a, n):
    r_matrix = []
    b0 = np.array([a, 0, 0])
    b1 = np.array([a / 2, a * np.sqrt(3) / 2, 0])
    b2 = np.array([a / 2, a * np.sqrt(3) / 6, a * np.sqrt(2 / 3)])
    for i0 in range(n):
        for i1 in range(n):
            for i2 in range(n):
                temp_r = (i0 - (n - 1) / 2) * b0 + (i1 - (n - 1) / 2) * b1 + (i2 - (n - 1) / 2) * b2
                r_matrix.append(temp_r)
    return np.array(r_matrix)


def generate_e_kin(N, k, t0):
    lambdas = np.random.rand(N, 1)
    e_kin = -k * t0 * np.log(lambdas) / 2
    return e_kin


def generate_p(N, k, t0):
    p_x = np.sqrt(2 * m * (generate_e_kin(N, k, t0)))
    pm = np.random.randint(2, size=N)
    pm[pm == 0] = -1
    p_x = np.multiply(p_x.T, pm)
    return p_x.T


def set_p(N, k, t0):
    return np.concatenate((generate_p(N, k, t0), generate_p(N, k, t0), generate_p(N, k, t0)), axis=1)


def remove_mass_center_move(p, N):
    P = p.sum(axis=0)
    return p - P / N


def calculate_potential_energy(r, R, epsilon, N, L, f):
    n_array = np.arange(N)
    nm1_array = np.arange(N - 1)
    cpespVec = np.vectorize(calculate_potential_energy_sub_p, excluded=['r', 'R', 'epsilon', 'N'], otypes=[np.float])
    cpessVec = np.vectorize(calculate_potential_energy_sub_s, excluded=['r', 'L', 'f'], otypes=[np.float])
    V_p = np.sum(cpespVec(nm1_array, r=r, R=R, epsilon=epsilon, N=N), axis=0)
    V_s = np.sum(cpessVec(n_array, r=r, L=L, f=f), axis=0)

    return V_p + V_s


def calculate_p_forces(r, R, epsilon, N):
    n_array = np.arange(N)
    cpfVec = np.vectorize(calculate_p_force, excluded=['r', 'R', 'epsilon', 'N'], otypes=[np.float, np.float, np.float])
    force_p_matrix = cpfVec(n_array, r=r, R=R, epsilon=epsilon, N=N)
    return np.array(force_p_matrix).T


def calculate_s_forces(r, N, L, f):
    force_s_matrix = []
    for i in range(N):
        temp_force = np.array([0, 0, 0])
        r_i = np.linalg.norm(r[i])
        if r_i >= L:
            temp_force = f * (L - r_i) / r_i * r[i]

        force_s_matrix.append(temp_force)

    return np.array(force_s_matrix)


def calculate_pressure(F_s, L):
    return np.sum(np.sum(np.abs(F_s) ** 2, axis=-1) ** (1. / 2)) / (4 * np.pi * L ** 2)


def calculate_p_force(i, r, R, epsilon, N):
    delta_R = (np.array([r[i], ] * (N - 1)) - np.concatenate((r[0:i], r[(i + 1):])))
    ret = 12*epsilon*np.sum((R**12*np.sum(delta_R**2, axis=1).reshape((N-1, 1))**(-7)-R**6*np.sum(delta_R**2, axis=1).reshape((N-1, 1))**(-4))*delta_R, axis=0)
    return ret[0], ret[1], ret[2]


def calculate_potential_energy_sub_p(i, r, R, epsilon, N):
    return epsilon * (R ** 12 * np.sum(
            np.sum(np.abs(np.array([r[i], ] * (N - 1 - i)) - r[(i + 1):]) ** 2, axis=-1) ** (-6)) - 2 * R ** 6 * np.sum(
            np.sum(np.abs(np.array([r[i], ] * (N - 1 - i)) - r[(i + 1):]) ** 2, axis=-1) ** (-3)))


def calculate_potential_energy_sub_s(i, r, L, f):
    r_i = np.linalg.norm(r[i])
    if r_i >= L:
        return f * (r_i - L) ** 2 / 2
    else:
        return 0

   
def export_data(r):
    f = open("avs.txt", "w")
    f.write(str(r.shape[0])+"\n")
    f.write("\n")
    for i in range(r.shape[0]):
        f.write("Ar"+" "+str(r[i][0])+" "+str(r[i][1])+" "+str(r[i][2])+"\n")
    f.close()


def move(r, p, F, tau, m, f):
    for i in range(r.shape[0]):
        p[i] = p[i] + F[i]*tau/2
        r[i] = r[i] + p[i]*tau/m
    F_p = calculate_p_forces(r, R, epsilon, N)
    F_s = calculate_s_forces(r, N, L, f)
    F = F_p + F_s
    for i in range(r.shape[0]):
        p[i] = p[i] + F[i]*tau/2

    return F_p, F_s


def save_xyz(f_xyz, r):
    f_xyz.write(str(r.shape[0]) + "\n")
    f_xyz.write("\n")
    for i in range(r.shape[0]):
        f_xyz.write("Ar" + " " + str(r[i][0]) + " " + str(r[i][1]) + " " + str(r[i][2]) + "\n")
    f_xyz.write("\n")


def calculate_T(N, k, p, m):
    return np.sum(np.sum(p**2, axis=-1), axis=0)/3/m/N/k


def import_parameters():
    #lst = open(str(sys.argv[1]), "r").readlines()
    lst = open("parametry_52.txt", "r").readlines()
    n = int(lst[0].split('\t')[0])
    m = float(lst[1].split('\t')[0])
    epsilon = int(lst[2].split('\t')[0])
    R = float(lst[3].split('\t')[0])
    f = float(lst[4].split('\t')[0])
    L = float(lst[5].split('\t')[0])
    a = float(lst[6].split('\t')[0])
    t0 = int(lst[7].split('\t')[0])
    tau = float(lst[8].split('\t')[0])
    S_o = int(lst[9].split('\t')[0])
    S_d = int(lst[10].split('\t')[0])
    S_out = int(lst[11].split('\t')[0])
    S_xyz = int(lst[12].split('\t')[0])
    return n, m, epsilon, R, f, L, a, t0, tau, S_o, S_d, S_out, S_xyz
    
    
def calculate_kinetic_energy(p, m):
    return np.sum(np.sum(p**2, axis=1), axis=0)/2/m
    
    
def save_out(f_out, t, T, p, H):
    if f_out is None:
        print("t="+str(round(t, 3))+", T="+str(round(T, 4))+", p="+str(round(p, 4))+", H="+str(round(H, 4)))
    else:
        f_out.write("t="+str(round(t, 3))+", T="+str(round(T, 4))+", p="+str(round(p, 4))+", H="+str(round(H, 4)))


if __name__ == "__main__":
    np.random.seed(1)
    k = 8.31e-3
    n, m, epsilon, R, f, L, a, t0, tau, S_o, S_d, S_out, S_xyz = import_parameters()
    N = n ** 3
    r = set_r(a, n)
    p = set_p(N, k, t0)
    p = remove_mass_center_move(p, N)
    V = calculate_potential_energy(r, R, epsilon, N, L, f)
    F_p = calculate_p_forces(r, R, epsilon, N)
    #print(F_p) co 100 krokow polozenia, co 10 energie?
    F_s = calculate_s_forces(r, N, L, f)
    pressure = calculate_pressure(F_s, L)
    F = F_p + F_s
    T_a = []
    t_a = []
    p_a = []
    H_a = []
    f_xyz = open("avs.txt", "w")
    f_xyz.close()
    f_xyz = open("avs.txt", "a")
    #out = str(sys.argv[2])
    out = '-'
    f_out = None
    if out != '-':
        f_out = open(out, "w")
    t.tic()
    for i in range(S_o + S_d):
        F_p, F_s = move(r, p, F, tau, m, f)
        F = F_p + F_s
        pe = calculate_potential_energy(r, R, epsilon, N, L, f)
        ke = calculate_kinetic_energy(p, m)
        if i > S_o:
            T_a.append(calculate_T(N, k, p, m))
            p_a.append(calculate_pressure(F_s, L))
            H_a.append(pe + ke)
            t_a.append((i+1-S_o)*tau)
        if i > S_o and (i + S_o) % S_xyz == 0:
            save_xyz(f_xyz, r)
        if i > S_o and (i + S_o) % S_out == 0:
            save_out(f_out, t_a[-1], T_a[-1], p_a[-1], H_a[-1])
    t.toc()
    f_xyz.close()
    if out != '-':
        f_out.close()
    #plt.figure(num=2)
    #plt.plot(t_a, H_a)
    #plt.title("Wykres energii potencjalnej w funkcji czasu")
    #plt.show()
