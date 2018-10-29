import numpy as np
import sys
from timeit import default_timer as time
from itertools import combinations


def calculate_kinetic_energy(p, m):
    return np.sum(np.sum(p**2, axis=1), axis=0)/2/m


def calculate_potential_energy(r, R, epsilon, L, f, l_vec, r_vec):
    norm_delta_r = np.linalg.norm(r[l_vec] - r[r_vec], axis=1)
    V_p = epsilon * (R ** 12 * np.sum(norm_delta_r ** -12, axis=0) - 2 * R ** 6 * np.sum(norm_delta_r ** -6, axis=0))
    r_i = np.linalg.norm(r, axis=1)
    r_i[r_i < L] = L
    V_s = f * (np.sum(r_i - L, axis=0)) ** 2 / 2

    return V_p + V_s


def calculate_pressure(F_s, L):
    return np.sum(np.sum(np.abs(F_s) ** 2, axis=-1) ** (1. / 2)) / (4 * np.pi * L ** 2)


def calculate_p_forces(r, R, epsilon, N):
    delta_r = r[l_arange_matrix] - r[r_arange_matrix]
    delta_norm_r_sqr = np.sum(delta_r ** 2, axis=2)
    delta_norm_r_sqr = delta_norm_r_sqr.reshape(delta_norm_r_sqr.shape[0], delta_norm_r_sqr.shape[1], 1)
    return 12 * epsilon * (R ** 12 * np.sum(delta_r * delta_norm_r_sqr ** -7, axis=0) - R ** 6 * np.sum(
        delta_r * delta_norm_r_sqr ** -4, axis=0))


def calculate_s_forces(r, N, L, f):
    r_i = np.linalg.norm(r, axis=1)
    r_i[r_i < L] = L
    s_force = (f * (L - r_i) / r_i).reshape((N, 1)) * r

    return s_force


def calculate_T(N, k, p, m):
    return np.sum(np.sum(p**2, axis=-1), axis=0)/3/m/N/k


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


def move(r, p, F, tau, m, f):
    p = p + F * tau / 2
    r = r + p * tau / m
    F_p = calculate_p_forces(r, R, epsilon, N)
    F_s = calculate_s_forces(r, N, L, f)
    F = F_p + F_s
    p = p + F * tau / 2

    return F_p, F_s, r, p


def remove_mass_center_move(p, N):
    P = p.sum(axis=0)
    return p - P / N


def set_p(N, k, t0):
    return np.concatenate((generate_p(N, k, t0), generate_p(N, k, t0), generate_p(N, k, t0)), axis=1)


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


def save_out(f_out, t, T, p, H):
    if f_out is None:
        print("t="+str(round(t, 3))+", T="+str(round(T, 4))+", p="+str(round(p, 4))+", H="+str(round(H, 4)))
    else:
        f_out.write("t="+str(round(t, 3))+", T="+str(round(T, 4))+", p="+str(round(p, 4))+", H="+str(round(H, 4)))


def save_xyz(f_xyz, r):
    f_xyz.write(str(r.shape[0]) + "\n")
    f_xyz.write("\n")
    for i in range(r.shape[0]):
        f_xyz.write("Ar" + " " + str(r[i][0]) + " " + str(r[i][1]) + " " + str(r[i][2]) + "\n")
    f_xyz.write("\n")


if __name__ == "__main__":
    np.random.seed(1)
    k = 8.31e-3
    n, m, epsilon, R, f, L, a, t0, tau, S_o, S_d, S_out, S_xyz = import_parameters()
    N = n ** 3

    potential_energy_vectors = combinations(np.arange(1, N+1), 2)
    l_pe_vec = []
    r_pe_vec = []
    for i in potential_energy_vectors:
        l_pe_vec.append(i[0])
        r_pe_vec.append(i[1])
    l_pe_vec = np.array(l_pe_vec) - 1
    r_pe_vec = np.array(r_pe_vec) - 1

    l_arange_matrix = np.zeros((N - 1, N), dtype=np.int)
    r_arange_matrix = np.zeros((N - 1, N), dtype=np.int)
    for i in range(N):
        arange = []
        for j in range(0, i):
            arange.append([i, j])
        for j in range(i + 1, N):
            arange.append([i, j])
        for j, ar in enumerate(arange):
            l_arange_matrix[j, i] = ar[0]
            r_arange_matrix[j, i] = ar[1]

    r = set_r(a, n)
    p = set_p(N, k, t0)
    p = remove_mass_center_move(p, N)
    V = calculate_potential_energy(r, R, epsilon, L, f, l_pe_vec, r_pe_vec)
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
    b = time()
    for i in range(S_o + S_d):
        F_p, F_s, r, p = move(r, p, F, tau, m, f)
        F = F_p + F_s
        pe = calculate_potential_energy(r, R, epsilon, L, f, l_pe_vec, r_pe_vec)
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

    e = time()
    t_loop = e - b
    f_xyz.close()
    if out != '-':
        f_out.close()

    print("Elapsed time {} seconds.".format(round(t_loop, 2)))
