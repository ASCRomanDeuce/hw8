import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import random as npr
import copy

import two_helper


def foo(J, H, m, N_monte_carlo, NT):              # E(T), M(T), C(T)
    T = np.linspace(0.05, 7, NT)
    E = np.zeros(NT)
    M = np.zeros(NT)
    C = np.zeros(NT)

    for i in range(0, NT):
        res = two_helper.calculate(J, H, m, T[i], N_monte_carlo)
        E[i] = res["E"]
        M[i] = res["M"]
        C[i] = res["C"]
    
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(T, E)
    axs[1].plot(T, M)
    axs[2].plot(T, C)
    plt.show()

def venus(J, H, m, T, N_monte_carlo):
    res = two_helper.zabor(J, H, m, T, N_monte_carlo)

    plt.imshow(res["initial"])
    plt.show()
    plt.figure()
    plt.imshow(res["final"])
    plt.show()


venus(1.0, 0.0, 100, 1.6, 200000)
foo(1.0, 0.0, 70, 200000, 70)
