import numpy as np
import random
cimport numpy as np
cimport cython

from libc.math cimport exp
# from cython.cimports.libc.stdlib import malloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free


cdef double S(int i, int j, int** n, int length):
    if (i < 0):
        i = i + length
    elif (i >= length):
        i = i - length
    if (j < 0):
        j = j + length
    elif (j >= length):
        j = j - length
    
    return n[i][j]


cpdef calculate(double J, double H_field, int length, double T, int N_monte_carlo):

    cdef int** n = <int**> PyMem_Malloc(
            length * sizeof(int*))
    cdef int i = 0
    for i in range(0, length):
        n[i] = <int*> PyMem_Malloc(
            length * sizeof(int)
        )
    if not n:
        raise MemoryError()
    

    cdef:
        double E_now = 0.0
        double dE = 0.0
        double E_averaged = 0.0
        double E_square_averaged = 0.0

        double M_now = 0.0
        double dM = 0.0
        double M_averaged = 0.0

        double C = 0.0

        int j = 0
        double xi = 0.0

    
    for i in range(0, length):           # Начальное состояние, все спины вверх
        for j in range(0, length):
            n[i][j] = 1
    
    E_now = 0.0                    # "нулевая энергия"
    for i in range(0, length):
        for j in range(0, length):
            E_now += -J * n[i][j] * (S(i + 1, j, n, length) + S(i, j + 1, n, length)) - H_field * n[i][j]
    
    E_averaged = E_now
    E_square_now = E_now**2
    E_square_averaged = E_square_now
    

    M_now = 0.0                    # "нулевой момент"
    for i in range(0, length):
        for j in range(0, length):
            M_now += n[i][j]

    M_now /= length**2.0
    M_averaged = M_now

    for i in range(2, N_monte_carlo):

        j1 = random.randint(0, length-1)
        j2 = random.randint(0, length-1)

        #Изменение энергии при перевороте спина № (j1,j2)
        dE = 2.0 * n[j1][j2] * ( H_field + J * (
                S(j1, j2-1, n, length) + S(j1, j2 + 1, n, length) +
                S(j1 - 1, j2, n, length) + S(j1 + 1, j2, n, length)
            )
        )

        dM = ( - 2.0 * n[j1][j2] / (length**2.0) )        #Изменение момента при перевороте спина № (j1,j2)

        # Алгоритм Метрополиса
        if (dE < 0):
            n[j1][j2] = n[j1][j2] * (-1)
            E_now = E_now + dE
            M_now = M_now + dM
            E_square_now = E_now**2
        else:
            xi = random.random()
            if (xi < exp(-dE / T)):
                n[j1][j2] = n[j1][j2] * (-1)
                E_now = E_now + dE
                M_now = M_now + dM
                E_square_now = E_now**2
        
        E_averaged = E_averaged * (i-1) / i + E_now / i 
        M_averaged = M_averaged * (i-1) / i + M_now / i 
        E_square_averaged = E_square_averaged * (i-1) / i + E_square_now / i
    
    C = (E_square_averaged - E_averaged**2) / length / (T**2)

    return {"E": E_averaged, "M": M_averaged, "C": C}



cpdef zabor(double J, double H_field, int length, double T, int N_monte_carlo):
    cdef int** n = <int**> PyMem_Malloc(
        length * sizeof(int*))
    cdef int** n_first = <int**> PyMem_Malloc(
        length * sizeof(int*)
    )
    cdef int i = 0
    cdef int j = 0
    for i in range(0, length):
        n[i] = <int*> PyMem_Malloc(
            length * sizeof(int)
        )
        n_first[i] = <int*> PyMem_Malloc(
            length * sizeof(int)
        )
    
    # Случайная конфигурация
    for i in range(0, length):
        for j in range(0, length):
            n[i][j] = random.randint(0, 1)
            if (n[i][j] == 0):
                n[i][j] = -1
            n_first[i][j] = n[i][j]
    

    cdef:
        int j1 = 0
        int j2 = 0
        double dE = 0.0
    
    for i in range(2, N_monte_carlo):

        j1 = random.randint(0, length-1)
        j2 = random.randint(0, length-1)

        #Изменение энергии при перевороте спина № (j1,j2)
        dE = 2.0 * n[j1][j2] * ( H_field + J * (
                S(j1, j2-1, n, length) + S(j1, j2 + 1, n, length) +
                S(j1 - 1, j2, n, length) + S(j1 + 1, j2, n, length)
            )
        )

        # Алгоритм Метрополиса
        if (dE < 0):
            n[j1][j2] = n[j1][j2] * (-1)
        else:
            xi = random.random()
            if (xi < exp(-dE / T)):
                n[j1][j2] = n[j1][j2] * (-1)
    
    n_first_np = np.zeros((length, length), dtype=int)
    n_last_np = np.zeros((length, length), dtype=int)
    for i in range(0, length):
        for j in range(0, length):
            n_first_np[i, j] = n_first[i][j]
            n_last_np[i, j] = n[i][j]
    return {"initial": n_first_np, "final": n_last_np}