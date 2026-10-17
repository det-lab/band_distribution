import numpy as np
import ctypes
import os
import sys

folderpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.name == 'posix': #Linux/Mac
    DLLname = 'lib/libband_distribution.so'
else:
    print ('OS other than Linux/Mac are not supported')

api = np.ctypeslib.load_library(DLLname,folderpath)

failures = []


def check(name, fort_val, py_val, *, rtol=1e-9, atol=0.0):
    """Record and print a PASS/FAIL comparison of the vectorized and scalar results."""
    ok = bool(np.isclose(fort_val, py_val, rtol=rtol, atol=atol))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: vector={fort_val!r}  scalar={py_val!r}")
    if not ok:
        failures.append(name)


# Define all our parameters and variables
k = 0.18
Z = 32.0
F0 = 0.122
eps = 3E-3
V=3.0
p0=0.06421907
p10=0.48998486
q0=0.06421907
q10=0.48998486

Eq_arr = np.array([100.0, 100.0, 100.0], dtype=np.float64)
Ep_arr = np.array([347.0, 346.0, 348.0], dtype=np.float64)
n = Ep_arr.size

# Set argtypes and restype for the function PpqN_vector
api.PpqN_vector.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # Ep_arr
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # Eq_arr
    ctypes.c_int,              # n
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double,  # scalar params
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")   # res_arr (output)
]
api.PpqN_vector.restype = None

# Output array
res_arr = np.empty(n, dtype=np.float64)

# Call the Fortran vectorized function
api.PpqN_vector(Ep_arr, Eq_arr, n, k, Z, F0, eps, V, p0, p10, q0, q10, res_arr)

print("Vectorized result of PpqN_vector:", res_arr)

# Check each vectorized point against the scalar PpqN entrypoint for the
# same (Ep, Eq) -- the vectorized call must agree exactly with calling the
# scalar function once per point, since it's the same underlying evaluation
# just batched.
api.PpqN.argtypes = [ctypes.c_double] * 11
api.PpqN.restype = ctypes.c_double
for i in range(n):
    scalar_val = api.PpqN(ctypes.c_double(Ep_arr[i]), ctypes.c_double(Eq_arr[i]),
                          ctypes.c_double(k), ctypes.c_double(Z),
                          ctypes.c_double(F0), ctypes.c_double(eps),
                          ctypes.c_double(V), ctypes.c_double(p0), ctypes.c_double(p10),
                          ctypes.c_double(q0), ctypes.c_double(q10))
    check(f"PpqN_vector[{i}] vs PpqN", res_arr[i], scalar_val)

# Set argtypes and restype for the function PpqG_vector
api.PpqG_vector.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),      # Ep_arr
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),      # Eq_arr
    ctypes.c_int,                                                        # n
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # scalar params
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")       # res_arr (output)
]
api.PpqG_vector.restype = None

# Inputs
Eq_arr = np.array([100.0, 100.0, 100.0], dtype=np.float64)
Ep_arr = np.array([347.0, 346.0, 348.0], dtype=np.float64)
n = Ep_arr.size

# Output array
res_arr = np.empty(n, dtype=np.float64)

# Call the Fortran vectorized function
api.PpqG_vector(Ep_arr, Eq_arr, n, F0, eps, V, p0, p10, q0, q10, res_arr)

print("Vectorized result of PpqG_vector:", res_arr)

api.PpqG.argtypes = [ctypes.c_double] * 9
api.PpqG.restype = ctypes.c_double
for i in range(n):
    scalar_val = api.PpqG(ctypes.c_double(Ep_arr[i]), ctypes.c_double(Eq_arr[i]),
                          ctypes.c_double(F0), ctypes.c_double(eps),
                          ctypes.c_double(V), ctypes.c_double(p0), ctypes.c_double(p10),
                          ctypes.c_double(q0), ctypes.c_double(q10))
    check(f"PpqG_vector[{i}] vs PpqG", res_arr[i], scalar_val)

print("Parameters Are ####################")
print("Ep: ", Ep_arr)
print("Eq: ", Eq_arr)
print("k: ", k)
print("Z: ", Z)
print("F0: ", F0)
print("eps: ", eps)
print("V: ", V)
print("p0: ", p0)
print("p10: ", p10)
print("q0: ", q0)
print("q10: ", q10)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
