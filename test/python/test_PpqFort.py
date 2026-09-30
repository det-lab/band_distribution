import numpy as np
import ctypes
import sys
import os

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path to the directory containing pq_dist_v10.py
module_dir = os.path.join(repo_root, 'python')
sys.path.append(module_dir)

# pq_dist_v10 contains the Python implementation with the numerical N integral
# (21-point Simpson's rule, sigp/sigq at noiseless energies), matching Fortran PpqN.
import pq_dist_v10 as ppq

folderpath = repo_root
if os.name == 'posix': #Linux/Mac
    DLLname = 'lib/libband_distribution.so'
else:
    print ('OS other than Linux/Mac are not supported')

api = np.ctypeslib.load_library(DLLname,folderpath)

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

Er = 234.0
Ep = 337.5
Eq = 100

# Define argument and return types for the function Y
api.Y.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
api.Y.restype = ctypes.c_double

result_fort = api.Y(ctypes.c_double(Er), ctypes.c_double(k), ctypes.c_double(Z))
result_py = ppq.Y(Er, k=k, Z=Z)
print('back in python after running function Y')
print('The fortran result is ', result_fort)
print('The python result is ', result_py)

# Define argument and return types for the function Nbar
api.Nbar.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
api.Nbar.restype = ctypes.c_double

result_fort = api.Nbar(ctypes.c_double(Er), ctypes.c_double(k), ctypes.c_double(Z), ctypes.c_double(eps))
result_py = ppq.Nbar(Er, k=k, Z=Z, eps=eps, is_gamma=False)
print('back in python after running function Nbar')
print('The fortran result is ', result_fort)
print('The python result is ', result_py)

# Define argument and return types for the function sigp
api.sigp.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
api.sigp.restype = ctypes.c_double

result_fort = api.sigp(ctypes.c_double(Ep), ctypes.c_double(eps), ctypes.c_double(V), 
                       ctypes.c_double(p0), ctypes.c_double(p10))
result_py = ppq.sigp(Ep, eps=eps, V=V, p0=p0, p10=p10)
print('back in python after running function sigp')
print('The fortran result is ', result_fort)
print('The python result is ', result_py)

# Define argument and return types for the function sigq
api.sigq.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
api.sigq.restype = ctypes.c_double

result_fort = api.sigq(ctypes.c_double(Eq), ctypes.c_double(q0), ctypes.c_double(q10))
result_py = ppq.sigq(Eq, q0=q0, q10=q10)
print('back in python after running function sigq')
print('The fortran result is ', result_fort)
print('The python result is ', result_py)

# Define argument and return types for the function PErN
api.PErN.argtypes = [ctypes.c_double]
api.PErN.restype = ctypes.c_double

result_fort = api.PErN(ctypes.c_double(Er))
result_py = ppq.PErN(Er)
print('back in python after running function PErN')
print('The fortran result is ', result_fort)
print('The python result is ', result_py)

# Define argument and return types for the function PErG
api.PErG.argtypes = [ctypes.c_double]
api.PErG.restype = ctypes.c_double

result_fort = api.PErG(ctypes.c_double(Er))
result_py = ppq.PErG(Er)
print('back in python after running function PErG')
print('The fortran result is ', result_fort)
print('The python result is ', result_py)

# Define argument and return types for the function PpqFullN
api.PpqFullN.argtypes = [ctypes.c_double] * 12
api.PpqFullN.restype = ctypes.c_double

result_fort = api.PpqFullN(ctypes.c_double(Er), ctypes.c_double(Ep), ctypes.c_double(Eq),
                           ctypes.c_double(k), ctypes.c_double(Z),
                           ctypes.c_double(F0), ctypes.c_double(eps),
                           ctypes.c_double(V), ctypes.c_double(p0), ctypes.c_double(p10),
                           ctypes.c_double(q0), ctypes.c_double(q10))
result_py = ppq.PpqFullN(Er, Ep, Eq, k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)
print('back in python after running function PpqFullN')
print('The fortran result is ', result_fort)
print('The python result is ', result_py)

# Define argument and return types for the function PpqN
api.PpqN.argtypes = [ctypes.c_double] * 11
api.PpqN.restype = ctypes.c_double

result_fort = api.PpqN(ctypes.c_double(Ep), ctypes.c_double(Eq),
                       ctypes.c_double(k), ctypes.c_double(Z),
                       ctypes.c_double(F0), ctypes.c_double(eps),
                       ctypes.c_double(V), ctypes.c_double(p0), ctypes.c_double(p10),
                       ctypes.c_double(q0), ctypes.c_double(q10))
(result_py, _), _, _ = ppq.PpqN_safe_inspect(Ep, Eq, k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10, res=0.1)
print('back in python after running function PpqN')
print('The fortran result is ', result_fort)
print('The python result is ', result_py)

# Define argument and return types for the function PpqG
api.PpqG.argtypes = [ctypes.c_double] * 9
api.PpqG.restype = ctypes.c_double

result_fort = api.PpqG(ctypes.c_double(Ep), ctypes.c_double(Eq),
                       ctypes.c_double(F0), ctypes.c_double(eps),
                       ctypes.c_double(V), ctypes.c_double(p0), ctypes.c_double(p10),
                       ctypes.c_double(q0), ctypes.c_double(q10))
(result_py, _), _, _ = ppq.PpqG_safe_inspect(Ep, Eq, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10, res=0.01)
print('back in python after running function PpqG')
print('The fortran result is ', result_fort)
print('The python result is ', result_py)

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

# Inputs
Eq_arr = np.array([100.0, 100.0, 100.0], dtype=np.float64)
Ep_arr = np.array([347.0, 346.0, 348.0], dtype=np.float64)
n = Ep_arr.size

# Output array
res_arr = np.empty(n, dtype=np.float64)

# Call the Fortran vectorized function
api.PpqN_vector(Ep_arr, Eq_arr, n, k, Z, F0, eps, V, p0, p10, q0, q10, res_arr)

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

print("Vectorized result:", res_arr)

# Package version
api.PpqFort_version.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3
api.PpqFort_version.restype = None
vmaj, vmin, vpatch = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
api.PpqFort_version(ctypes.byref(vmaj), ctypes.byref(vmin), ctypes.byref(vpatch))
print("Fortran version: ", (vmaj.value, vmin.value, vpatch.value))
print("Python version: ", ppq.__version__)

print("Parameters Are ####################")
print("Er: ", Er)
print("Ep: ", Ep)
print("Eq: ", Eq)
print("k: ", k)
print("Z: ", Z)
print("F0: ", F0)
print("eps: ", eps)
print("V: ", V)
print("p0: ", p0)
print("p10: ", p10)
print("q0: ", q0)
print("q10: ", q10)
