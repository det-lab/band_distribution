import numpy as np
import scipy.integrate as integrate
from scipy.optimize import curve_fit

__version__ = "2.0.0"

#the ionization yield: the Lindhard model.
#Y(Er) = k*g(eps_L) / (1 + k*g(eps_L)), with
#eps_L = 11.5*Er*Z**(-7/3) and g(eps_L) = 3*eps_L**0.15 + 0.7*eps_L**0.29 + eps_L
def Y(Er,
	*,
	k, Z):
    #Er should be in keV; Z is the target's atomic number
    eps_L = 11.5 * np.absolute(Er) * Z**(-7.0/3.0)
    g = 3.0*eps_L**0.15 + 0.7*eps_L**0.29 + eps_L
    return k*g / (1.0 + k*g)

#average numbers of e/h pairs, for either band.  is_gamma selects the
#electron-recoil yield (Y = 1) instead of the Lindhard NR yield
#Y(Er,k=k,Z=Z); k, Z are unused (pass any value) when is_gamma=True.
def Nbar(Er,
	  *,
	  k, Z,
	  eps,
	  is_gamma):
    yield_val = 1.0 if is_gamma else Y(Er,k=k,Z=Z)
    return yield_val*Er/eps

#phonon and ionization resolutions
def sigp(Ep,
	  *,
	  eps,
	  V,
	  p0, p10):
    e2pre = (p10**2 - p0**2)
    e2e = (Ep/(10*(1+(V/eps/1000))))**2
    #e2e = (Ep/(10*((V/eps/1000))))**2 #THIS ERROR WAS IN pq_dist v1!! it makes a difference!
    return np.sqrt(p0**2+e2pre*e2e)

def sigq(Eq,
	  *,
	  q0, q10):
    e2pre = (q10**2 - q0**2)
    e2e = (Eq/10)**2
    return np.sqrt(q0**2+e2pre*e2e)

#probability distribution of _true_ recoil energy, separate for gammas, Cfgammas or neutrons
#fit parameters -- matches the Fortran PErN_impl constants exactly (not
#exposed as arguments there either: PErN(Er) is the public signature)
_PNa, _PNb, _PNd = 0.53693208, 6.41515782, 23.71789286

def PErN(Er):
    Er = np.asarray(Er)  # Convert input to a NumPy array
    result = np.where(Er < 0, 0, _PNa * (1 / _PNb) * np.exp(-Er / _PNb) + (1 - _PNa) * (1 / _PNd) * np.exp(-Er / _PNd))
    return result
#...TBD gammas

# Cfgammas -- matches the Fortran PErG_impl constants exactly
_PGa, _PGb, _PGd = 5.73211975e-01, 1.69520023e-01, 2.79552394e+02

def PErG(Er):
    Er = np.asarray(Er)  # Convert Er to a NumPy array
    result = np.where(Er < 0, 0, _PGa * (1 / _PGb) * np.exp(-Er / _PGb) + (1 - _PGa) * (1 / _PGd) * np.exp(-Er / _PGd))
    return result

def aN(Er,Ep,Eq,
	      *,
	      F0,
	      eps,
	      V,
	      p0, p10,
	      q0, q10):
    t1 = (V/1e3)*(Ep-Er)/sigp(Ep,eps=eps,V=V,p0=p0,p10=p10)**2
    t2 = eps*Eq/sigq(Eq,q0=q0,q10=q10)**2
    t3 = 1/F0
    return t1+t2+t3

#is_gamma selects the electron-recoil yield (Y=1) instead of the Lindhard
#NR yield Y(Er,k=k,Z=Z); k, Z are unused (pass any value) when is_gamma=True.
def bN(Er,Ep,Eq,
	      *,
	      k,Z,
	      F0,
	      eps,
	      V,
	      p0, p10,
	      q0, q10,
	      is_gamma):
    Nbar_val = Nbar(Er,k=k,Z=Z,eps=eps,is_gamma=is_gamma)
    t1 = 1/(2*Nbar_val*F0)
    t2 = eps**2/(2*sigq(Eq,q0=q0,q10=q10)**2)
    t3 = V**2/(2*(sigp(Ep,eps=eps,V=V,p0=p0,p10=p10)*1e3)**2)
    return t1+t2+t3

def cN(Er,Ep,Eq,
	      *,
	      k,Z,
	      F0,
	      eps,
	      V,
	      p0, p10,
	      q0, q10,
	      is_gamma):
    Nbar_val = Nbar(Er,k=k,Z=Z,eps=eps,is_gamma=is_gamma)
    t1 = -(Ep-Er)**2/(2*sigp(Ep,eps=eps,V=V,p0=p0,p10=p10)**2)
    t2 = -Eq**2/(2*sigq(Eq,q0=q0,q10=q10)**2)
    t3 = -Nbar_val/(2*F0)
    return t1+t2+t3

#is_gamma selects the electron-recoil yield (Y=1) instead of the Lindhard
#NR yield Y(Er,k=k,Z=Z); k, Z are unused (pass any value) when is_gamma=True.
def PpqExp(Er,Ep,Eq,
	          *,
	          k,Z,
	          F0,
	          eps,
	          V,
	          p0, p10,
	          q0, q10,
	          is_gamma):
    cN_val = cN(Er,Ep,Eq,k=k,Z=Z,F0=F0,eps=eps,V=V,p0=p0,p10=p10,q0=q0,q10=q10,is_gamma=is_gamma)
    aN_val = aN(Er,Ep,Eq,F0=F0,eps=eps,V=V,p0=p0,p10=p10,q0=q0,q10=q10)
    bN_val = bN(Er,Ep,Eq,k=k,Z=Z,F0=F0,eps=eps,V=V,p0=p0,p10=p10,q0=q0,q10=q10,is_gamma=is_gamma)
    exponent = cN_val + (aN_val**2 / (4*bN_val))
    return exponent

def PpqFullN(Er, Ep, Eq,
             *,
             k, Z,
             F0,
             eps,
             V,
             p0, p10,
             q0, q10):
    F_val    = F0
    Nbar_val = Nbar(Er, k=k, Z=Z, eps=eps, is_gamma=False)
    if Nbar_val <= 0.0 or F_val <= 0.0:
        return 0.0
    sigma_N    = np.sqrt(Nbar_val * F_val)
    norm_const = 1.0 / (2.0 * np.pi) ** 1.5
    sp0 = p0 ** 2
    sp1 = (p10 ** 2 - p0 ** 2) / (10.0 * (1.0 + V / (eps * 1.0e3))) ** 2
    sq0 = q0 ** 2
    sq1 = (q10 ** 2 - q0 ** 2) / 100.0
    EP_mean      = Er + (V * 1.0e-3) * Nbar_val
    EQ_mean      = eps * Nbar_val
    sigp_mean_sq = sp0 + sp1 * EP_mean ** 2
    sigq_mean_sq = sq0 + sq1 * EQ_mean ** 2
    a_coeff = ((V * 1.0e-3) * (Ep - Er) / sigp_mean_sq
               + eps * Eq / sigq_mean_sq
               + 1.0 / F_val)
    b_coeff = (1.0 / (2.0 * Nbar_val * F_val)
               + eps ** 2 / (2.0 * sigq_mean_sq)
               + V ** 2 / (2.0 * 1.0e6 * sigp_mean_sq))
    N_star     = a_coeff / (2.0 * b_coeff)
    sigma_Neff = 1.0 / np.sqrt(2.0 * b_coeff)
    if N_star - 5.0 * sigma_Neff < 0.0:
        N_lo, N_hi = 0.0, 10.0 * sigma_Neff
    else:
        N_lo, N_hi = N_star - 5.0 * sigma_Neff, N_star + 5.0 * sigma_Neff
    N_pts = np.linspace(N_lo, N_hi, 21)
    h_N   = (N_hi - N_lo) / 20.0
    EP_nl = Er + (V * 1.0e-3) * N_pts
    EQ_nl = eps * N_pts
    sig_p = np.sqrt(sp0 + sp1 * EP_nl ** 2)
    sig_q = np.sqrt(sq0 + sq1 * EQ_nl ** 2)
    exp_arg = (-0.5 * ((Ep - EP_nl) / sig_p) ** 2
               - 0.5 * ((Eq - EQ_nl) / sig_q) ** 2
               - 0.5 * ((N_pts - Nbar_val) / sigma_N) ** 2)
    integrand = np.zeros(21)
    mask = exp_arg >= -700.0
    integrand[mask] = (norm_const / (sigma_N * sig_p[mask] * sig_q[mask])
                       * np.exp(exp_arg[mask]))
    simps_w = np.array([1,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,1], dtype=float)
    return float((h_N / 3.0) * np.dot(simps_w, integrand) * PErN(Er))

def PpqN_safe_inspect_vec(Ep, Eq,
              *,
              k, Z,
              F0,
              eps,
              V,
              p0, p10,
              q0, q10,
              res):
    ppqNArr = []
    for this_Ep, this_Eq in zip(Ep, Eq):
        PpqNval, _, _ = PpqN_safe_inspect(this_Ep, this_Eq, k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10, res=res)
        ppqNArr.append(PpqNval[0])
    return ppqNArr

"""
Usage to just get the value of PpqN:
(PpqN, _), _, _ = PpqN_safe_inspect(args)
"""
def PpqN_safe_inspect(Ep, Eq,
              *,
              k, Z,
              F0,
              eps,
              V,
              p0, p10,
              q0, q10,
              res):
    if F0 == 0:
        raise ValueError("Fano factor F(Er) = F0 is zero for all Er")

    # find Er_max
    f = lambda er,ep,eq: -1*PpqExp(er,ep,eq,k=k,Z=Z,F0=F0,eps=eps,V=V,p0=p0,p10=p10,q0=q0,q10=q10,is_gamma=False)

    # WARNING
    # this will only work if your peaks have a width
    # greater than 0.1 keV
    # which seems to be true for all reasonable CDMS parameters
    # the advantage is that the minimizer sometimes fails and this never does
    maxx = max(Ep, Eq) + 5
    er_arr = np.arange(5e-21, maxx, res)
    f_arr = np.zeros(len(er_arr), dtype=np.float64)
    for idx, er in enumerate(er_arr):
        #print (f(er, Ep, Eq), type(f(er, Ep, Eq)))
        try:
            f_arr[idx] = -1*PpqExp(er,Ep,Eq,k=k,Z=Z,F0=F0,eps=eps,V=V,p0=p0,p10=p10,q0=q0,q10=q10,is_gamma=False)
        except Exception as e:
            pass
            # print (er, Ep, Eq)
            # print (-1*PpqExp(er,Ep,Eq,k=k,Z=Z,F0=F0,eps=eps,V=V,p0=p0,p10=p10,q0=q0,q10=q10,is_gamma=False))
    real_val_idx = np.where(f_arr > 0)[0]
    if len(real_val_idx) == 0:
        if np.all(f_arr == 0):
            # print ("we really couldn't find any real values, there is likely no peak")
            return (0, 0), (er_arr, f_arr), []
        elif np.all(np.isnan(f_arr)):
            # print ("all the evaluated values are NaN!")
            return (np.nan, np.nan), (er_arr, f_arr), []
    else:
        min_idx = np.argmin(f_arr[~np.isnan(f_arr)])
        Ermx = er_arr[min_idx]
        # print ("We found a peak at ", Ermx, " in the exponent term")

    # Define the function (replace PpqFullG with your actual implementation)
    g = lambda er, ep, eq: PpqFullN(er, ep, eq, k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)

    return integrate_g_safe_inspect(g, Ep, Eq, Ermx, res)

def PpqFullG(Er, Ep, Eq,
             *,
             F0,
             eps,
             V,
             p0, p10,
             q0, q10):
    F_val    = F0
    Nbar_val = Nbar(Er, k=0.0, Z=1.0, eps=eps, is_gamma=True)   # k, Z unused: Y=1 for electron recoils
    if Nbar_val <= 0.0 or F_val <= 0.0:
        return 0.0
    sigma_N    = np.sqrt(Nbar_val * F_val)
    norm_const = 1.0 / (2.0 * np.pi) ** 1.5
    sp0 = p0 ** 2
    sp1 = (p10 ** 2 - p0 ** 2) / (10.0 * (1.0 + V / (eps * 1.0e3))) ** 2
    sq0 = q0 ** 2
    sq1 = (q10 ** 2 - q0 ** 2) / 100.0
    EP_mean      = Er + (V * 1.0e-3) * Nbar_val
    EQ_mean      = eps * Nbar_val
    sigp_mean_sq = sp0 + sp1 * EP_mean ** 2
    sigq_mean_sq = sq0 + sq1 * EQ_mean ** 2
    a_coeff = ((V * 1.0e-3) * (Ep - Er) / sigp_mean_sq
               + eps * Eq / sigq_mean_sq
               + 1.0 / F_val)
    b_coeff = (1.0 / (2.0 * Nbar_val * F_val)
               + eps ** 2 / (2.0 * sigq_mean_sq)
               + V ** 2 / (2.0 * 1.0e6 * sigp_mean_sq))
    N_star     = a_coeff / (2.0 * b_coeff)
    sigma_Neff = 1.0 / np.sqrt(2.0 * b_coeff)
    if N_star - 5.0 * sigma_Neff < 0.0:
        N_lo, N_hi = 0.0, 10.0 * sigma_Neff
    else:
        N_lo, N_hi = N_star - 5.0 * sigma_Neff, N_star + 5.0 * sigma_Neff
    N_pts = np.linspace(N_lo, N_hi, 21)
    h_N   = (N_hi - N_lo) / 20.0
    EP_nl = Er + (V * 1.0e-3) * N_pts
    EQ_nl = eps * N_pts
    sig_p = np.sqrt(sp0 + sp1 * EP_nl ** 2)
    sig_q = np.sqrt(sq0 + sq1 * EQ_nl ** 2)
    exp_arg = (-0.5 * ((Ep - EP_nl) / sig_p) ** 2
               - 0.5 * ((Eq - EQ_nl) / sig_q) ** 2
               - 0.5 * ((N_pts - Nbar_val) / sigma_N) ** 2)
    integrand = np.zeros(21)
    mask = exp_arg >= -700.0
    integrand[mask] = (norm_const / (sigma_N * sig_p[mask] * sig_q[mask])
                       * np.exp(exp_arg[mask]))
    simps_w = np.array([1,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,1], dtype=float)
    return float((h_N / 3.0) * np.dot(simps_w, integrand) * PErG(Er))

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

# Define Gaussian function
def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def integrate_g_safe_inspect(g, Ep, Eq, Ermx, resolution):
    minx = 5e-21
    maxx = max(Ep, Eq) + 5
    # print("minx and maxx are ", minx, maxx)

    # Amy thinks the following block might be dead code that could be deleted
    # it came from when we were trying to estimate Ermx,
    # but were doing so poorly
    # so we were seeing all zeros because we weren't near the maximum
    # Test for all zeros
    if g(Ermx, Ep, Eq) == 0 or np.isnan(g(Ermx, Ep, Eq)):
        # print ("The maximum value is smaller than machine precision")
        # print ("The value of the integrand is effectively zero")
        return (0, 0), (None, None), []
    else:
        pass
        # print ("The maximum value is ", g(Ermx, Ep, Eq))

    # Scan the integrand on a uniform grid.  If it is still non-zero
    # near the end of the scan range, extend the range and re-scan the
    # whole grid, so that all group indices below refer to one
    # consistent array.
    scan_min = minx
    while True:
        er_arr = np.arange(scan_min, maxx, resolution / 2)
        er_integrand_arr = np.zeros(np.size(er_arr))
        for idx, er in enumerate(er_arr):
            er_integrand_arr[idx] = g(er, Ep, Eq)
        nonZero_idx = np.where(er_integrand_arr > 0)[0]
        if len(nonZero_idx) == 0 or nonZero_idx[-1] + 5 < len(er_arr) - 1:
            break
        maxx = 1.5 * maxx

    if len(nonZero_idx) == 0:
        # the peak is narrower than the scan step and fell between grid
        # points; integrate around the known maximum instead
        val, err = integrate.quad(g, er_arr[0], er_arr[-1], args=(Ep, Eq),
                                  points=[Ermx], epsabs=0, epsrel=1e-12,
                                  limit=200)
        peak_info = {"minx": er_arr[0], "maxx": er_arr[-1],
                     "integral": val, "error": err,
                     "width": np.nan, "Ermx": Ermx, "integrand_max": np.nan}
        return (val, err), (er_arr, er_integrand_arr), [peak_info]

    # We expect a structure of peaks surrounded by zeros: group the
    # consecutive non-zero samples and pad each group's range by 5 grid
    # points on both sides.
    nonZero_idx_groups = consecutive(nonZero_idx)

    peak_info_array = []
    for peak_idx in nonZero_idx_groups:
        min_idx = max(0, min(peak_idx) - 5)
        max_idx = min(np.size(er_integrand_arr) - 1, max(peak_idx) + 5)
        peak_info_array.append({"minx": er_arr[min_idx],
                                "maxx": er_arr[max_idx]})

    # Merge overlapping ranges (union), so that quad below neither
    # double-counts the overlap nor drops a peak.  The groups are in
    # ascending Er order by construction.
    merged = [peak_info_array[0]]
    for peak_info in peak_info_array[1:]:
        if peak_info["minx"] <= merged[-1]["maxx"]:
            merged[-1]["maxx"] = max(merged[-1]["maxx"], peak_info["maxx"])
        else:
            merged.append(peak_info)
    peak_info_array = merged

    # Integrate using quad now that you have accurate limits
    for jdx, peak_info in enumerate(peak_info_array):
        minx = peak_info["minx"]
        maxx = peak_info["maxx"]
        # print ("final minx and maxx for peak ", jdx, " are ", minx, maxx)
        # integrate and record the value
        # epsabs=0 forces quad to meet the *relative* tolerance: these
        # integrands can be ~1e-20, far below the default epsabs=1.49e-8,
        # so with the default quad accepts its first coarse estimate over
        # the wide window without ever subdividing around the narrow peak.
        # epsrel=1e-12 (vs the 1.49e-8 default) so this reference is
        # guaranteed tighter than the ~1e-12 accuracy of the windowed
        # Fortran integration it validates, rather than only typically so
        ans = integrate.quad(g, minx, maxx, args = (Ep,Eq,), epsabs=0, epsrel=1e-12, limit=200)
        peak_info_array[jdx]["integral"] = ans[0]
        peak_info_array[jdx]["error"] = ans[1]
        
        # this array is for the fitting
        er_arr = np.linspace(minx, maxx, 2000)
        er_integrand_arr = np.zeros(np.size(er_arr))
        for idx, er in enumerate(er_arr):
            er_integrand_arr[idx] = g(er, Ep, Eq)

        # find the width
        # Find maximum function value and corresponding energy 
        er_arr = er_arr.flatten()

        # Filter data to only include points where the value is not NaN
        mask_nan = np.isnan(er_integrand_arr)
        mask_zeros = er_integrand_arr == 0
        filtered_energies = er_arr[~mask_nan & ~mask_zeros]
        filtered_values = er_integrand_arr[~mask_nan & ~mask_zeros]

        max_index = np.argmax(filtered_values)
        max_value = filtered_values[max_index]
        max_energy = filtered_energies[max_index]
        width_estimate = np.nan
        #print (filtered_energies, filtered_values)

        try:
            # estimate the width of the function
            curvature = np.diff(np.diff(filtered_values))

            if np.all(curvature == 0):
                print ("this is a flat line!!")
                peak_info_array[jdx]["width"] = width_estimate
                peak_info_array[jdx]["Ermx"] = max_energy
                peak_info_array[jdx]['integrand_max'] = max_value

            else:
                # Walk to the right of max_index
                for i in range(max_index + 1, len(curvature) - 1):
                    if curvature[i] > 0 and curvature[i-1] <= 0:  
                        inflection_index = i + 2 # Return the index where it switches from negative to positive
                        break
                    
                width_estimate = filtered_energies[inflection_index] - filtered_energies[max_index]

                # Fit Gaussian to filtered data
                # note that we fit the values filtered_values * 1/max_value
                # this is so that when the values are small the fit still works!
                popt, _ = curve_fit(gaussian, filtered_energies, filtered_values * 1/max_value, p0=[1, max_energy, width_estimate])
                # print(f"Fitted Gaussian width (sigma): {width}")

                # Er_max is the value of Er for which g is maximum
                peak_info_array[jdx]['integrand_max'] = popt[0] * max_value
                peak_info_array[jdx]["width"] = popt[2]
                peak_info_array[jdx]["Ermx"] = popt[1]
        except Exception as e:
            print ("Could not fit width")
            peak_info_array[jdx]["width"] = width_estimate
            # not setting an Ermx will mean this call will fail to fill any values
            # I kind of want this to fail in a way that's noticeable
            #peak_info_array[jdx]["Ermx"] = max_energy
            print ("Starting parameters are, ", max_value, max_energy, width_estimate)
            print ("Ep and Eq are ", Ep, Eq)
            print (e)

    # sum the integrals from all the peaks to get the final total integral and error
    integral = 0
    error = 0
    for peak_info in peak_info_array:
        integral += peak_info["integral"]
        error += peak_info["error"]

    # find the overall minimum and maximum and evaluate er_integrand_arr over that range
    min_minx = min(d["minx"] for d in peak_info_array)
    max_maxx = max(d["maxx"] for d in peak_info_array)
    er_arr = np.arange(min_minx, max_maxx, resolution / 2)
    er_integrand_arr = np.zeros(np.size(er_arr))
    for idx, er in enumerate(er_arr):
        val = g(er, Ep, Eq)
        er_integrand_arr[idx] = val

    # now we're ready to return everything to the user
    return (integral, error), (er_arr, er_integrand_arr), peak_info_array

def PpqG_safe_inspect_vec(Ep, Eq,
              *,
              F0,
              eps,
              V,
              p0, p10,
              q0, q10,
              res):
    ppqGArr = []
    for this_Ep, this_Eq in zip(Ep, Eq):
        PpqGval, _, _ = PpqG_safe_inspect(this_Ep, this_Eq, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10, res=res)
        ppqGArr.append(PpqGval[0])
    return ppqGArr

"""
Usage to just get the value of PpqN:
(PpqG, _), _, _ = PpqG_safe_inspect(args)
"""
def PpqG_safe_inspect(Ep, Eq,
              *,
              F0,
              eps,
              V,
              p0, p10,
              q0, q10,
              res):
    if F0 == 0:
        raise ValueError("Fano factor F(Er) = F0 is zero for all Er")

    # Set parameters for integration
    # ER peaks can be as narrow as the zero-energy ionization
    # resolution q0 (~0.06 keV for test parameters), so the scan
    # grid must be finer than that to set accurate integration
    # limits; res=0.01 scans at 0.005 keV, finer than the 0.01 keV
    # grid the Fortran PpqG integrates on

    # Define the function (replace PpqFullG with your actual implementation)
    g = lambda er, ep, eq: PpqFullG(er, ep, eq, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)

    # k, Z are unused placeholders here: is_gamma=True forces Y=1 in PpqExp
    f = lambda er,ep,eq: -1*PpqExp(er,ep,eq,k=0.0,Z=1.0,F0=F0,eps=eps,V=V,p0=p0,p10=p10,q0=q0,q10=q10,is_gamma=True)

    # WARNING
    # this will only work if your peaks have a width
    # greater than 0.1 keV
    # which seems to be true for all reasonable CDMS parameters
    maxx = max(Ep, Eq) + 5
    er_arr = np.arange(5e-21, maxx, res)
    f_arr = np.zeros(len(er_arr))
    for idx, er in enumerate(er_arr):
        f_arr[idx] = f(er, Ep, Eq)

    if not np.any(f_arr > 0):
        if np.all(f_arr == 0):
            # print ("we really couldn't find any real values, there is likely no peak")
            return (0, 0), (er_arr, f_arr), []
        elif np.all(np.isnan(f_arr)):
           # print ("all the evaluated values are NaN!")
            return (np.nan, np.nan), (er_arr, f_arr), []
    else:
        min_idx = np.argmin(f_arr[~np.isnan(f_arr)])
        Ermx = er_arr[min_idx]
        # print ("We found a peak at ", Ermx, " in the exponent term")

    return integrate_g_safe_inspect(g, Ep, Eq, Ermx, res)

