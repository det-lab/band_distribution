# Profiling with TAU

This is a maintainer workflow for profiling the Fortran library itself — not
needed to use `band_distribution` for a fit or MCMC.  See the main
[README](../README.md) for that.

TAU is built into `Dockerfile_tau_intel`. This image uses the ifx compiler because that compiler is the easiest to install outside a Docker container.  Build that image first:

```
docker build -f Dockerfile_tau_intel -t tau_intel .
```

Once the image is built, you can run the container and shell into it with the command below. The repository is mounted at `/repo` so that any changes to files persist on the host. The built library remains at `/app`.

```
# for fish shell
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -v "$XAUTH/.Xauthority:/root/.Xauthority" -v $PWD:/repo tau_intel /bin/bash

# for bash shell
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -v "$XAUTH/.Xauthority:/root/.Xauthority" -v ($pwd):/repo tau_intel /bin/bash
```

Inside the docker prompt, you'll need to set the DISPLAY environment variable.  After this, tools that require a GUI window like `paraprof` will work.

```
export DISPLAY=:0.0
```

Set the TAU environment variables:

```
export TAU_MAKEFILE=/packages/tau2/x86_64/lib/Makefile.tau-pthread
export TAU_OPTIONS="-optCompInst -optVerbose -optNoRevert"
export TAU_THROTTLE=0
```

Compile the profiling driver using `tau_f90.sh`, which instruments every Fortran source file.
The module and submodule must be compiled before the driver (the `-c` step generates the `.mod` files needed downstream):

```
cd /tmp

tau_f90.sh -O3 -g -qopenmp -fpp -I/app/include -c /app/src/PpqFort_m.f90
tau_f90.sh -O3 -g -qopenmp -fpp -I/app/include -c /app/src/PpqFort_s.f90

tau_f90.sh -O3 -g -qopenmp -fpp \
    -I/app/include \
    PpqFort_m.o PpqFort_s.o /repo/app/profile_driver.f90 \
    -L/app/lib -lband_distribution \
    -Wl,-rpath,/app/lib \
    -o /repo/profile_driver
```

Run the instrumented binary directly (no `tau_exec` needed with compiler instrumentation):

```
/repo/profile_driver
```

Running the code will produce `profile.*` files in the current directory (`/tmp` if you followed the steps above), which you can investigate using `pprof` (terminal summary) or `paraprof` (GUI view).

## Re-profiling after a source change

If you edit source files in `/repo` and want to re-profile, rebuild the library from `/repo` first, then recompile the profile driver against the new library.

**Step 1: Rebuild the library**
```
cd /repo
python scripts/generate_version_include.py   # src/version.f90.inc is gitignored; fpm needs it generated first
fpm install --compiler ifx --flag "-fpp -O3 -qopenmp -DHAVE_MULTI_IMAGE_SUPPORT=0" --profile release --prefix /tmp/band_new
```

**Step 2: Add the dependency libraries to `LD_LIBRARY_PATH`**

`fpm install` copies the main library but not its dependencies (`libassert.so`, `libjulienne.so`). Point the linker at the fpm build directory:
```
export LD_LIBRARY_PATH=$(find /repo/build -name "libassert.so" -printf "%h\n" | head -1):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(find /repo/build -name "libjulienne.so" -printf "%h\n" | head -1):$LD_LIBRARY_PATH
```

**Step 3: Recompile the profile driver with TAU instrumentation**
```
cd /tmp

tau_f90.sh -O3 -g -qopenmp -fpp -I/tmp/band_new/include -c /repo/src/PpqFort_m.f90
tau_f90.sh -O3 -g -qopenmp -fpp -I/tmp/band_new/include -c /repo/src/PpqFort_s.f90

tau_f90.sh -O3 -g -qopenmp -fpp \
    -I/tmp/band_new/include \
    PpqFort_m.o PpqFort_s.o /repo/app/profile_driver.f90 \
    -L/tmp/band_new/lib -lband_distribution \
    -Wl,-rpath,/tmp/band_new/lib \
    -o /repo/profile_driver
```

**Step 4: Run and inspect**
```
/repo/profile_driver
pprof
```
