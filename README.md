Monte carlo wave function implementation for the Heisenberg chain in a lossy / thermal cavity. Also implements direct solution with superoperator notation and Runge-Kutta integration (not optimized, slow!).

Build by running make and you will obtain an executable mcwf.out.

Some model parameters are shown in the config file config.yaml. These configs can always be overwritten by explicitly setting the parameters on the command line.

Depends on eigen3 and yaml-cpp. You will have to create a lib and include folder and download/build them. Not uploading those to a git or automating the process. Too much hassle.

Example for the usage of python bindings are currently the plethora of scripts just in the root folder (Most functionality is bound to python).
Tests are currently all contained in tests.py and can be run after compiling the shared library.