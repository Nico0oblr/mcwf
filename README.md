Monte carlo wave function implementation for the Heisenberg chain in a lossy / thermal cavity. Also implements direct solution with superoperator notation and Runge-Kutta integration (not optimized, slow!).

Build by running make and you will obtain an executable mcwf.out.

Some model parameters are shown in the config file config.yaml. These configs can always be overwritten by explicitly setting the parameters on the command line.

Tests are currently not in a suite, just in the header tests and you can run them by calling the methods.