# Workspace guide

---------------

This is the experimental workspace of training word embedding (WE) with BLSTM-RNN.

To run the experiment, the following softwares and packages must be installed:  
* Python (2.7.6, 32bit)
* Python package: numpy (1.8.1, win32, for python2.7), http://www.numpy.org/
* Python package: netCDF (1.0.9, win32, for python2.7), https://code.google.com/p/netcdf4-python/

Detailed information of the software we used is listed in the parenthesis.
Higher version should work but is not tested. 

After installing all the required softwares and packages, 
type cmd:
python blstmauen.1.g-3.py
to run the whole training SE experiment.


Trained phone embedding can be found in exp/blstmauen.1.g-3/result
Line format:
word\tnum1\tnum2\tnum3....

e.g.
green	0.001	0.002	0.012...


Note:
This experiment is slow, may take days to complete