# epistasis_4_4_20
I ran this code using Python 2.7.11 from anaconda (https://www.anaconda.com/download/). I installed the additional programs
```
#monotonic fit requirement
conda install -y -c python cvxopt
#Bayesian lasso requirement
conda install -y -c astropy emcee
#Merge Figure 1A,B with 1C,D
conda install -y -c conda-forge svgutils

conda install matplotlib=2.0.2
```

Additionally, I installed pymol on MacOS as

```
#Structural figure requirement
sudo port install pymol
```

All figures can be created by running

```
./make_figs.sh
```
This code performs all analysis except for the simulations used in figure 3C, D, which require considerable cpu time, and were performed beforehand. These results were saved in cdr1h.csv and cdr3h.csv.
