# epistasis_4_4_20
This is the code used to create the figures in my preprint "Physical epistatic landscape of antibody binding affinity" (https://www.biorxiv.org/content/early/2017/12/11/232645). I ran this code using Python 3.6.6 from anaconda (https://www.anaconda.com/download/). In particular I used used scipy=1.0.1, numpy=1.14.5, pandas=0.20.3, and matplotlib=2.2.2. I installed the additional programs
```
#monotonic fit requirement
conda install -y -c conda-forge cvxopt
#Bayesian lasso requirement
conda install -y -c astropy emcee
#Merge Figure 1A,B with 1C,D
conda install -y -c conda-forge svgutils
```
This installed cvxopt=1.1.8, emcee=2.2.1, and svgutils=0.2.0. 

I made a beeswarm plot, which can be installed by pip
```
pip install pybeeswarm
```

Additionally, I installed pymol on MacOS as

```
#Structural figure requirement
sudo port install pymol
```
However, any way that pymol can be run from the command line will work.

All figures can be created by running
```
./make_figs.sh
```
This code performs all analyses except for the simulations used in figure 3C, D, which require considerable cpu time, and were performed beforehand. These results were saved in cdr1h.csv and cdr3h.csv. Runtime on a 2014 Macbook air with 1.3 GHz Intel Core i5 and 8 GB 1600 MHz DDR3 ram takes approximately 6 hours.

An example using the monotonic transformation algorithm can be found monotonic_fit_example.ipynb

