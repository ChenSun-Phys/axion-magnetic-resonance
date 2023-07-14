# Enhanced Axion-Photon Conversion: <br/> A Novel Approach With Rotating Magnetic Profile

A series of axion search experiments rely on converting them into photons inside a constant magnetic field background. We propose a novel method to enhance the conversion rate bewteen photons and axions with a varying magnetic field profile. We show that the enhancement can be achieved by both a rotating magnetic field and a harmonic oscillation of the magnitude. Our approach can extend the projected ALPS II reach in the axion-photon coupling ($g_{a\gamma}$) by two orders of magnitude at $m_a = 10^{-3}\;\mathrm{eV}$ with moderate assumptions. 

This is the numerical code that accompanies the publication. 


Requirements
-----------------------------------------

1. Python 3
2. numpy
3. scipy
4. pickle



How to run
-----------------------------------------

To run this code, check out the notebook `demo.ipynb`. To reproduce the scan in $m_a$ with Gaussian noise, run the following

	python scan.py
		-s < initial coordinate >
		 -e < end of propagation >
		 -B < magnetic field in Tesla >
		 -w < laser wavelength in nm >
		 -N < number of domains >
		 -l < lower value of log10ma >
		 -u < lower value of log10ma >
		 -g < grid size >
		 -o < output folder >
		 -n < number of polls>
		 -c < ga in GeV**-1>
		 -v < variation of noise>
		 -f < fraction variation of noise
		 -t < theta dot mean>
		 -p < initial state: photon 0, axion 1>

The results can then be loaded using `demo.ipynb`. The scans used in this work was produced with the following directive for the axion production mode:

    python scan.py -s 0 -e 106 -B 5.3 -w 1064 -N 2000 -l -4.5 -u -2.5 -g 20  -n 100 -c 1.e-11 -f 0.01 -t 1. -p 0 -o chains/run024_prod_N2000_f001
	python scan.py -s 0 -e 106 -B 5.3 -w 1064 -N 2000 -l -4.5 -u -2.5 -g 20  -n 100 -c 1.e-11 -f 0.10 -t 1. -p 0 -o chains/run024_prod_N2000_f010
	python scan.py -s 0 -e 106 -B 5.3 -w 1064 -N 10 -l -4.5 -u -2.5 -g 20  -n 100 -c 1.e-11 -f 0.01 -t 1. -p 0 -o chains/run024_prod_N10_f001
	python scan.py -s 0 -e 106 -B 5.3 -w 1064 -N 10 -l -4.5 -u -2.5 -g 20  -n 100 -c 1.e-11 -f 0.10 -t 1. -p 0 -o chains/run024_prod_N10_f010

and the following for the photon regeneration:

    python scan.py -s 0 -e 106 -B 5.3 -w 1064 -N 2000 -l -4.5 -u -2.5 -g 20  -n 100 -c 1.e-11 -f 0.01 -t 1. -p 1 -o chains/run021_N2000_f001
	python scan.py -s 0 -e 106 -B 5.3 -w 1064 -N 2000 -l -4.5 -u -2.5 -g 20  -n 100 -c 1.e-11 -f 0.10 -t 1. -p 1 -o chains/run021_N2000_f010
	python scan.py -s 0 -e 106 -B 5.3 -w 1064 -N 10 -l -4.5 -u -2.5 -g 20  -n 100 -c 1.e-11 -f 0.01 -t 1. -p 1 -o chains/run021_N10_f001
	python scan.py -s 0 -e 106 -B 5.3 -w 1064 -N 10 -l -4.5 -u -2.5 -g 20  -n 100 -c 1.e-11 -f 0.10 -t 1. -p 1 -o chains/run021_N10_f010

It takes about 1-2 hours on a 48 core cluster (`Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz`) to finish the run. If the cluster in your physics department is jammed, you can first play with the pickled scan results we create. They can be downloaded from [here](https://cloud.cosmicdiscord.net/s/AEsKSkW2NPfwYxf) or by request. 


Bibtex entry
-----------------------------------------

If you find this study useful and/or use this code for your work, please consider citing [Seong, Sun, & Yun 2023](https://arxiv.org/abs/xxxx.xxxxx). The BiBTeX is the following:

	@article{Seong:2023xyz,
	    author = "Seong, Hyeonseok and Sun, Chen and Yun, Seokhoon",
	    title = "{Enhanced Axion-Photon Conversion: \\ A Novel Approach With Rotating Magnetic Profile}",
	    eprint = "23xx.xxxxx",
	    archivePrefix = "arXiv",
	    primaryClass = "hep-ph",
	    month = "xx",
	    year = "2023"
	}

