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

The results can then be loaded using `demo.ipynb`.


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

