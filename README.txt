To compile the code, follow the steps below:

1. Install dependancies:
   sudo apt install libboost-all-dev libeigen3-dev libcgal-dev cmake g++ gfortran
   pip3 install numpy scipy matplotlib minieigen pygmsh optimesh

2. Type `cmake . && make` in the terminal from that directory to generate the shared library SPFEMexp.so.

3. Install MFront/MGIS following the link https://thelfer.github.io/mgis/web/install.html

4. Generate shared library of the material constitutive model by running `mfront --obuild --interface=generic MisesVocePlasticity.mfront`

5. Execute the Python code by `python3 retrogressive.py`.