# Autoencoders for flexible wiretap code design in {Python}

This Python3 framework allows designing wiretap codes with finite blocklength
using autoencoders.
It uses Keras (with Tensorflow backend) for the neural network implementation.

## Usage
You can simply run the `autoencoder_wiretap.py` script to start a simulation.
If you want to modify the parameters, e.g. code parameters or the autoencoder
structure, modify the variables at the `main()` function call.

A comparison with a polar wiretap code can be found in the script
`polar_wiretap_comparsion.py`.
