# Common path tomographic diffractive microscope (cTDM)
Image reconstruction for cTDM

To construct a three-dimensional (3-D) refractive index (RI) map of a biological cell, a series of two-dimensional amplitude/phase images of the cell was acquired using a nearly cTDM.

The amplitude/phase images are retrieved from the interference images by band-pass filtering in the spatial frequency domain without zero-padding, followed by a Fourier-based phase unwrapping method.
3-D RI maps are reconstructed iteratively based on optical diffraction tomography with direct interpolation in the Fourier domain and the positivity constraint.
All of the phase retrieval and 3-D reconstruction steps were performed on graphic processing units.
