# abEELS: EELS simulations from MD simulations

This are based on the works of Paul Zeiger (PRB 104, 104301, 2021) and José Ángel Castellanos-Reyes (https://arxiv.org/html/2401.15599v1)

You'll need to have run an MD simulation, with a dump file containing positions and velocities at various points in time. (examples included in "inputs" folder)

You'll also need an input file for abEELS (examples included in "inputs" folder)

run "python3 abEELS.py inputs/yourinputfile.txt" to generate an ω vs kx vs ky simulated 3D dispersion.

various post-processing functions are also available:
"python3 abEELS.py diffraction inputs/yourinputfile.txt" - generates the diffraction image (incoherent sum across all values of ω)
"python3 abEELS.py DOS inputs/yourinputfile.txt" - generates the vibrational density of states plots (sums ω vs kx vs ky across the latter two axes)
"python3 abEELS.py dispersion inputs/yourinputfile.txt" - generates phonon dispersion plots (as "measured" via EELS) (takes a slice across k through the ω vs kx vs ky cube)
"python3 abEELS.py sliceE inputs/yourinputfile.txt" - generates energy-resolved diffraction images (slices in ω)

