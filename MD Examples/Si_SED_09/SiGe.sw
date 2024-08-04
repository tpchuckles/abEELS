# multiple entries can be added to this file, LAMMPS reads the ones it needs
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# Constructed based on: 
# Si-Si : F. H. Stillinger and T. A. Weber, Phys. Rev. B 31, 5262 (1985). (also see Si.sw)
# Ge-Ge : K. Ding and H. C. Andersen, Phys. Rev. B 34, 6987 (1986)
# Si-Ge : M. Laradji, D. P. Landau, and B. Dünweg, Phys. Rev. B 51, 4894 (1995).

#elem1  elem2  elem3  epsilon  sigma    a    lambda  gamma  costheta0  A            B             p  q  tol
Si      Si     Si     2.16816  2.0951   1.8  21.0    1.20   -.3333333  7.049556277  0.6022245584  4  0  0
Ge      Ge     Ge     2.16816  2.0951   1.8  21.0    1.20   -.3333333  7.049556277  0.6022245584  4  0  0

Si      Ge     Ge     2.16816  2.0951   1.8  21.0    1.20   -.3333333  7.049556277  0.6022245584  4  0  0
Ge      Si     Ge     2.16816  2.0951   1.8  21.0    1.20   -.3333333  7.049556277  0.6022245584  4  0  0
Ge      Ge     Si     2.16816  2.0951   1.8  21.0    1.20   -.3333333  7.049556277  0.6022245584  4  0  0

Si      Si     Ge     2.16816  2.0951   1.8  21.0    1.20   -.3333333  7.049556277  0.6022245584  4  0  0
Si      Ge     Si     2.16816  2.0951   1.8  21.0    1.20   -.3333333  7.049556277  0.6022245584  4  0  0
Ge      Si     Si     2.16816  2.0951   1.8  21.0    1.20   -.3333333  7.049556277  0.6022245584  4  0  0

