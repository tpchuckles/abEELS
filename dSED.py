import sys,os
sys.path.insert(1,"../../../MD")
from lammpsScrapers import *

sys.path.insert(1,"../../niceplot")
from niceplot import *
from nicecontour import *

a,b,c=5.43729,5.43729,5.43729
nx,ny,nz=120,5,5

path="../../../MD/projects/Si_SED_03/"

positions,velocities,ts=qdump(path+"NVE.qdump")

print("averaging")
if os.path.exists(path+"avg.npy"):
	avg=np.load(path+"avg.npy")
	disp=np.load(path+"disp.npy")
else:
	avg,disp=avgPos(positions,nx*a,ny*b,nz*c)
	np.save(path+"avg.npy",avg) ; np.save(path+"disp.npy",disp)

print("SED")

#Zs,ks,ws=SED(avg,disp,v_xyz=0,p_xyz=0,a=a/4,nk=100) # longitudinal, k𝘹, v𝘹
#Zs+=SED(avg,disp,v_xyz=1,p_xyz=0,a=a/4,nk=100)[0] # transverse, k𝘹, v𝘺

#Zs,ks,ws=SED(avg,velocities,v_xyz=0,p_xyz=0,a=a/4,nk=100) # longitudinal, k𝘹, v𝘹
#Zs+=SED(avg,velocities,v_xyz=1,p_xyz=0,a=a/4,nk=100)[0] # transverse, k𝘹, v𝘺

Zs,ks,ws=SED(avg,velocities,v_xyz=[1,1,0],p_xyz=[1,1,0],a=a/4*np.sqrt(2)/2,nk=100) # longitudinal, k𝘹, v𝘹
Zs+=SED(avg,velocities,v_xyz=[0,0,1],p_xyz=[1,1,0],a=a/4*np.sqrt(2)/2,nk=100)[0] # transverse, k𝘹, v𝘺


ks/=np.pi # convert to 1/wavelength
ws/=(.002*10) # convert to THz: .002 picosecond timesteps, every 10th timestep logged

ZsL=np.log(Zs[:,:]**2)
ZsL[ZsL<0]=0
title=path
contour(ZsL,ks,ws,xlabel="wavelength^-1 (1/Å)",ylabel="frequency (THz)",title=title,filename="figs_momentumresolved/110.png")
