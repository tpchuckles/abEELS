import sys,os
from datetime import datetime
import numpy as np

nx,ny,nz=4,4,40	# how many unit cells in x,y,z # BZ FOLDS ONCE IN X (100 BZ PTS) AND TWICE IN Y (124 BZ BTS)
a,b,c,alpha,beta,gamma=3.188930,3.188930,5.192357,90,90,120 # unit cell definition, Å,°

positionsInUnitCell=[[1/3,2/3,0],[1/3,2/3,3/8],[2/3,1/3,1/2],[2/3,1/3,7/8]] # hexagonal structure, GaN (RELATIVE COORDS, NEED TO BE SKEWED FOR GAMMA)
z_interface=c*nz/2+.1
masses=[26.981539,14.006700]

# set up atom field (cycle through unit cells)
atoms=np.zeros((nx*ny*nz*len(positionsInUnitCell),4)) # list of atoms, including x,y,z position, and type
UCi=[] ; UCj=[] ; UCk=[] ; atInd=[]
ct=-1
for i in range(nx):
	for j in range(ny):
		for k in range(nz):
			for n in range(len(positionsInUnitCell)):
				ct+=1
				x,y,z=positionsInUnitCell[n]	# UC positions
				x*=a ; y*=b ; z*=c
				x+=i*a ; y+=j*b ; z+=k*c	# offset by whichever unit cell we're in
				atoms[ct,:]=[x,y,z,n%2+1]
				UCi.append(i) ; UCj.append(j) ; UCk.append(k); atInd.append(n)
# update the type for all atoms where z position is above the interface
#z_atoms=atoms[:,2]
#atoms[z_atoms>z_interface,3]=2
# let's also sort by z location
#sortIndices=np.argsort(z_atoms)
#z_atoms=z_atoms[sortIndices]
#atoms=atoms[sortIndices]

data=np.zeros((len(atoms),5),dtype=int)
data[:,0]=np.arange(len(atoms)) ; data[:,1]=UCi ; data[:,2]=UCj ; data[:,3]=UCk ; data[:,4]=atInd
np.savetxt("lattice.dat",data.astype(int),fmt='%i')


alpha*=np.pi/180 ; beta*=np.pi/180 ; gamma*=np.pi/180

# Skew atoms, and calculate bbox parameters for LAMMPS. it's almost like a rotation matrix, but "how much it rotates" depends on distance in y (ie, the atoms on the x axis for example, do not get rotated)
# https://en.wikipedia.org/wiki/Rotation_matrix ; https://en.wikipedia.org/wiki/Shear_mapping
skew=np.eye(3) ; skew[0,1]=-np.sin(gamma-np.pi/2) ; skew[1,1]=np.cos(gamma-np.pi/2)
for n,at in enumerate(atoms):
	xyz=at[:3].reshape((3,1))
	xyz=np.matmul(skew,xyz)
	atoms[n,:3]=xyz.reshape((1,3))

# https://docs.lammps.org/Howto_triclinic.html
la=a*nx ; lb=b*ny ; lc=c*nz
lx=la
xy=lb*np.cos(gamma)
xz=lc*np.cos(beta)
ly=np.sqrt(lb**2-xy**2)
yz=(lb*lc*np.cos(alpha)-xy*xz)/ly
lz=np.sqrt(lc**2-xz**2-yz**2)

# FOR abEELS SIMS WE WANT TO LOOK DOWN Y DIRECTION, AND WE WANT OUR SAMPLE FLAT. WE NEED TO REMAP TO GET BACK TO A RECTANGULAR SIM VOLUME
#  _______        ______
#  \  |   \      |   \  |
#   \ |    \ --> |    \ |
#    \|_____\    |_____\|
atoms[atoms[:,0]<0,0]+=lx


# save off positions file
projectName=os.getcwd().split("/")[-1]

#[ "0.0 "+str(n*l)+" "+c+"lo "+c+"hi" for n,l,c in zip([nx,ny,nz],[a,b,c],["x","y","z"])] +\
lines=[ "########### "+sys.argv[0]+" "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+" ###########",
	str(len(atoms))+" atoms" , "" , "2 atom types", "" ]
lines=lines+[ "0.0 "+str(lx)+" xlo xhi" , "0.0 "+str(ly)+" ylo yhi" , "0.0 "+str(lz)+" zlo zhi" ]
#lines=lines+[ str(xy)+" "+str(xz)+" "+str(yz)+" xy xz yz" ]
lines=lines+[ "" , "Masses" , "" ]
lines=lines+[ str(n+1)+" "+str(m) for n,m in enumerate(masses) ]
lines=lines+[ "Atoms" , "" ]
lines=lines+[ str(n+1)+" 1 "+str(int(a[3]))+" "+str(float(a[0]))+" "+str(float(a[1]))+" "+str(float(a[2])) for n,a in enumerate(atoms) ]

with open("positions.pos",'w') as f:
	for l in lines:
		f.write(l+"\n")

# get atomID lists
#cutoffs=[0,.6*c,3*c,(nz-3)*c,(nz-.6)*c,(nz+1)*c] ; names=["lfix","hotres","lead","coldres","rfix"]
#for n,name in enumerate(names):
#	mask=np.zeros(len(atoms))
#	z1=cutoffs[n] ; z2=cutoffs[n+1]
#	mask[z_atoms>=z1]=1 ; mask[z_atoms>=z2]=0
#	where=np.where(mask==1)[0]+1 # +1, because these are indices (list starts at 0), and we want atom IDs (which start at 1)
#	print("group "+name+" id "+str(min(where))+":"+str(max(where)))
#print("group moving union hotres lead coldres")



