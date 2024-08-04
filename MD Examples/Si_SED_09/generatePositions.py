import sys,os
from datetime import datetime
import numpy as np

nx,ny,nz=25,25,5	# how many unit cells in x,y,z

a,b,c=5.43729,5.43729,5.43729 # unit cell definition, Å,°

positionsInUnitCell=[[0.,0.,0.],[.5,.5,0.],[.5,0.,.5],[0.,.5,.5],[.25,.25,.25],[.75,.75,.25],[.75,.25,.75],[.25,.75,.75]]
UCID=[1,2,3,4,1,2,3,4] ; CCID=[1,1,1,1,2,2,2,2]
masses=[28.0855]

# set up atom field (cycle through unit cells)
nA=nx*ny*nz*len(positionsInUnitCell)
atoms=np.zeros((nA,4)) # list of atoms, including x,y,z position, and type
CCs=np.zeros(nA,dtype=int) ; UCIDs=np.zeros(nA,dtype=int) # crystal coordinate of each atom ("which atom is this within the unit cell") and unit cell ID ("which unit cell")
ct=-1
for i in range(nx):
	for j in range(ny):
		for k in range(nz):
			for n in range(len(positionsInUnitCell)):
				ct+=1
				x,y,z=positionsInUnitCell[n]	# UC positions
				x*=a ; y*=b ; z*=c
				x+=i*a ; y+=j*b ; z+=k*c	# offset by whichever unit cell we're in
				
				atoms[ct,:]=[x,y,z,1]
				#cc=n+1 ; uc=ct//len(positionsInUnitCell)+1 # WRONG. there technically aren't 8 atoms per UC, there are 2! using 8 will give unnecessary folding
				ucid = ct//len(positionsInUnitCell) # WHICH (macro) UC are we in?
				cc=CCID[n] ; ucid=ucid*4+UCID[n]
				CCs[ct]=cc ; UCIDs[ct]=ucid # USED TO CREATE LATTICE MAPPING FILE FOR pSED ("lattice.dat")
# update the type for all atoms where z position is above the interface

# DISPLACE 1% RANDOM ATOMS
#nFrenkel=int(round(nA*.01))
#edges=np.linspace(0,nA,nFrenkel+1)
#for i in range(nFrenkel):
#	ni=int(edges[i]) ; nf=int(edges[i+1])
#	ids=np.arange(ni,nf)
#	np.random.shuffle(ids)
#	print(i,ni,nf,ids[0])
#	i=ids[0]
#	atoms[i,1]+=2.5*b ; CCs[i]=0 ; UCIDs[i]=0


#z_atoms=atoms[:,2]
#atoms[z_atoms>z_interface,3]=2
# let's also sort by z location
#sortIndices=np.argsort(z_atoms)
#z_atoms=z_atoms[sortIndices]
#atoms=atoms[sortIndices]
lx=a*nx ; ly=b*ny ; lz=c*nz

# save off positions file
projectName=os.getcwd().split("/")[-1]

#[ "0.0 "+str(n*l)+" "+c+"lo "+c+"hi" for n,l,c in zip([nx,ny,nz],[a,b,c],["x","y","z"])] +\
lines=[ "########### "+sys.argv[0]+" "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+" ###########",
	str(len(atoms))+" atoms" , "" , "1 atom types", "" ] +\
[ "0.0 "+str(lx)+" xlo xhi" , "0.0 "+str(ly)+" ylo yhi" , "0.0 "+str(lz)+" zlo zhi" ] +\
[ "" , "Masses" , "" ] +\
[ str(n+1)+" "+str(m) for n,m in enumerate(masses) ] +\
[ "Atoms" , "" ] +\
[ str(n+1)+" 1 "+str(int(a[3]))+" "+str(float(a[0]))+" "+str(float(a[1]))+" "+str(float(a[2])) for n,a in enumerate(atoms) ]

with open("positions.pos",'w') as f:
	for l in lines:
		f.write(l+"\n")

with open("lattice.dat",'w') as f:
	for a in range(nA):
		write=[a+1,UCIDs[a],CCs[a],masses[int(atoms[a,3])-1]]
		write=" ".join([ str(v) for v in write ])
		f.write(write+"\n")


# get atomID lists
#cutoffs=[0,.6*c,3*c,(nz-3)*c,(nz-.6)*c,(nz+1)*c] ; names=["lfix","hotres","lead","coldres","rfix"]
#for n,name in enumerate(names):
#	mask=np.zeros(len(atoms))
#	z1=cutoffs[n] ; z2=cutoffs[n+1]
#	mask[z_atoms>=z1]=1 ; mask[z_atoms>=z2]=0
#	where=np.where(mask==1)[0]+1 # +1, because these are indices (list starts at 0), and we want atom IDs (which start at 1)
#	print("group "+name+" id "+str(min(where))+":"+str(max(where)))
#print("group moving union hotres lead coldres")



