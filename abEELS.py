# eigenVectors.py read from Si_300K and appears to successfully regenerates displacements-based vibrational DOS
# momentumResolved.py and .ipynb read from Si_SED_03 and looks close to having figured out the momentum-resolved aspect
# here, we combine those results (and explore whether the complex diffraction plane results still give vDOS
# and we write it outside of jupyter so we can run on rivanna. and we skip the rotation so we can directly compare it against SED for Si_SED_04

import sys,os,pyfftw,time,shutil,glob
#sys.path.insert(1,"copiedPythonUtils")
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../../MD') )	# in case someone "from abEELS import *" from somewhere else
from lammpsScrapers import *
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../niceplot') )
from niceplot import *
from nicecontour import *

import abtem,ase
import matplotlib.pyplot as plt

# TODO HUGE PROBLEM. ABTEM HAS LOOSE DEFINITIONS (OR MAYBE I DO) OF MATRIX INDICES. SOMETIMES AN INDEX WILL BE PRESENT FOR FP=1, OTHERTIMES NOT. IDK THE RHYME OR REASON. ALSO IF YOU HAVE MULTIPLE SCAN POSITIONS, THERE'S AN INDEX FOR IT, IF NOT, NO SCAN POSITION INDEX. WHICH MAKES IT VERY MESSY TO HAVE THE SAME CODE PROCESSING OUTPUTS FROM PLANE WAVES, CONVERGENT WAVES, MULTIPLE ATOM CONFIGURATIONS OR JUST ONE, AND MULTIPLE OR SINGULAR PROBE POSITIONS. NEED TO HAVE SOME SORT OF UNIFIED WAY OF HANDLING ALL THESE. 
# NEED TO TEST ALL PERMUTATIONS OF PARAMETERS:
# CONVERGENCE (0, !0),layerwise (0,!0), parallelFP (T,F), probePositions (len==0, len!=0)

# GENERAL PRINCIPLES OF OPERATION: 
# read run parameters (readInputFile())
# read molecular dynamics dump file (importPositions(), assumes filenames and stuff from input file. default behavior is to expect a dump file which should have atomic positions at numerous timesteps. we calculate an average position for each atom displacements from the mean. (variant 1: you can also feed your initial positions file (useLattice="filename.pos") and we'll use this instead of the average positions, with displacements from this "perfect" structure. variant 2: you can provide a file with positions only (not time-dependent) (dumpfile="filename.pos" ; dumptype="positions"). you won't get phonons, so we'll have to add them in artificially (addModes variable. see Si_cprobe input)). 
# trimming/rotation/mirroring/stacking/tiling operations may be optionally applied, in this order. (preprocessPositions() handles trimming/rotation/mirroring, getFPconfigs() handles stacking/tiling. input file variables: trim, tiling, beamAxis, flipBeam)
# a gaussian band-pass filter is applied to displacements (generateEnergyMasks() generates these, bandPassDisplacements() applies them)
# an electron-wave propagation simulation is performed (ewave()). 
# energy loss is simulated by subtracting coherent and incoherent parts from a number of frozen phonon configurations for a given energy bin (energyLoss()). this means the simulations are technically "energy resolved diffraction images" as opposed to "momentum resolved electron energy loss". no electrons lose energy within these simulations! (purely elastic electron wave propagation). 

# POST PROCESSING: 
# You may be interested in simulating the phonon density of states (eDOS()), which is simply the sum over the 

#, as a function of 
# the sum

# BEGIN GENERAL FUNCTIONS

def main():
	infile=sys.argv[-1]
	readInputFile(infile)
	importPositions()
	#calculateLattice()
	preprocessPositions()
	energyCenters,masks=generateEnergyMasks(0,Emax,Ebins) # performs FFT on displacements, and generates gaussians we'll use for energy masking

	for m,(ec,mask) in enumerate(zip(energyCenters,masks)):
	#for m,ec,mask in zip(reversed(range(len(energyCenters))),reversed(energyCenters),reversed(masks)):
		#if not (30<=m<40):
		#	continue
		print("PROCESSING ENERGY BAND:",ec)
		#if os.path.exists(outDirec+"/psi_"+str(m)+".npy"):
		#	print("skipping")
		#	continue
		bdisp=bandPassDisplacements(mask)	# energy band pass on displacements = iFFT(FFT(displacements)*gaussianMask)
		atomStack=getFPconfigs(bdisp,plotOut=m)	# tiling and timestep selection for frozen phonon configurations
		if parallelFP:	# may cause ram overload. e.g. 150Å x 150Å at 0.05Å spacing = 3000x3000 real-space sampling of waves, 64 bit complex
			psi=ewave(atomStack,fileSuffix="e"+str(m),plotOut=(m==0)) # values. 1 FP config = 68 MB. multiply by the number of FP configs
		else:
			psi0=ewave([atomStack[0]],fileSuffix="e"+str(m)+"_fp0",plotOut=(m==0))
			#shape=list(np.shape(psi0)) ; shape[0]=len(atomStack) ; print(shape)
			#psi=np.zeros(shape,dtype=type(psi0[0,0,0])) ; psi[0,:,:]+=psi0[0,:,:]
			for n,aS in enumerate(atomStack):
				psiN=ewave([atomStack[n]],fileSuffix="e"+str(m)+"_fp"+str(n),plotOut=False)
				#psi[n,:,:]=psiN[0,:,:]
		#energyLoss(psi)

def readInputFile(infile):
	# READ IN THE FOLLOWING VARIABLES FROM THE INPUT FILE
	# path,dumpfile,atomTypes,nx,ny,nz,a,b,c,dt,trim,beamAxis,flipBeam,tiling,CONVERGENCE
	# Note no skewed primitive cells are currently supported for tiling
	global path, dumpfile, dumptype, useLattice, atomTypes, nx, ny, nz, a, b, c ,dt, trim, beamAxis, flipBeam, tile
	global CONVERGENCE, outDirec, probePositions, saveExitWave, restartFrom, restartDirec, addModes, modifyProbe
	global kmask_xis,kmask_xfs,kmask_yis,kmask_yfs,kmask_cxs,kmask_cys,kmask_rad,kmask_lbl
	kmask_xis,kmask_xfs,kmask_yis,kmask_yfs,kmask_cxs,kmask_cys,kmask_rad,kmask_lbl=[],[],[],[],[],[],[],[]
	dumptype="qdump" ; probePositions=[] ; addModes={} ; useLattice=False ; modifyProbe=False
	saveExitWave=False ; restartFrom=False
	lines=open(infile,'r').readlines()
	exec("".join(lines),globals())
	kmask_lbl=kmask_lbl+[ str(v) for v in range(len(kmask_xis)+len(kmask_cxs)-len(kmask_lbl)) ]
	print(kmask_lbl)
	outDirec="figs_abEELS_"+infile.strip("/").split("/")[-1].split(".")[0]
	if restartFrom:
		restartDirec="figs_abEELS_"+restartFrom ; RD=restartDirec
		if not os.path.exists(restartDirec):	# First run, restart direc might not exist (if so, turn off restartFrom)
			restartFrom=False
		else:
			for i in range(numCycles):	# new output appends "cycleN"
				OD=outDirec+"_cycle"+str(i)
				if not os.path.exists(OD):	# if output direc already exists, that becomes our new restart direc
					outDirec=OD
					break
				else:
					RD=restartDirec+"_cycle"+str(i)
			else:
				print("NUM CYCLES",numCycles,"REACHED. QUITTING")
				sys.exit()
			restartDirec=RD			# only update restartDirec once we've found the right N
			print("RESTART SET: outDirec:",outDirec,"restartDirec",restartDirec)
	os.makedirs(outDirec,exist_ok=True)
	shutil.copy(infile,outDirec+"/")

def importPositions():
	global velocities,ts,types,avg,disp #,positions

	# read in from custom dump file, containing positions *and* velocities
	if dumptype=="qdump":
		# TODO WE DON'T ACTUALLY NEED ALL OF THESE (positions,velocities,ts,types,avg,disp). 
		# IF avg AND disp FILES EXIST, WE SHOULD READ ONLY THOSE IN. (save on ram). 
		#if not os.path.exists(path+"avg.npy") and os.path.exists(path+dumpfile+"_pos.npy"):
		#	positions=np.load(path+dumpfile+"_pos.npy")
		#	avg,disp=avgPos(positions,nx*a,ny*b,nz*c) # nt, na, 3
		#	np.save(path+"avg.npy",avg)
		#	np.save(path+"disp.npy",disp)

		positions,velocities,ts,types=qdump(path+dumpfile)
	
		# calculate average positions (includes wrapping) and each atom's instantaneous displacement from the equilibrium position
		print("averaging")
		if os.path.exists(path+"avg.npy"):
			positions=[]
			avg=np.load(path+"avg.npy")
			disp=np.load(path+"disp.npy")
		else:
			avg,disp=avgPos(positions,nx*a,ny*b,nz*c) # nt, na, 3
			np.save(path+"avg.npy",avg)
			np.save(path+"disp.npy",disp)
			positions=[]

		if useLattice:
			print("OVERWRITING AVERAGE POSITIONS FROM LATTICE FILE:",useLattice)
			avg,typ=scrapePos(path+useLattice)

	elif dumptype=="positions":
		pos,types=scrapePos(path+dumpfile) # qdump positions,velocities are [nt,na,3]. avg is [na,3] and disp is [nt,na,3]
		positions=np.asarray([pos,pos])
		velocities=np.zeros(np.shape(positions))
		disp=np.zeros(np.shape(positions)) ; avg=pos ; ts=np.asarray([0,1])

	#elif dumptype=="lattice":
	#	

	# WRAPPING: imagine a plane of atoms initialized at z=0. these atoms jiggle through the PBC, and there's a 50% chance their average is at the top vs bottom of the simulation. then when a wave hits "dangling" top atoms, we'll find asymmetry where there is none, and other strange e-wave interactions
	# infer spacing 
	spacing=[]
	for i in range(3):
		xyzs=list(sorted(avg[:,i]))	# all positions in x
		dxyz=np.gradient(xyzs)		# spacing between atoms in x
		dxyz=dxyz[dxyz>.1]		# assume jiggling is less than .1, assume no interatomic spacing is less than .1
		spacing.append(min(dxyz))
	print("INFERRED INTERATOMIC SPACING",spacing,"(used for haircut)")
	spacing=[ s/2 for s in spacing ]
	for i in range(3):
		mask=np.zeros(len(avg))
		l=[nx*a,ny*b,nz*c][i]
		mask[avg[:,i]>l-spacing[i]]=1
		print("haircut applied to",len(mask[mask==1]),"atoms")
		avg[mask==1,i]-=l
	outToPositionsFile(outDirec+"/haircutchecker.pos",avg,types,nx*a,ny*b,nz*c,list(range(1,len(set(types))+1)))

	if "A" in addModes.keys(): # addModes={"THz":[...],"iA":[...],"pdirec":[...],"vdirec":[...],"phase":[...]}
		for i in range(len(addModes["A"])):
			A,k,w,p,v,phi=[ addModes[key][i] for key in ["A","k","w","pdirec","vdirec","phase"] ]
			print("add ",A,"*","sin(",k,"*","xyz[",p,"]","+",phi,")")
			added=addWave(A,k,phi,p,v)
			avg[:,:]+=added[:,:]
			#positions[:,:,:]+=added[None,:]

def addWave(A,k,phi,p_xyz,v_xyz):
	na=len(avg)
	# VECTOR MATH RIPPED STRAIGHT OUTA lammpsScapers.py > SED()
	if isinstance(p_xyz,(int,float)): # 0,1,2 --> x,y,z
		xs=avg[:,p_xyz] # a,xyz --> a
	else:	# [1,0,0],[1,1,0],[1,1,1] and so on
		# https://math.stackexchange.com/questions/1679701/components-of-velocity-in-the-direction-of-a-vector-i-3j2k
		# project a vector A [i,j,k] on vector B [I,J,K], simply do: A•B/|B| (dot, mag)
		# for at in range(na): x=np.dot(avg[at,:],p_xyz)/np.linalg.norm(p_xyz)
		# OR, use np.einsum. dots=np.einsum('ij, ij->i',listOfVecsA,listOfVecsB)
		p_xyz=np.asarray(p_xyz)
		d=p_xyz[None,:]*np.ones((na,3)) # xyz --> a,xyz
		xs=np.einsum('ij, ij->i',avg[:,:],d) # pos • vec, all at once
		xs/=np.linalg.norm(p_xyz)
	if isinstance(v_xyz,(int,float)):
		displacementDirection=np.zeros(3)
		displacementDirection[v_xyz]=1
	else:
		v_xyz/=np.sqrt(np.sum(np.asarray(v_xyz)**2))
	displacementsToAdd=np.zeros(np.shape(avg))
	for ijk in range(3):
		displacementsToAdd[:,ijk]+=v_xyz[ijk]*A*np.sin(k*xs+phi)
	return displacementsToAdd


# HACKY FUNCTION USED FOR CALCULATING THE PRISTINE-LATTICE RESULTS. 
# after importing pos/avg/disp, we simply read in your positions file (overwriting the calculated average positions) and zero-out displacements
# all subsequent code (rotation, tiling, trimming, adding displacements for energy bands, calculating wave) thus are tricked into using the lattice
#def calculateLattice():
#	global avg,disp
#	#scrapePos
#	#avg=np.loadtxt(path+"positions.pos",skiprows=15)[:,3:]
#	disp*=0

# TRIM, TILE, ROTATE, FLIP
#trim=[[],[],[]] # add two values (unit cell counts) to any dimension to trim to that size
#tile=[1,10,1]	# expanding the simulation in direction parallel to beam will lead to artifacts in reciprical space
#beamAxis=1	# abtem default is to look down z axis. we'll rotate the sim if you specify 0 or 1s
#flipBeam=False	# default is to look from +z towards -z, but we can mirror the simulation too (e.g. if we're more sensitivity to top layers)
def preprocessPositions():
	global positions,velocities,ts,types,avg,disp,a,b,c,nx,ny,nz,tiling

	# TRIM A TOO-BIG VOLUME
	for ijk,lims in enumerate(trim):
		if len(lims)!=2:
			continue
		abc=[a,b,c][ijk]
		xs=avg[:,ijk] ; xi=lims[0]*abc ; xf=lims[1]*abc
		print("TRIM:",ijk,xi,xf)
		mask=np.zeros(len(xs))
		mask[xs>=xi]=1 ; mask[xs>=xf]=0
		#mask[xs>=xi]=1 ; mask[xs>=xi+2]=0 ; mask[xs>=xf-2]=1 ; mask[xs>=xf]=0 # CUSTOM MASK MOD TO CLEAR OUT ATOMS SO WE CAN VISUALIZE THE PROBE
		avg=avg[mask==1,:] ; disp=disp[:,mask==1,:] ; velocities=velocities[:,mask==1,:] ; types=types[mask==1]
		avg[:,ijk]-=xi
		if ijk==0:
			nx=lims[1]-lims[0]
		if ijk==1:
			ny=lims[1]-lims[0]
		if ijk==2:
			nz=lims[1]-lims[0]

	# ROTATE SO BEAM AXIS IS Z
	if beamAxis!=2: # default is to look down z, so we need to flip axes if we want to look down a different direction
		rollby=[-1,1][beamAxis] # to look down x, roll -1 [x,y,z] --> [y,z,x]. to look down y, roll +1 [x,y,z] --> [z,x,y]
		avg=np.roll(avg,rollby,axis=1) # na,[x,y,z] --> na,[y,z,x]
		disp=np.roll(disp,rollby,axis=2) # nt,na,[x,y,z] --> nt,na,[y,z,x]
		velocities=np.roll(velocities,rollby,axis=2) 
		a,b,c=np.roll([a,b,c],rollby)
		nx,ny,nz=np.roll([nx,ny,nz],rollby)
		tiling=np.roll(tiling,rollby)
	if flipBeam:
		avg[:,2]*=-1 ; avg[:,2]+=nz*c # # na,[x,y,z], flip z, shift position
		disp[:,:,2]*=-1 # # nt,na,[x,y,z], no shift for displacements or velo
		velocities[:,:,2]*=-1

	# TODO TECHNICALLY BOTH OF THESE ARE WRONG. swapping axes is effectively mirroring across a 45° plane, and flibBeam mirrors across one of the three cartesian planes. mirroring is wrong though, think about chirality: you'll flip handedness

def generateEnergyMasks(fmin,fmax,numBands,bandWidth=0):
	global dfft

	if numBands==1:
		energyCenters=[0] ; masks=[np.ones(len(disp))] ; dfft=np.zeros(np.shape(disp)) ; ws=np.linspace(0,1,len(disp)) ; ddos=[1] ; vdos=[1]
		#return energyCenters,masks
	else:
		# STEP 2, APPLY A FREQUENCY BAND-PASS TO DISPLACEMENTS TO ENABLE FROZEN-PHONON ABTEM SIMS IN THE SPIRIT OF ZEIGER'S FREQUENCY-ISOLATED THERMOSTAT
		print("FFTing positions and velocities")
		dfft=np.fft.fft(disp,axis=0) # FFT on displacements tells displacements associated with each frequency. THIS IS JUST EIGENVECTOR?
		vfft=np.fft.fft(velocities,axis=0) # FFT on displacements tells displacements associated with each frequency. THIS IS JUST EIGENVECTOR?
		ws=np.fft.fftfreq(len(ts),ts[1]-ts[0])
		n=int(len(ws)/2)
		ws/=dt # convert to THz: e.g. .002 picosecond timesteps, every 10th timestep logged
		#print(np.shape(dfft)) # nω, na, 3

		print("saving dDOS and vDOS")
		ddos=np.sum(np.absolute(dfft[:,:,1]),axis=1) # nω, na, 3 --select-Y--> nω, na --sum-a--> nω
		vdos=np.sum(np.absolute(vfft[:,:,1]),axis=1)
		#plot([ws,ws],[ddos/np.trapz(ddos),vdos/np.trapz(vdos)],markers=['-']*2,labels=["ddos","vdos"],xlim=[0,None],ylim=[0,None])
		out=np.zeros((len(ddos),3)) ; out[:,0]=ws ; out[:,1]=ddos ; out[:,2]=vdos
		np.savetxt(outDirec+"/DOS.txt",out)
		ddos/=np.trapz(ddos) ; vdos/=np.trapz(vdos)

		print("generating frequency masks")
		Xs_d=[ws] ; Ys_d=[ddos] ; masks=[] ; Xs_v=[ws] ; Ys_v=[vdos]

		energyCenters=np.linspace(fmin,fmax,numBands+1)[1:]
		if bandWidth==0:
			bandWidth=(energyCenters[1]-energyCenters[0])/2.5

		for i,ec in enumerate(energyCenters):
			mask=np.exp(-(ws-ec)**2/2/bandWidth**2)
			mask+=np.exp(-(-ws-ec)**2/2/bandWidth**2) # apply mask to both sides
			mask/=np.amax(mask)
			Xs_d.append(ws) ; Ys_d.append(mask*ddos)			# nω vs nω
			Xs_v.append(ws) ; Ys_v.append(mask*vdos)			# nω vs nω
			masks.append(mask)

		markers=['k-']+rainbow(len(Xs_d)-1)
		plot(Xs_d,Ys_d,markers=markers,xlim=[0,Emax],ylim=[0,None],xlabel="frequency (THz)",ylabel="DOS",title="FFT(disp)",labels=[""]*len(Xs_d),filename=outDirec+"/DOS_d.png")
		plot(Xs_v,Ys_v,markers=markers,xlim=[0,Emax],ylim=[0,None],xlabel="frequency (THz)",ylabel="DOS",title="FFT(velo)",labels=[""]*len(Xs_v),filename=outDirec+"/DOS_v.png")


	out=np.zeros((len(ws)+1,len(energyCenters)+1))
	out[1:,0]=ws ; out[0,1:]=energyCenters
	for i,m in enumerate(masks):
		out[1:,i+1]=m
	header=["ws"]+["m"+str(i) for i in range(len(energyCenters)) ] 
	np.savetxt(outDirec+"/energyMasks.csv",out,header=",".join(header)+" # first row is center of energy bin",delimiter=",")


	return energyCenters,masks

def bandPassDisplacements(energyMask):
	print("calculate band pass on displacements")
	disp_bandpass=np.real(np.fft.ifft(dfft*energyMask[:,None,None],axis=0)) # ω,a,3 --> t,a,3 with band-pass filter applied to only select displacements associated with specific modes
	return disp_bandpass

def getFPconfigs(disp_bandpass,plotOut=False):

	#configIDs=np.linspace(0,len(ts),numFP,endpoint=False,dtype=int)+offsetFP # TODO might be a nice idea to use different atom configs for each tile? if so, multiply numFP by tiling[0]*tiling[1]*tiling[2] and then parse them out during tiling (maybe use an iterator for getting positions to tile?)
	#configIDs=np.linspace(0,len(ts),numFP*tiling[0]*tiling[1]*tiling[2],endpoint=False,dtype=int)+offsetFP
	#np.random.shuffle(configIDs)
	# the purpose of linspacing 0 to t for FP configs was to ensure "equal sampling" over the duration of the run.
	# but do we also want equal sampling (and then shuffling) for tiled configs?
	# also, the shuffling required to avoid stacking-in-order effects (z-periodicity resulting from t-periodicity) somewhat negates the linspace
	#configIDs=np.linspace(0,len(ts),numFP,endpoint=False,dtype=int)+offsetFP
	if numFP*tiling[0]*tiling[1]*tiling[2] > len(ts):
		print("WARNING! number of FP configs x number of tiles (",numFP*tiling[0]*tiling[1]*tiling[2],") > number of timesteps (",len(ts),"). some timestep configurations will be reused. if you want to avoid this, run your MD simulation for longer, or with finer time sampling")
	configIDs=np.random.randint(len(ts),size=numFP*tiling[0]*tiling[1]*tiling[2])
	if numFP==len(ts) and parallelFP==False:
		configIDs=np.arange(len(ts))

	#if restartFrom:
	#	configIDs+=np.random.randint(len(ts))
	#	configIDs[configIDs>=len(ts)]-=len(ts)

	print("configIDs",configIDs,len(configIDs))

	print("configIDs",configIDs,len(configIDs))

	# STEP 3, PRE-PROCESS THE DISPLACEMENTS
	print("assembling frozen phonon potential")
	pos=avg[None,:,:]+disp_bandpass[configIDs,:,:] # determine positions instead of displacements, only select every 50th position (for computational sanity)
	atomStack=[] ; lx,ly,lz=a*nx,b*ny,c*nz
	#for n,p in enumerate(pos):
	p=0
	for n in range(numFP):
		if max(tiling)>1:
			cloned=[]
			for i in range(tiling[0]):
				for j in range(tiling[1]):
					for k in range(tiling[2]):
						translate=np.asarray([i*lx,j*ly,k*lz])
						# 13 FP, tile 3 in x, tile 5 in y, tile 7 in z. there will be 13*3*5*7=1365 configs selected
						cloned.append(pos[p]+translate)	
						#print("FP",n,"tx",i,"ty",j,"tz",k,"=",configIDs[p])
						p+=1
			full=np.reshape(cloned,(len(pos[0])*len(cloned),3))
		else:
			full=pos[p] ; p+=1
		aseString=[ atomTypes[t-1]+"1" for t in types ]
		aseString="".join(aseString)
		aseString=aseString*(tiling[0]*tiling[1]*tiling[2])
		atoms = ase.Atoms(aseString,positions=full,pbc=[True,True,True],cell=[lx*tiling[0],ly*tiling[1],lz*tiling[2]])
		atomStack.append(atoms)
		box=np.asarray(atomStack[0].cell).flat[::4]
		#if m==0 and n==0:
		if isinstance(plotOut,int) and n==0:
			abtem.show_atoms(atoms,plane="xy"); plt.title("beam view")
			if CONVERGENCE!=0:
				if len(probePositions)>0:
					plt.plot(*np.asarray(probePositions).T)
				else:
					plt.plot(box[0]/2,box[1]/2)
			plt.savefig(outDirec+"/"+str(plotOut)+"_beam.png")
			if CONVERGENCE!=0:
				pP=np.asarray(probePositions)
				if len(pP)==0:
					pP=np.asarray([[box[0]/2,box[1]/2]])
				plt.xlim([min(pP[:,0])-5,max(pP[:,0])+5])
				plt.ylim([min(pP[:,1])-5,max(pP[:,1])+5])
				plt.savefig(outDirec+"/"+str(plotOut)+"_beam2.png")
			abtem.show_atoms(atoms,plane="yz"); plt.title("side view") ; plt.savefig(outDirec+"/"+str(plotOut)+"_side.png")
	return atomStack


# TODO ought to have a savewave function to capture all the if/else/for conditions for layerwise and convergence
# TODO need to figure out the "right" set of exports. e.g. validate if sum(realspace) is what we care about about and if we have any other use for the realspace file, then just save off sums to a text file. and validate if inversespace image can be remade from saved realspace (maybe we only should save off realspaces)

# DONE: validate if export(realspace.diffraction) == reload(realspace).diffraction. see ewave_to_psi.py, we can convert saved-off ewaves to reciprocal space (but not the other way) and it matches (although there's a wild scaling issue, idk why, maybe from missing params in loading in the ewave as an abtem.waves.Waves object). 
# DONE: validate dDOS calculations: sum(energyLoss(realspace)) == sum(energyLoss(reciprocal)) (or how different it will be). see checkloss.py
# DONE: how far out in diffraction space do we *actually* need? see checkloss.py. we apply a mask to the imports psi.npy files and see how far in we can go bafore it no longer agreed with ewave.npy

def matstrip(ary): # strip all len==1 indices out of an N-D array. shape 2,1,3,4,1,7 turns into shape 2,3,4,7. useful for getting rid of spurious axes
	shape=np.asarray(np.shape(ary))
	ones=np.where(shape==1)[0]
	for i in reversed(ones):
		ary=np.sum(ary,axis=i)
	return ary

def ewave(atomStack,fileSuffix,plotOut=False):
	print("ewave",fileSuffix)

	# CHECK IF THIS HAS ALREADY BEEN RUN
	lookForFile=outDirec+"/psi_"+fileSuffix # suffix order is: e, l, p. e will have been passed by caller
	if layerwise!=0:
		lookForFile=lookForFile+"_l0"
	npts=1
	if CONVERGENCE!=0 and len(probePositions)>1:
		npts=len(probePositions)
		lookForFile=lookForFile+"_p0"
	lookForFile=lookForFile+".npy"
	print(lookForFile)
	if os.path.exists(lookForFile):
		print("FIRST PSI-FILE FOUND FOR THIS ENERGY BIN. SKIPPING")
		print(lookForFile)
		return 0

	# STEP 4, SIMULATE E WAVES
	# 1. create the potential from the atomic configuration(s)
	# 2. define the probe
	# 3. denote where the probe is parked (create the GridScan object)
	# 4. calculate the exit wave
	# 5. transform exit wave to diffraction plane
	# 6. calculate coherent/incoherent components from (complex) wave function


	# STEP 1.a CREATE ATOM ARRANGEMENT
	frozen_phonons=abtem.AtomsEnsemble(atomStack)
	# STEP 1.b CREATE POTENTIAL
	print("setting up wave sim")
	box=np.asarray(atomStack[0].cell).flat[::4] ; print("cell size: ",box)
	potential_phonon = abtem.Potential(frozen_phonons, sampling=.05)#,slice_thickness=.01) # default slice thickness is 1Å
	nLayers=1
	if layerwise!=0:
		nslices=potential_phonon.num_slices # 14.5 Å --> potential divided into 33 layers
		nth=nslices//layerwise # 33 layers, user asked for 10? 33//10 --> every 3rd layer --> 11 layers total. 
		nth=max(nth,1)		# user asked for 1000? 33//1000 --> 0! oops! just give them every layer instead. 
		potential_phonon = abtem.Potential(frozen_phonons, sampling=.05,exit_planes=nth)#,slice_thickness=.01) # default slice thickness is 1Å
		nLayers=potential_phonon.num_exit_planes
		print("potential sliced into",nslices,"slices. keeping every",nth,"th layer, we'll have",nLayers,"exit waves")
		with open(outDirec+"/nLayers.txt",'w') as f:
			f.write(str(nLayers))
		# abtem/potentials/iam.py > class Potential() states: 
		# exit_planes : If 'exit_planes' is an integer a measurement will be collected every 'exit_planes' number of slices.
	if plotOut:
		print("preview potential")
		potential_phonon.show() ; plt.savefig(outDirec+"/potential.png")
	print("potential extent:",potential_phonon.extent) #; sys.exit()
	# STEP 2 DEFINE PROBE (or plane wave)
	if restartFrom:
		wavefile=restartDirec+"/ewave_"+fileSuffix+".npy" # TODO FOR NOW, WE'RE NOT ALLOWING RESTART FROM npts>1 OR nLayers>1
		print("GET PROBE FROM EXIT WAVE",wavefile)
		ewave=np.load(wavefile)
		if len(np.shape(ewave))==2:
			entrance_waves=abtem.waves.Waves(ewave,  energy=100e3, sampling=.05, reciprocal_space=False) 
		else:
			meta={'label': 'Frozen phonons', 'type': 'FrozenPhononsAxis' }
			meta=[abtem.core.axes.FrozenPhononsAxis(meta)]
			entrance_waves=abtem.waves.Waves(ewave,  energy=100e3, sampling=.05, reciprocal_space=False,ensemble_axes_metadata=meta)

		# if we don't do this, we get extent=n*sampling, when really, the sampling should be more of a recommendation (real sampling = size/n)
		entrance_waves.grid.match(potential_phonon) 
	
	elif CONVERGENCE!=0: # CONVERGENT PROBE INSTEAD OF PLANE WAVE (needs probe defined, and scan to denote where probe is positioned)
		probe = abtem.Probe(energy=100e3, semiangle_cutoff=CONVERGENCE) # semiangle_cutoff is the convergence angle
		probe.grid.match(potential_phonon)
		print(probe.gpts)

		if plotOut:
			print("preview probe")
			probe.show() ; plt.savefig(outDirec+"/probe.png")

		# STEP 3, PROBE POSITION (user defined, or center of the sim volume)
		if len(probePositions)>0:
			custom_scan=abtem.CustomScan(probePositions)
		else:
			custom_scan=abtem.CustomScan([box[0]/2, box[1]/2])
		print("scan positions:",custom_scan.get_positions())
		npts=len(custom_scan.get_positions())

		if modifyProbe: # your input file can define a function which does an in-plane modify of the complex probe function! 
			probewave=probe.build(custom_scan).compute()
			Z=probewave.array ; nx,ny=np.shape(Z)
			LX,LY=probe.extent
			xs=np.linspace(-LX/2,LX/2,nx) ; ys=np.linspace(-LY/2,LY/2,ny)
			modifyProbe(Z,xs,ys)
			contour(np.real(probewave.array).T,xs,ys,xlabel="x ($\AA$)",ylabel="y ($\AA$)",title="Real(probe)",filename=outDirec+"/probe_Re.png")
			contour(np.imag(probewave.array).T,xs,ys,xlabel="x ($\AA$)",ylabel="y ($\AA$)",title="Imag(probe)",filename=outDirec+"/probe_Im.png")


	else: # PLANE WAVE INSTEAD OF CONVERGENT PROBE (just define the plane wave params)
		plane_wave = abtem.PlaneWave(energy=100e3, sampling=0.05)


	# STEP 4 CALCULATE EXIT WAVE
	print("calculating exit waves") ; start=time.time()
	if restartFrom:
		exit_waves = entrance_waves.multislice(potential_phonon).compute()
		# PROBLEM:  stack of nFP entrance_waves and stack of nFP potential_phonon = nFP x nFP results! 
		#abtem/waves.py > class Waves > def multislice > 
		#	> potential = _validate_potential(potential, self)
		# 	> multislice_transform = MultisliceTransform(potential=potential, detectors=detectors)
		#	> waves = self.apply_transform(transform=multislice_transform)
		#	> return _reduce_ensemble(waves)
		#	abtem/multislice.py > class MultisliceTransform
		#	abtem/array.py > def apply_transform > abtem/transform.py > def apply
		# TODO WE'LL SELECT THE DIAGONAL FOR NOW, BUT IS THIS COMPUTATIONALLY HORRIBLE? SINCE WE CALCULATE NxN AND THROW OUT NxN-N?
		# yes, much slower. 5 FPs takes 6s to generate the first round (using a single probe), and 41s to generate second round. 6.8x longer
		# (6.8x longer to run 5x at once defeats the purpose of parallelFP)
		if len(np.shape(ewave))==3:
			ij=np.arange(numFP) ; ewave=exit_waves.array[ij,ij] 
			meta={'label': 'Frozen phonons', 'type': 'FrozenPhononsAxis' }
			meta=[abtem.core.axes.FrozenPhononsAxis(meta)]
			exit_waves=abtem.waves.Waves(ewave,  energy=100e3, sampling=.05, reciprocal_space=False,ensemble_axes_metadata=meta)
			entrance_waves.grid.match(potential_phonon) 
	elif CONVERGENCE!=0:
		if modifyProbe:
			exit_waves = probewave.multislice(potential_phonon).compute()
		else:
			exit_waves = probe.multislice(potential_phonon,scan=custom_scan).compute()
	else:
		exit_waves = plane_wave.multislice(potential_phonon).compute()
	print("(took",time.time()-start,"s)")
	print("ewave:",np.shape(exit_waves.array))
	
	# NOTE: shape of exit_waves can range from 2D to 6D depending on the potential. if atomStack is a list of atoms, we get nx,ny. 
	# if atomStack is list-of-configurations, add nFP. with layerwise, add nlayers. 
	# convergent beam gridscan adds probe positions x,y, and customscan adds singular probe position index. 
	# plane-wave: 		nFP,[nlayers],nx,ny
	# convergent gridscan: 	nFP,[nlayers],px,py,nx,ny # we don't do this anymore
	# convergent custom:	nFP,[nlayers],ps,nx,ny
	# OUTPUTTING: for any indices that are length 1, we get rid of them, via matstrip(). this means we can then fall back to variables npts, nLayers, and parallelFP to decide the dimensionality of whatever we're exporting and importing. any time npts>1 for example, we will loop through points and save each point's psi function separately. same goes for nLayers. only FPs are saved together.

	if saveExitWave:
		for l in range(nLayers):
			for p in range(npts): # for psi, we do: psi_e9_[fp9]_[l8]_[p7].npy,
				wavefile=outDirec+"/ewave_"+fileSuffix +\
					{True:"_l"+str(l),False:""}[nLayers>1] +\
					{True:"_p"+str(p),False:""}[npts>1] + ".npy"
				np.save(wavefile,matstrip(exit_waves.array))

	if plotOut:
		print("generating basic diffraction pattern")
		diffraction = exit_waves.diffraction_patterns(max_angle=maxrad,block_direct=False)#.interpolate(.01)
		diffraction.show() ; plt.savefig(outDirec+"/diffraction2.png")
		kx=diffraction.sampling[-2]*diffraction.shape[-2]/2 ; ky=diffraction.sampling[-1]*diffraction.shape[-1]/2
		kxs=np.linspace(-kx,kx,diffraction.shape[-2]) ; kys=np.linspace(-ky,ky,diffraction.shape[-1])
		np.save(outDirec+"/kxs.npy",kxs) ; np.save(outDirec+"/kys.npy",kys)

		diffarray=diffraction.array ; print("np.shape(diffarray)",np.shape(diffarray))
		if nLayers>1:	# select *last* layer for diffraction pattern
			diffarray=diffarray[:,-1]
		if npts>1: # for psi, we'll do: psi_e9_[fp9]_[l8]_[p7].npy, but for diff, one fp is fine, last layer is fine. only loop through points
			for p in range(npts):
				diff=diffarray[:,p]
				np.save(outDirec+"/diff_p"+str(p)+".npy",matstrip(diff))
				diff=np.sum(diff,axis=0) # sum FP axis
				contour(np.log(diff.T),kxs,kys,filename=outDirec+"/diffraction.png",xlabel="$\AA$^-1",ylabel="$\AA$^-1")
		else:
				diff=np.sum(diffarray,axis=0)
				diff=matstrip(diff)
				np.save(outDirec+"/diff.npy",diff)
				out=np.log(diff.T) ; out[out<-23]=-23
				contour(out,kxs,kys,filename=outDirec+"/diffraction.png",xlabel="$\AA$^-1",ylabel="$\AA$^-1")
	else:
		kxs=np.load(outDirec+"/kxs.npy") ; kys=np.load(outDirec+"/kys.npy")

	# STEP 5 CONVERT TO DIFFRACTION SPACE
	print("converting exit wave to k space",end=" ")
	sys.stdout.flush() # https://stackoverflow.com/questions/35230959/printfoo-end-not-working-in-terminal
	start=time.time()
	diffraction_complex = exit_waves.diffraction_patterns(max_angle=maxrad,block_direct=False,return_complex=True)
	print("took:",time.time()-start)
	zr=diffraction_complex.array

	print("saving off psi",end=" ")
	sys.stdout.flush() # https://stackoverflow.com/questions/35230959/printfoo-end-not-working-in-terminal
	start=time.time()
	
	# RECALL: SHAPE OF exit_waves, with exit_planes arg in potential_phonon:
	# plane-wave: 		nFP,[nlayers],nx,ny
	# convergent gridscan: 	nFP,[nlayers],px,py,nx,ny # we don't do this anymore
	# convergent custom:	nFP,[nlayers],[ps],nx,ny

	# for psi, we'll do: psi_e9_[fp9]_[l8]_[p7].npy,
	for l in range(nLayers):
		for p in range(npts): # below: filesuffix includes psi_e9_[fp9]
			psifile=outDirec+"/psi_"+fileSuffix +\
				{True:"_l"+str(l),False:""}[nLayers>1] +\
				{True:"_p"+str(p),False:""}[npts>1] + ".npy"
			zo=zr
			if nLayers>1:
				zo=zr[:,l]
			if npts>1:
				zo=zo[:,p]
			zo=matstrip(zo)
			np.save(psifile,zo)
	print("took:",time.time()-start)

	return zr # even if layerwise!=0, we'll end up with the last (thickest) psi. 

def energyLoss(psi,plotOut=False,mask=None):
	#print("calculating energy losses")
	# COMPARING TO ZEIGER PRB 104, 104301 (2021) : They list:
	# They use Ψ(q,r,R), where q is momentum, r is position of STEM probe, R is frozen-phonon configurations
	# Our exit wave appears to be: Ψ(R,rx,ry,x,y), with 2 positional indices (x,y, from our scan)
	# and our exit wave in real-space. to convert, use return_complex arg for diffraction_patterns: Ψ(R,rx,ry,ky,kx)
	#np.save(outDirec+"/ewave_"+str(m)+".npy",exit_waves.array[:,0,0,:,:])
	if mask is not None:
		print("energyLoss, applying mask")
		psi[:,:,:]*=mask[None,:,:]
	#psi_o=np.load(outDirec+"/psi_avg.npy")
	#kxs=np.load(outDirec+"/kxs.npy") ; kys=np.load(outDirec+"/kys.npy")
	#psi=psi-psi_o[None,:,:]

	#angles=np.zeros((len(kxs),len(kys)))
	#for fp in tqdm(range(len(psi))):
	#	for i in range(len(kxs)):
	#		for j in range(len(kys)):
	#			p1=psi[fp,i,j] ; p2=psi_o[i,j]
	#			v1=[np.real(p1),np.imag(p1)] ; v2=[np.real(p2),np.imag(p2)]
	#			m1=np.absolute(p1) ; m2=np.absolute(p2)
	#			c=np.dot(v1,v2)/m1/m2
	#			if c>1:
	#				c=1
	#			if c<-1:
	#				c=-1
	#			#else:
	#			angles[i,j]+=np.arccos(c)
	#angles/=len(psi)
	#Ivib=angles #np.mean(crosses,axis=0)

	#dpsi=psi-psi_o[None,:,:]
	#Ivib=np.absolute(dpsi)**2/np.absolute(psi_o[None,:,:])**2
	#Ivib=np.sum(Ivib,axis=0)/len(psi)

	nFP=len(psi) ; print("nFP",nFP)
	abpsisq=np.absolute(psi)**2 # incoherent = 1/N ⟨ | Ψ(q,r,Rₙ(ω)) |² ⟩ₙ (where "⟨ ⟩ₙ" means Σₙ)
	inco=1/nFP*np.sum(abpsisq,axis=0) # sum FP configs, keep kx axis
	sumpsi=1/nFP*np.sum(psi,axis=0) # coherent =  | 1/N ⟨ Ψ(q,r,Rₙ(ω)) ⟩ₙ | ²
	cohe=np.absolute(sumpsi)**2
	Ivib=inco-cohe

	if plotOut:
		kxs=np.load(outDirec+"/kxs.npy") ; kys=np.load(outDirec+"/kys.npy")
		contour(np.log(cohe).T,kxs,kys,filename=outDirec+"/cohe_"+str(plotOut)+".png",xlabel="$\AA$^-1",ylabel="$\AA$^-1",title=str(energyCenters[plotOut])+" THz")
		contour(np.log(inco).T,kxs,kys,filename=outDirec+"/inco_"+str(plotOut)+".png",xlabel="$\AA$^-1",ylabel="$\AA$^-1",title=str(energyCenters[plotOut])+" THz")
		#kxs=np.load(outDirec+"/kxs.npy") ; kys=np.load(outDirec+"/kys.npy")
		#zlim=max(abs(np.amin(Ivib)),abs(np.amax(Ivib))) ; zlim=[-zlim,zlim] ; print("ZLIM",zlim)
		contour(Ivib.T,kxs,kys,filename=outDirec+"/Ivib_"+str(plotOut)+".png",xlabel="$\AA$^-1",ylabel="$\AA$^-1",title=str(energyCenters[plotOut])+" THz")#,zlim=zlim)
	return Ivib

# END GENERAL FUNCTIONS


# POST PROCESSING CODE:
# nukeBadFP: occasionally we'll get nans in our psifiles. fix those
# psifileToIvib: "flatten" a psifile stack: exit-wave-per-FP (3D) --> energyLoss
#   calculation --> kspace energy slice (2D) --> ivib file. this means all 
#   subsequent analysis code can be very efficient, as it simply needs to scoop 
#   up the ivib file (instead of re-processing the psi files)
# preprocess: loop through all psifiles, call psifileToIvib
# eDOS: simply loop through energy bins, scoop up ivib files, sum, and plot
# layerDOS: loop through layers, calling eDOS

# on rare occasion (don't yet know why) psi files will have nan for all entries for one (or more) frozen phonon configuration. energyLoss will then choke on it (summing across FP configs, where one is nan, will produce nan). run this on the problematic file (assumes parallelFP=True, ie, psi file indices are nFP,nkx,nky). In the future we may add a check to energyLoss and a call to here, but I want to keep an eye on it. 
# Harrison saw an issue with this. deleting the psi-file and re-running did *not* fix the issue (had to run nukeBadFP anyway)
def nukeBadFP(psifile):
	psi=np.load(psifile)
	where=np.where(np.isnan(psi))
	badFPids=list(set(where[0]))
	where=np.where(psi>1e12)
	badFPids=badFPids+list(set(where[0]))
	mask=np.zeros(len(psi)) ; mask[badFPids]=1
	psi=psi[mask!=1]
	os.rename(psifile,psifile.replace(".npy","_bak.npy"))
	np.save(psifile,psi)
	
# pass either a single 3D psifile (nFP,nkx,nky), OR a list of 2D psifiles (one per FP, nkx,nky) and we'll read it in, call energyLoss(), and save off the result. this makes later remasking faster (read in ivib file, instead of needing to read in full psi file(s) and re-calculating)
def psifileToIvib(psifiles,e):
	print(psifiles)
	# check if ivib file exists
	if isinstance(psifiles,list):
		outfile=psifiles[0].replace("_fp0","").replace("psi_","ivib_")
	else:
		outfile=psifiles.replace("psi_","ivib_")
	# if so, return contents (skip re-reading psifile and recalculation)
	if os.path.exists(outfile):
		print("psifileToIvib, found outfile",outfile,", skipping")
		return np.load(outfile)
	# for parallelFP=False, we need to assemble psi wave stack of FP configs
	if isinstance(psifiles,list):
		print("parallFP=False, assembling psi")
		pwave0=np.load(psifiles[0])[:,:]
		pwaves=pwave0[None,:,:]*np.ones(len(psifiles))[:,None,None]
		for fp,psifile in enumerate(psifiles):
			if not os.path.exists(fp):
				break
			pwaves[fp]=np.load(psifile)[:,:]
	else:
		pwaves=np.load(psifiles)
	# calculate eloss
	#plotOut={True:e,False:False}[plotDiffractions] # pass the energy band number (gets appended to filename), ifof pD=True
	print("calling energyLoss")
	Ivib=energyLoss(pwaves,plotOut=False)
	np.save(outfile,Ivib)
	return Ivib

def preprocessOutputs():
	infile=sys.argv[-1]
	readInputFile(infile)
	# LOOPING: if layerwise!=0, calculate for each layer. if CONVERGENCE!=0 & len(probePositions)>1, calculate for each probe position. if parallelFP==False, MUST ASSEMBLE PSI STACK and then calculate Ivib once over the whole stack
	ecdata=np.loadtxt(outDirec+"/energyMasks.csv",delimiter=",")
	global energyCenters
	energyCenters=ecdata[0,1:]
	npts=1 ; nLayers=1
	if CONVERGENCE!=0 and len(probePositions)>1:
		npts=len(probePositions)
	if layerwise!=0:
		nLayers=int(open(outDirec+"/nLayers.txt").readlines()[0])
	for e in range(Ebins):
		for p in range(npts):
			for l in range(nLayers):
				if parallelFP:
					psifile=outDirec+"/psi_e"+str(e) +\
						{True:"_l"+str(l),False:""}[nLayers>1] +\
						{True:"_p"+str(p),False:""}[npts>1] +".npy"
				else:
					psifile=[ outDirec+"/psi_e"+str(e)+"_fp"+str(fp) +\
						{True:"_l"+str(l),False:""}[nLayers>1] +\
						{True:"_p"+str(p),False:""}[npts>1] +".npy" for fp in range(numFP) ]
				psifileToIvib(psifile,e)

# layerDOS > eDOS > psifileToIvib > energyLoss

# CALCULATE energyLoss() FOR EACH ENERGY BIN (for a given layer, point, k-space mask)
def eDOS(layer=-1,point=0,mask=None,plotDiffractions=True):
	infile=sys.argv[-1]
	readInputFile(infile)
	ecdata=np.loadtxt(outDirec+"/energyMasks.csv",delimiter=",")
	global energyCenters
	energyCenters=ecdata[0,1:]
	npts=1 ; nLayers=1
	if CONVERGENCE!=0 and len(probePositions)>1:
		npts=len(probePositions)
	if layerwise!=0:
		nLayers=int(open(outDirec+"/nLayers.txt").readlines()[0])
	if nLayers>1:
		layers=list(range(nLayers)) ; layer=layers[layer] ; print("layer",layer)# handle "-1" to denote "last layer"

	outfile = outDirec+"/eDOS" + {True:"_l"+str(layer),False:""}[nLayers>1] + {True:"_p"+str(point),False:""}[npts>1]
	if mask is None and os.path.exists(outfile+".txt"):
		print("RELOADING DOS FROM SAVED FILE",outfile+".txt")
		out=np.loadtxt(outfile+".txt")
		xs,ys=out.T
		return xs,ys

	xs=[] ; ys=[]
	for e in tqdm(range(len(energyCenters))):
		# psi_e9_[fp9]_[l8]_[p7].npy : fp ifof parallelFP=True, l ifof layerwise!=0, p ifof CONVERGENCE!=0 & len(probePositions)>1
		if parallelFP:
			psifile=outDirec+"/psi_e"+str(e) +\
				{True:"_l"+str(layer),False:""}[nLayers>1] +\
				{True:"_p"+str(point),False:""}[npts>1] +".npy"
		else:
			psifile=[ outDirec+"/psi_e"+str(e)+"_fp"+str(fp) +\
				{True:"_l"+str(layer),False:""}[nLayers>1] +\
				{True:"_p"+str(point),False:""}[npts>1] +".npy" 
				for fp in range(numFP) ]
		print("reading psi file",psifile)
		Ivib=psifileToIvib(psifile,e) # EITHER process psifile(s) OR read in saved ivib file
		if mask is not None:
			Ivib*=mask
		if plotDiffractions:
			kxs=np.load(outDirec+"/kxs.npy") ; kys=np.load(outDirec+"/kys.npy")
			#zlim=max(abs(np.amin(Ivib)),abs(np.amax(Ivib))) ; zlim=[-zlim,zlim] ; print("ZLIM",zlim)
			contour(Ivib.T,kxs,kys,filename=outDirec+"/Ivib_"+str(e)+".png",xlabel="$\AA$^-1",ylabel="$\AA$^-1",title=str(energyCenters[e])+" THz")#,zlim=zlim,cmap='Spectral')


		xs.append(energyCenters[e]) ; ys.append(np.sum(Ivib))
	# Ivib = 1/N Σₙ[ | Ψ |² ] - 1/N | Σₙ[ Ψ ] | ²
	#outfile = outDirec+"/eDOS" + {True:"_l"+str(layer),False:""}[nLayers>1] + {True:"_p"+str(point),False:""}[npts>1]
	plot([xs],[ys],xlabel="Frequency (THz)",ylabel="$\Sigma$ |$\Psi$|^2 - | $\Sigma$ $\Psi$ |^2 (-)",filename=outfile+".png",markers=['k-'])
	if mask is None:
		out=np.zeros((len(energyCenters),2)) ; out[:,0]=energyCenters ; out[:,1]=ys
		np.savetxt(outfile+".txt",out)
	return xs,ys

# LOOP THROUGH LAYERS, CALLING eDOS (sum over k for each energy bin, for each layer, generate EELS spectrum at each layer)
def layerDOS(mask=None,maskname=""): # prototype of this code in figs_abEELS_AlN_SED_01a/checkeloss.py
	infile=sys.argv[-1]
	readInputFile(infile)
	# if layerwise!=0:
	Xs=[] ; Ys=[] ; lbls=[]
	# LOOPING: if layerwise!=0, calculate for each layer. if CONVERGENCE!=0 & len(probePositions)>1, calculate for each probe position. if parallelFP==False, MUST ASSEMBLE PSI STACK and then calculate Ivib once over the whole stack
	npts=1 ; nLayers=1
	if CONVERGENCE!=0 and len(probePositions)>1:
		npts=len(probePositions)
	if layerwise!=0:
		nLayers=int(open(outDirec+"/nLayers.txt").readlines()[0])

	for p in range(npts):
		Xs=[] ; Ys=[] ; lbls=[]
		for l in range(nLayers):
			print("point",p,"layer",l)
			xs,ys=eDOS(l,p,mask=mask,plotDiffractions=(l==nLayers-1))
			Xs.append(xs) ; Ys.append(ys) ; lbls.append("")
		outfile=outDirec+"/layerDOS" +\
			{True:"_p"+str(p),False:""}[npts>1] +\
			{True:"",False:"_"+maskname}[ mask is None ]
		out=np.zeros((len(xs),len(Xs)+1)) ; out[:,0]=xs
		for i,ys in enumerate(Ys):
			out[:,i+1]=ys
		np.savetxt(outfile+".txt",out)
		plot(Xs,Ys,xlabel="Frequency (THz)",ylabel="Energy Loss (a.u.)",labels=lbls,title=outDirec,markers=rainbow(nLayers),xlim=[0,Xs[0][-1]],ylim=[0,None],filename=outfile+".png")
	return Xs,Ys

def processKmasks(preview=True):

	kxs=np.load(outDirec+"/kxs.npy") ; kys=np.load(outDirec+"/kys.npy")
	diff=np.load(outDirec+"/diff.npy")

	print("ASSEMBLE MASKS")
	global kmask_lbl
	overplot=[] ; masks=[]

	for i,(xi,xf,yi,yf) in enumerate(zip(kmask_xis,kmask_xfs,kmask_yis,kmask_yfs)):
		print("rectangular mask:",np.round(xi,4),"<= x <=",np.round(xf,4),",",np.round(yi,4),"<= y <=",np.round(yf,4))
		mask=np.ones((len(kxs),len(kys)))
		mask[:,kys<yi]=0 ; mask[:,kys>yf]=0
		mask[kxs<xi,:]=0 ; mask[kxs>xf,:]=0
		masks.append(mask)
		overplot.append({"kind":"line","xs":[xi,xi,xf,xf,xi],"ys":[yi,yf,yf,yi,yi]})
		overplot.append({"kind":"text","xs":[(xi+xf)/2],"ys":[(yi+yf)/2],"text":[kmask_lbl[len(masks)-1]]})
	for i,(cx,cy,r) in enumerate(zip(kmask_cxs,kmask_cys,kmask_rad)):
		if isinstance(r,(float,int)):
			print("circular mask: center = (",np.round(cx,4),",",np.round(cy,4),") , r = ",np.round(r,4)) ; r=[r]
		else:
			print("nested circular mask: center = (",np.round(cx,4),",",np.round(cy,4),") , r = ",[ np.round(v,4) for v in r])
		mask=np.zeros((len(kxs),len(kys)))
		radii=np.sqrt( (kxs[:,None]-cx)**2+(kys[None,:]-cy)**2 )
		pm=1
		for v in reversed(sorted(r)):
			mask[radii<=v]=pm ; pm=(pm+1)%2 ; print(pm)
		masks.append(mask)
		for v in r:
			xline=np.linspace(cx-v*.99,cx+v*.99,100) ; yline=np.sqrt(v**2-(xline-cx)**2)
			overplot.append({"kind":"line","xs":list(xline)+list(xline[::-1]),"ys":list(cy+yline)+list(cy-yline),'c':'k'})
		overplot.append({"kind":"text","xs":[cx],"ys":[cy],"text":[kmask_lbl[len(masks)-1]],'c':'k'})

	lblkwargs={"horizontalalignment":"center","verticalalignment":"center","size":10}
	for op in overplot:
		if op["kind"]!="text":
			continue
		for k in lblkwargs:
			op[k]=lblkwargs[k]
	#print(overplot)
	if preview:
		print("PREVIEW MASKS")
		for i,mask in enumerate(masks):
			diff[mask==1]*=10000
		contour(np.log(diff).T,kxs,kys,xlabel="$\AA$^-1",ylabel="$\AA$^-1",overplot=overplot,filename=outDirec+"/diffmasks.png")

	return masks

# maskDOS > layerDOS > eDOS > psifileToIvib > energyLoss
def maskDOS():
	infile=sys.argv[-1]
	readInputFile(infile)
	print(outDirec)

	masks=processKmasks()

	print("PROCESS EWAVES FOR MASKS")
	#Xs=[] ; Ys=[]
	for i,mask in enumerate(masks):
		# layerDOS (layerDOS_maskname.png, layerDOS_maskname.txt) > eDOS (Ivib...png, eDOS...png)
		X,Y=layerDOS(mask=mask,maskname=kmask_lbl[i])
		#shutil.move(outDirec+"/postProcessDOS.txt",outDirec+"/postProcessDOS_"+str(i)+".txt")
		picDirec=outDirec+"/Ivib_"+kmask_lbl[i]
		os.makedirs(picDirec,exist_ok=True)
		for f in glob.glob(outDirec+"/Ivib_*.png")+[outDirec+"/layerDOS_"+kmask_lbl[i]+".png",outDirec+"/layerDOS_"+kmask_lbl[i]+".txt"]+glob.glob(outDirec+"/eDOS_*.png"):
			shutil.move(f,picDirec)
		#plot(Xs,Ys,markers=rainbow(len(Xs)),title=kmask_lbl[i])

	#Xs=[] ; Ys=[]
	#ecdata=np.loadtxt(outDirec+"/energyMasks.csv",delimiter=",")
	#energyCenters=ecdata[0,1:]

	#for i in range(len(masks)):
	#	data=np.loadtxt(outDirec+"/postProcessDOS_"+str(i)+".txt")
	#	Xs.append(data[:,0]) ; Ys.append(data[:,-1])
	
	#plot(Xs,Ys,markers=rainbow(len(Xs)),filename=outDirec+"/maskedDOS.png",labels=kmask_lbl)

# slopefit: y,x indices! for an intensity field z as a function of x and y, find the line that best bisects it (used for fitting on diffraction spots)
# how does it work? "center of mass in x and in y" are easy: cx=np.sum(zs*xs[None,:])/np.sum(zs), which is simply "weight each value by it's distance in x" and so on. we're calculating the "moment" about zero, mass*distance. for slope, we first find the centers, then calculate the slope to each point from the center: slopes=(ys[None,:]-cy)/(xs[:,None]-cx) and the distance to each point l=np.sqrt((xs[:,None]-cx)**2+(ys[None,:]-cy)**2). then we weight the slopes by the mass and the distance from the center: m=np.sum(Ivib*slopes*l)/np.sum(Ivib*l)
# try it yourself, here's a couple useful example distributions to try: 
# zs=np.exp(-(xs[None,:]-ys[:,None])**2/2/2**2)*np.exp(-(ys[:,None]-5)**2/2/2**2) # for a slanted gaussian centered at 5,5
# zs=np.exp(-((xs[None,:]+5)-2*ys[:,None])**2/2/2**2)*np.exp(-(ys[:,None]-5)**2/2/2**2) # same, but with a different slope
def slopefit(zs,xs,ys,cx=None,cy=None):
	if cx is None:
		cx=np.sum(zs*xs[None,:])/np.sum(zs)
	if cy is None:
		cy=np.sum(zs*ys[:,None])/np.sum(zs)
	xc=xs-cx ; yc=ys-cy
	slopes=yc[:,None]/xc[None,xc!=0]
	distances=np.sqrt(xc[None,:]**2+yc[:,None]**2)
	m =np.sum(zs[:,xc!=0]*slopes*distances[:,xc!=0])/np.sum(zs[:,xc!=0]*distances[:,xc!=0])
	islopes=xc[None,:]/yc[yc!=0,None]
	im=np.sum(zs[yc!=0,:]*islopes*distances[yc!=0,:])/np.sum(zs[yc!=0,:]*distances[yc!=0,:])
	overplot=[ {"xs":xs,"ys":m*(xs-cx)+cy,"kind":"line","c":"r","linestyle":":"},
		{"xs":im*(ys-cy)+cx,"ys":ys,"kind":"line","c":"r","linestyle":":"} ]
	contour(zs,xs,ys,overplot=overplot)
	return m,cx,cy

def kmaskCOG():
	infile=sys.argv[-1]
	readInputFile(infile)
	print(outDirec)

	masks=processKmasks(preview=False)
	print(masks)

	ecdata=np.loadtxt(outDirec+"/energyMasks.csv",delimiter=",")
	global energyCenters
	energyCenters=ecdata[0,1:]
	npts=1 ; nLayers=1 ; layer=0 ; point=0
	if CONVERGENCE!=0 and len(probePositions)>1:
		npts=len(probePositions)
	if layerwise!=0:
		nLayers=int(open(outDirec+"/nLayers.txt").readlines()[0])
	if nLayers>1:
		layers=list(range(nLayers)) ; layer=layers[layer] ; print("layer",layer)# handle "-1" to denote "last layer"
	kxs=np.load(outDirec+"/kxs.npy") ; kys=np.load(outDirec+"/kys.npy")	

	for e in tqdm(range(len(energyCenters))):
		overplot=[]
		for i,mask in enumerate(masks):
			print(i,mask,mask.shape)
			# psi_e9_[fp9]_[l8]_[p7].npy : fp ifof parallelFP=True, l ifof layerwise!=0, p ifof CONVERGENCE!=0 & len(probePositions)>1
			if parallelFP:
				psifile=outDirec+"/psi_e"+str(e) +\
					{True:"_l"+str(layer),False:""}[nLayers>1] +\
					{True:"_p"+str(point),False:""}[npts>1] +".npy"
			else:
				psifile=[ outDirec+"/psi_e"+str(e)+"_fp"+str(fp) +\
					{True:"_l"+str(layer),False:""}[nLayers>1] +\
					{True:"_p"+str(point),False:""}[npts>1] +".npy" 
					for fp in range(numFP) ]
			print("reading psi file",psifile)
			Ivib=psifileToIvib(psifile,e) # EITHER process psifile(s) OR read in saved ivib file
			Ivib*=mask
			m,cx,cy=slopefit(Ivib.T,kxs,kys,cx=kmask_cxs[i],cy=kmask_cys[i])
			# center of mass of disk:
			#cx=np.sum(Ivib*kxs[:,None])/np.sum(Ivib) # prototyped in figs_abEELS_AlN_thiccc/COM.py
			#cy=np.sum(Ivib*kys[None,:])/np.sum(Ivib)
			# this isn't actually what we want to do? this will tell us the "offcenteredness" of the disks. 
			# we want, like, a line through the disk at some angle like they do in https://arxiv.org/pdf/2407.08982
			overplot.append({ "xs":kxs,"ys":m*(kxs-cx)+cy,"kind":"line","c":"r"})

		Ivib=psifileToIvib(psifile,e)
		for m in masks:
			Ivib*=(10+m)
		contour(Ivib.T,kxs,kys,xlabel="$\AA$^-1",ylabel="$\AA$^-1",title=str(energyCenters[e])+" THz",overplot=overplot)#,zlim=zlim,cmap='Spectral')


		#if plotDiffractions:
		#kxs=np.load(outDirec+"/kxs.npy") ; kys=np.load(outDirec+"/kys.npy")
		#zlim=max(abs(np.amin(Ivib)),abs(np.amax(Ivib))) ; zlim=[-zlim,zlim] ; print("ZLIM",zlim)
		#contour(Ivib.T,kxs,kys,xlabel="$\AA$^-1",ylabel="$\AA$^-1",title=str(energyCenters[e])+" THz",overplot=overplot)#,zlim=zlim,cmap='Spectral')


		#xs.append(energyCenters[e]) ; ys.append(np.sum(Ivib))

	
	#print("PROCESS EWAVES FOR MASKS")
	##Xs=[] ; Ys=[]
	#for i,mask in enumerate(masks):
	#	# layerDOS (layerDOS_maskname.png, layerDOS_maskname.txt) > eDOS (Ivib...png, eDOS...png)
	#	X,Y=layerDOS(mask=mask,maskname=kmask_lbl[i])
	#	#shutil.move(outDirec+"/postProcessDOS.txt",outDirec+"/postProcessDOS_"+str(i)+".txt")
	#	picDirec=outDirec+"/Ivib_"+kmask_lbl[i]
	#	os.makedirs(picDirec,exist_ok=True)
	#	for f in glob.glob(outDirec+"/Ivib_*.png")+[outDirec+"/layerDOS_"+kmask_lbl[i]+".png",outDirec+"/layerDOS_"+kmask_lbl[i]+".txt"]+glob.glob(outDirec+"/eDOS_*.png"):
	#		shutil.move(f,picDirec)


# use settings restartFrom="..." and saveExitWave=True. the exit wave from one run will be used as the entrance wave (instead of a probe) for the next run, which will allow simulating infinitely-thick samples without needing to build out a ridiculously thick potential. 
# BUT, this means a huge number of beam view images are created (annoying) and wave files are stored for each run (wastes space). 
# here, we loop through all cycles, delete extra image files, and delete all but last two cycles' wave files
def cycleCleanup():
	infile=sys.argv[-1]
	#readInputFile(infile) # oops! this will create the next folder! 
	outDirec="figs_abEELS_"+infile.strip("/").split("/")[-1].split(".")[0]
	print(outDirec)
	direcs=[outDirec]
	for i in range(100000):
		if os.path.exists(outDirec+"_cycle"+str(i)):
			ewaves=glob.glob(outDirec+"_cycle"+str(i)+"/ewave*.npy")
			if len(ewaves)==50:
				direcs.append(outDirec+"_cycle"+str(i))
			#else:
			#	print("ignore incomplete directory",outDirec+"_cycle"+str(i))
			#	break
		else:
			break
	print(direcs)
	#return
	for direc in direcs:
		pics=glob.glob(direc+"/*beam*.png")+glob.glob(direc+"/*side.png")
		print("remove",pics)
		for p in pics:
			os.remove(p)
	# do not delete wave files out of the LAST folder (might want it later), or SECOND TO LAST (if last is still being generated, it might be operating out of the second-to-last
	for i,direc in enumerate(direcs):
		if i<len(direcs)-2:
			wavefiles=glob.glob(direc+"/ewave_*.npy")
			print("remove",wavefiles)
			for w in wavefiles:
				os.remove(w)

# use settings restartFrom="..." and saveExitWave=True. the exit wave from one run will be used as the entrance wave (instead of a probe) for the next run, which will allow simulating infinitely-thick samples without needing to build out a ridiculously thick potential. 
# This only works well for numFP=1, so you should run many for each different FP config, and then we'll combine them here
def cycleCombiner():
	#baseFPs=[ "AlN_"+l for l in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" ] # YOU NEED TO EDIT THIS LINE TO POINT TO YOUR SHIT
	#print(baseFPs)
	inputFiles=sys.argv[2:]
	names=[ f.split("/")[-1].replace(".txt","") for f in inputFiles ]
	#for i in range(100000):
	for i in [54]:
		if i==0:
			direcs=[ "figs_abEELS_"+n for n in names ]
		else:
			direcs=[ "figs_abEELS_"+n+"_cycle"+str(i-1) for n in names ]
		if not os.path.exists(direcs[0]):
			break
		append="_l"+str(i)
		print("CALLING psiCombiner:",direcs,append)
		psiCombiner(direcs,append)

def psiCombiner(direcs,append):
	#append="_l4" ; direcs=sys.argv[2:]
	#append=sys.argv[2] ; direcs=sys.argv[3:]
	psifiles=[ glob.glob(d+"/psi*.npy") for d in direcs ] # each direc gets a list of psi files
	os.makedirs("psiCombined",exist_ok=True)
	for f1 in psifiles[0]: # loop through all files in first direc
		name=f1.replace(direcs[0],"").replace("/","") # get file's name (excluding directory name)
		outfile="psiCombined/"+name.replace(".npy",append+".npy")
		if os.path.exists(outfile):
			print("skip",name)
			continue
		selected=[ d+"/"+name for d in direcs ]
		psis=[]
		for s in selected:
			if os.path.exists(s):
				psis.append(np.load(s))
				print(s,np.shape(psis[-1]))
			else:
				print("WARNING: missing psi file",s)
		psi=np.asarray(psis)
		print(name,np.shape(psi))
		np.save(outfile,psi)

def fakePhononCycles():
	infile=sys.argv[-1]
	readInputFile(infile)
	lines=open(infile,'r').readlines()+["\n"]*4 # add dummy lines so we can set appropriate variables
	newinfile=infile.replace(".txt","-edited.txt")

	phases=np.linspace(0,2*np.pi,numPhases,endpoint=False)	
	for i,p in enumerate(phases):
		print("RUNNING FOR PHASE",p)

		# EDIT LINES OF INPUT FILE
		lines[-2]="numFP=1 ; parallelFP=False ; offsetFP=0 \n"
		#lines[22]='addModes={"A":['+A+'],"w":[0],"k":['+iA+'*2*np.pi],"pdirec":[[1,1,0]],"vdirec":[[1,1,0]],"phase":['+str(p)+']}\n'
		addModes["phase"]=[p+pr for pr in addModes["phase_rel"]]
		lines[-1]="addModes="+str(addModes)+"\n"

		# SAVE OFF INPUT FILE
		with open(newinfile,'w') as f:
			for l in lines:
				f.write(l)
		# RUN NEW INPUT FILE
		sys.argv[-1]=newinfile
		main()
		# COPY OFF RESULTS
		os.makedirs(outDirec+"/FPs",exist_ok=True)
		shutil.copy(outDirec+"/psi_e0_fp0.npy",outDirec+"/FPs/psi_e0_fp"+str(i)+".npy")
		shutil.copy(outDirec+"/0_beam.png",outDirec+"/FPs/"+str(i)+".png")
		os.makedirs(outDirec+"/phase"+str(i),exist_ok=True)
		files=glob.glob(outDirec+"/*") ; print("MOVING",files)
		files=[ f for f in files if "phase" not in f and "FP" not in f ]
		for f in files:
			shutil.move(f,outDirec+"/phase"+str(i)+"/")
	# COPY FP CONFIGS BACK OUT TO MAIN OUTPUT FOLDER AND RENAME. ALSO COPY REQUIRED FILES FOR IVIB
	for i,p in enumerate(phases):
		shutil.copy(outDirec+"/phase"+str(i)+"/psi_e0_fp0.npy",outDirec+"/psi_e0_fp"+str(i)+".npy")
		shutil.copy(outDirec+"/phase"+str(i)+"/0_beam.png",outDirec+"/"+str(i)+"_beam.png")
	for f in ["energyMasks.csv","kxs.npy","kys.npy"]:
		shutil.copy(outDirec+"/phase0/"+f,outDirec+"/"+f)

	# EDIT LINES OF INPUT FILE, FOR FULL NUMBER OF FP CONFIGS
	lines[-2]="numFP="+str(numPhases)+" ; parallelFP=False ; offsetFP=0 \n"
	with open(newinfile,'w') as f:
		for l in lines:
			f.write(l)
	# RUN DOS
	eDOS()

def cycle():			# use settings restartFrom="..." and saveExitWave=True. the exit wave from one run will be used as the entrance
	while True:	 	# wave (instead of a probe) for the next run, which will allow simulating infinitely-thick samples without 
		main()		# needing to build out a ridiculously thick potential

funcAliases={"pre":"preprocessOutputs","DOS":"eDOS","cc":"cycleCleanup"}
# if called directly, run main(). does not run if imported "from abEELS import *"
if __name__=='__main__':
	if len(sys.argv)==2:
		main()
	else:
		fun=sys.argv[1] 
		if fun in funcAliases:
			fun = funcAliases[ fun ]
		c=fun+"()"
		exec(c)

	#elif sys.argv[1]=="pre":
	#	preprocessOutputs()
	#elif sys.argv[1]=="DOS":
	#	eDOS()
	#elif sys.argv[1]=="layerDOS":
	#	layerDOS()
	#elif sys.argv[1]=="maskDOS":
	#	maskDOS()
	#elif sys.argv[1]=="cycle":
	#	while True:
	#		main()
	#elif sys.argv[1]=="cc":
	#	cycleCleanup()
	#elif sys.argv[1]=="psiCombiner":	# if you run "python3 abEELS.py psiCombiner figs_abEELS_AlN_*_cycle0", bash will explode out the "*"
	#	psiCombiner()			# before passing to python! then we'll combine like-files from each
	#elif sys.argv[1]=="cycleCombiner":
	#	cycleCombiner()
	#elif sys.argv[1]=="fakePhononCycles":
	#	fakePhononCycles()