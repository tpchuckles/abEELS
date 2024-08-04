import sys,os,pyfftw,time,shutil,glob
#sys.path.insert(1,"copiedPythonUtils")
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../../MD') )	# in case someone "from abEELS import *" from somewhere else
from lammpsScrapers import *
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../niceplot') )
from niceplot import *
from nicecontour import *

import abtem,ase
import matplotlib.pyplot as plt

abtem.config.set({"precision": "float64"})

# (manual) unit test permutations: 
# semiAngle (0, !0), layerwise (0,!0), probePositions (len==0, len!=0), mode ("PZ","JACR")

# TWO METHODLOGIES:
# Paul Zeiger PRB 104, 104301 (2021): read in positions+displacements, preprocess (rotations/trimming/etc), perform band-pass filtering, run multislice on random configs ("frozen phonon" configurations) for each energy band, calculate Ψ(E,q,r) from coherent/incoherent parts across those multiple FP configs
# José Ángel Castellanos-Reyes https://arxiv.org/html/2401.15599v1: read in positions+displacements,  preprocess (rotations/trimming/etc), run multislice on every timestep, calculate Ψ(ω,q,r) via fourier transform

# SHARED FUNCTIONS: 
# readInputFile - setup
# importPositions - read in positions+displacements
# preprocessPositions - preprocess positions+displacements (rotations/trimming/etc)
# ewave = run multislice on [one or more configurations]
# UNIQUE FUNCTIONS:
# bandPassDisplacements - perform band-pass filtering
# energyLoss - calculate Ψ(E,q,r) from coherent/incoherent parts
# np.fft.fft - calculate Ψ(ω,q,r) from Ψ(t,q,r)
# TODO OLD FUNCTIONS NEED TO BE CLEANED UP, AND NONE OF THEM SHOULD BE HANDLING ANYTHING BUT IVIB FILES: nukeBadFP, psifileToIvib, preprocessOutputs, eDOS, layerDOS, processKmasks, maskDOS, kmaskCOG, cycleCleanup, cycleCombiner, psiCombiner, fakePhononCycles, cycle

# GENERAL FUNCTIONS

# OLD METHOD: Paul Zeiger et al: PRB 104, 104301 (2021)
# energy-binned displacements serve as FP configs. coherent/incoherent averaging yields energy-binned scattering
# R(t) --> F{ } --> R(ω) mask=gaussian(ω) ; Rₘₐₛₖ(ω)=R(ω)*gaussian(E,ω) --> F⁻¹{ } --> Rₙ(E)
# multislice for n configurations: Ψ(E,q,r,Rₙ)
# Ψⁱⁿᶜᵒʰᵉʳᵉⁿᵗ(E,q,r)=1/N ⟨ | Ψ(q,r,Rₙ(E)) |² ⟩ₙ ; Ψᶜᵒʰᵉʳᵉⁿᵗ = | 1/N ⟨ Ψ(q,r,Rₙ(E)) ⟩ₙ | ²
# Ψ(E,q,r) = Ψⁱⁿᶜᵒʰᵉʳᵉⁿᵗ(E,q,r) - Ψⁱⁿᶜᵒʰᵉʳᵉⁿᵗ(E,q,r)
def main_PZ():
	global energyCenters
	importPositions()
	#calculateLattice()
	preprocessPositions()
	energyCenters,masks=generateEnergyMasks(0,Emax,Ebins) # performs FFT on displacements, and generates gaussians we'll use for energy masking
	for m,(ec,mask) in enumerate(zip(energyCenters,masks)):
		#if m<=110:
		#	continue
		print("PROCESSING ENERGY BAND:",ec)
		# TODO ewave exits if the psi_eN*.npy file exists, but maybe we could check here too, and skip bandPassDisplacements and getFPconfigs?
		bdisp=bandPassDisplacements(mask)	# energy band pass on displacements = iFFT(FFT(displacements)*gaussianMask)
		atomStack=getFPconfigs(bdisp,plotOut=m)	# tiling and timestep selection for frozen phonon configurations
		#if parallelFP:	# may cause ram overload. e.g. 150Å x 150Å at 0.05Å spacing = 3000x3000 real-space sampling of waves, 64 bit complex
		ewave(atomStack,fileSuffix="e"+str(m),plotOut=(m==0)) # values. 1 FP config = 68 MB. multiply by the number of FP configs

	outfiles=psiFileNames(prefix="psi")
	for chunk in outfiles:
		psi=[ energyLoss(np.load(f)) for f in chunk ] # psi_eN*.npy files (nFP,kx,ky) (in)coherently summed --> kx,ky (for each energy)
		if not os.path.exists(outDirec+"/ws.npy"):
			np.save(outDirec+"/ws.npy",energyCenters)
		outname=chunk[0].replace("psi","ivib").replace("_e0","")
		np.save(outname,psi)
# SOME PERFORMANCE CONSIDERATIONS: each layer slice and probe position and FP config are stored in memory (one gargantuan 6D data cube?), so if you run into issues with ram, consider setting numFP=1, then manually stitching your psi*.npy files together (each will be 2D kx,ky in that case, so load them and re-save them: psi_FP=[ np.load(f) for f in glob.glob("*/**/psi_eN_lN_pN.npy",recursive=True) ] ; np.save("psi_lN_pN.npy",psi_FP)
# for extremely thick simulations, the potential itself is also enourmous. use thin slices, saveExitWave=True, and restartFrom=thisinputfilesname, run, rerun, rerun, etc. it will save the real-space exit wave, then use the previous real-space exit wave as the input-wave: we'd basically be doing multislice ourselves looped (add a layer via a separate run) instead of all at once in memory. this uses a lot of disk space, so if you need, all psi*.npy files can be deleted before the subsequent run (those were generated, but if this isn't the last layer we're interested in, they can be trashed), and all ewave*.npy files can be deleted *after* the subsequent run (those were only used to feed the subsequent run)

# NEW METHOD: José Ángel Castellanos-Reyes et al. https://arxiv.org/html/2401.15599v1
# electron wave is propagated for subsequent timesteps, and the wavefunction (as a function of time) if fouriered:
# P(t,a,xyz) --> Ψ(t,q,r) --> F --> Ψ(ω,q,r)
def main_JACR(): 
	importPositions()	# reads in globals: velocities,ts,types,avg,disp
	preprocessPositions()	# handles trimming/rotation, but NOT tiling
	aseString=[ atomTypes[t-1]+"1" for t in types ]
	aseString="".join(aseString) ; lx,ly,lz=a*nx,b*ny,c*nz
 
	t0=0 
	#t0=np.random.randint(len(ts)-maxTimestep) # TODO WHAT IS THE PROPER WAY TO STACK? for PZ method we select random (frequency-binned) timesteps for stacking in z. and tiling in x,y. here though, the temporal continuity maybe matters? 
	if restartFrom:
		offsets=np.linspace(0,len(ts)-maxTimestep,numCycles+2,endpoint=False,dtype=int)
		#np.random.seed(1) # ALWAYS USE THE SAME SEED SO EACH RUN STEPS THROUGH THE SAME RANDOMIZED LIST
		#np.random.shuffle(offsets)
		t0=offsets[cycleID]
		
	# run ewave concurrently for chunks of timesteps
	for t in np.arange(len(ts))[::concurrentTimesteps]:
		if t>=maxTimestep:
			break
		atomStack=[]
		for i in range(concurrentTimesteps):
			print("USING TIMESTEP",t+i+t0)
			atoms = ase.Atoms(aseString,positions=avg+disp[t+i+t0,:,:],pbc=[True,True,True],cell=[lx,ly,lz])
			atomStack.append(atoms)
		ewave(atomStack,fileSuffix="t"+str(t),plotOut=False)

	outfiles=psiFileNames(prefix="psi")
	for chunk in outfiles:
		psi_t=[] # stack up chunked timesteps into one big list of timesteps
		for f in chunk: # psi_t0.npy,psi_t10.npy,psi_t20.npy...
			p=np.load(f) # ts,kx,ky
			if concurrentTimesteps==1:	# if one timestep at a time, output files are 2D kx,ky instead of 3D t,kx,ky
				psi_t.append(p)
			else:
				for i in range(concurrentTimesteps):
					psi_t.append(p[i,:,:])
		# FFT along time axis: Ψ(t,q,r) --> Ψ(ω,q,r)		# NOT SURE I SAW IT IN THE PAPER, BUT NEED TO SUBTRACT 
		psi=np.fft.fft(psi_t-np.mean(psi_t,axis=0),axis=0) 	# OFF MEAN TO AVOID HIGH ZERO-FREQUENCY PEAK
		ws=np.fft.fftfreq(n=len(psi_t),d=ts[1]-ts[0])
		n=len(ws)//2 ; ws/=dt # convert to THz: e.g. .002 picosecond timesteps, every 10th timestep logged
		# kxs=np.load(outDirec+"/kxs.npy") ; kys=np.load(outDirec+"/kys.npy")
		# TODO NEED TO LOOP THROUGH LAYERS AND PROBE POSITIONS FOR SAVING
		if not os.path.exists(outDirec+"/ws.npy"):
			np.save(outDirec+"/ws.npy",ws)
		outname=chunk[0].replace("psi","ivib").replace("_t0","")
		np.save(outname,psi)

# SHARED FUNCTIONS: 
# for method=="PZ", we save off a "psi*.npy" file for each energy bin, which is a 3D matrix with FP,kx,ky indices. these are pre-summed, and are just the exit waves for each FP config. these are then "collapsed" into a single "ivib*.npy" file, which is 3D E,kx,ky indices. 
# for method=="JACR", we save off "psi_t*.npy" files which are time-chunks, 3D matrix t,kx,ky indices. these are then summed into a single "ivib*.npy" file which is the same as above, a 3D E,kx,ky indices. 
# layerwise!=0 will produce multiple of the above for each layer exported
# semiAngle!=0 and len(probePositions)>1 will produce multiple of the above for each probe position. 
# e.g. "psi_e9_[l8]_[p7].npy" ; "psi_t0_[l8]_[p7].npy" ; "ivib_[l8]_[p7].npy"
# this function returns a list of groupings of files which should all be processed together [l0:e0,e1,e2...,p1:e0,e1,e2... etc]
def psiFileNames(prefix="psi"):
	nt=1 ; ne=1 ; nl=1 ; npt=1
	if prefix=="psi": # look for "psi_eNNNN" or "psi_tNNNN"
		filename=prefix+"_"+{"JACR":"t","PZ":"e"}[mode]
		if mode=="PZ":
			ne=len(energyCenters)
		else:
			nt=len(ts)
	if layerwise!=0:
		nl=len(np.load(outDirec+"/layers.npy"))
	if semiAngle!=0 and len(probePositions)>1:
		npt=len(probePositions)
	filenames=[]
	for l in reversed(range(nl)): # loop through layers in reverse! final exit wave is probably more relevant to most people
		for p in range(npt):
			filenames.append([])
			for t in range(nt): # TODO NEED TO HANDLE SKIPPING BASED ON concurrentTimesteps
				if t%concurrentTimesteps!=0:
					continue
				if t>=maxTimestep:
					break
				for e in range(ne):
					psifile=outDirec+"/"+prefix+\
					{True:"_e"+str(e),False:""}[ne>1] +\
					{True:"_t"+str(t),False:""}[nt>1] +\
					{True:"_l"+str(l),False:""}[nl>1] +\
					{True:"_p"+str(p),False:""}[npt>1] + ".npy"
					filenames[-1].append(psifile)
	print(filenames)
	return filenames

def readInputFile(infile):
	# READ IN THE FOLLOWING VARIABLES FROM THE INPUT FILE
	# path,dumpfile,atomTypes,nx,ny,nz,a,b,c,dt,trim,beamAxis,flipBeam,tiling,semiAngle
	# Note no skewed primitive cells are currently supported for tiling
	global path, dumpfile, dumptype, useLattice, atomTypes, nx, ny, nz, a, b, c ,dt, trim, beamAxis, flipBeam, tile
	global semiAngle, outDirec, probePositions, saveExitWave, restartFrom, restartDirec, addModes, modifyProbe, numFP
	global kmask_xis,kmask_xfs,kmask_yis,kmask_yfs,kmask_cxs,kmask_cys,kmask_rad,kmask_lbl,deleteInputWave, cycleID
	global concurrentTimesteps, maxTimestep ; concurrentTimesteps=10 ; maxTimestep=1000 ; numFP=1 ; deleteInputWave=False
	kmask_xis,kmask_xfs,kmask_yis,kmask_yfs,kmask_cxs,kmask_cys,kmask_rad,kmask_lbl=[],[],[],[],[],[],[],[]
	dumptype="qdump" ; probePositions=[] ; addModes={} ; useLattice=False ; modifyProbe=False ; cycleID=0
	saveExitWave=False ; restartFrom=False
	lines=open(infile,'r').readlines()
	exec("".join(lines),globals())
	kmask_lbl=kmask_lbl+[ str(v) for v in range(len(kmask_xis)+len(kmask_cxs)-len(kmask_lbl)) ]
	print(kmask_lbl)
	outDirec="outputs/"+infile.strip("/").split("/")[-1].split(".")[0]
	# set restartFrom to be the input file's own name, and we'll run once (no picking up from an input wave), then subsequent runs will append "cycleN" to the output folder (cycle0 picks up from the original output direc, cycle1 picks up from cycle0 and so on).
	# if restartFrom is something *else* though, we won't append "cycleN" (useful if the cycling fails and you have an incomplete sub-run)
	if restartFrom:
		if (mode=="PZ" and numFP>1) or (mode=="JACR" and concurrentTimesteps>1):
			print("ERROR: layer cycling not allowed for numFP>1 or concurrentTimesteps>1")
			sys.exit()
		restartDirec="outputs/"+restartFrom ; RD=restartDirec
		if not os.path.exists(restartDirec):	# First run, restart direc might not exist (if so, turn off restartFrom)
			restartFrom=False ; cycleID=0
		else:
			if restartFrom!=infile.split("/")[-1].replace(".txt",""):
				restartDirec="outputs/"+restartFrom
			else:
				for i in range(numCycles):	# new output appends "cycleN"
					OD=outDirec+"_cycle"+str(i) ; cycleID+=1
					if not os.path.exists(OD):	# if output direc already exists, that becomes our new restart direc
						outDirec=OD
						break
					else:
						RD=restartDirec+"_cycle"+str(i)
				else:
					print("NUM CYCLES",numCycles,"REACHED. QUITTING")
					#sys.exit()
				restartDirec=RD			# only update restartDirec once we've found the right N
			print("RESTART SET: outDirec:",outDirec,"restartDirec",restartDirec)
	os.makedirs(outDirec,exist_ok=True)
	shutil.copy(infile,outDirec+"/")

def importPositions():
	global velocities,ts,types,avg,disp,positions

	# read in from custom dump file, containing positions *and* velocities
	if dumptype=="qdump":
		if os.path.exists(path+"avg.npy"): # best case scenario: average positions / displacements npy files already exist: read them
			print("reading existing npy files: avg,disp,typ,ts")
			avg=np.load(path+"avg.npy")
			disp=np.load(path+"disp.npy")
			types=np.load(path+dumpfile+"_typ.npy")
			ts=np.load(path+dumpfile+"_ts.npy")
			positions=avg+disp
			velocities=np.load(path+dumpfile+"_vel.npy")
		elif os.path.exists(path+dumpfile+"_pos.npy"): # next-best: positions npy file exists: read it, then run averaging.
			print("reading existing npy files: pos,typ,ts")
			positions=np.load(path+dumpfile+"_pos.npy") # note: qdump() will read these anyway, but we can skip that if they already exist
			velocities=np.load(path+dumpfile+"_vel.npy")
			types=np.load(path+dumpfile+"_typ.npy")
			ts=np.load(path+dumpfile+"_ts.npy")
			print("averaging positions")
			avg,disp=avgPos(positions,nx*a,ny*b,nz*c) # nt, na, 3
			np.save(path+"avg.npy",avg)
			np.save(path+"disp.npy",disp)
		else:					# last scenario: no npy files, so read qdump file
			print("reading dump file")
			positions,velocities,ts,types=qdump(path+dumpfile) # this will also save off pos/types/etc npy files
			print("averaging positions")
			avg,disp=avgPos(positions,nx*a,ny*b,nz*c) # nt, na, 3
			np.save(path+"avg.npy",avg)
			np.save(path+"disp.npy",disp)
		if useLattice:
			print("OVERWRITING AVERAGE POSITIONS FROM LATTICE FILE:",useLattice)
			avg,typ=scrapePos(path+useLattice)
	elif dumptype=="positions":
		print("reading positions from pos file")
		pos,types=scrapePos(path+dumpfile) # qdump positions,velocities are [nt,na,3]. avg is [na,3] and disp is [nt,na,3]
		positions=[pos,pos]
		velocities=np.zeros(np.shape(positions))
		disp=np.zeros(np.shape(positions)) ; avg=pos ; ts=np.asarray([0,1])

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
	print("spacing",spacing,"nx,ny,nz",nx,ny,nz,"a,b,c",a,b,c)
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

# TRIM, TILE, ROTATE, FLIP
#trim=[[],[],[]] # add two values (unit cell counts) to any dimension to trim to that size
#tile=[1,10,1]	# expanding the simulation in direction parallel to beam will lead to artifacts in reciprical space
#beamAxis=1	# abtem default is to look down z axis. we'll rotate the sim if you specify 0 or 1s
#flipBeam=False	# default is to look from +z towards -z, but we can mirror the simulation too (e.g. if we're more sensitivity to top layers)
# TODO TECHNICALLY OUR IMPLEMENTATION OF beamAxis AND flipBeam ARE WRONG. swapping axes is effectively mirroring across a 45° plane, and flibBeam mirrors across one of the three cartesian planes. mirroring is wrong though, think about chirality: you'll flip handedness
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

def matstrip(ary): # strip all len==1 indices out of an N-D array. shape 2,1,3,4,1,7 turns into shape 2,3,4,7. useful for getting rid of spurious axes
	shape=np.asarray(np.shape(ary))
	ones=np.where(shape==1)[0]
	for i in reversed(ones):
		ary=np.sum(ary,axis=i)
	return ary

# SIMULATE E WAVES THROUGH A LIST OF ASE CONFIGURATIONS
# 1. create the potential from the atomic configuration(s)
# 2. define the probe
# 3. denote where the probe is parked (create a GridScan object)
# 4. calculate the exit wave
# 5. transform exit wave to diffraction plane
# ( Paul Zeiger method for frequency-resolved frozen phonon multislice: run this multiple energy bins (atomic configurations came from a band-pass filter on MD configurations), then calculate coherent/incoherent components across each energy bin's FP configs )
# ( José Ángel Castellanos-Reyes method: run this for many consecutive timesteps, fourier time to ω: F{ Ψ(t,q,r) } --> Ψ(ω,q,r) )
# this function generates a "psi_[fileSuffix]_[lN]_[pN].npy" file, where l optionally denotes a layer number (if layerwise!=0) and p denotes a probe position (if a convergent probe is used, semiAngle!=0, and if multple probePositions are specified)
# ewave will NOT return the ewave (numerous may be created). we create the psi file, and calling code should go look for those files inside outDirec, which is "outputs/yourinputfilename/"
def ewave(atomStack,fileSuffix,plotOut=False):
	if "e0" in fileSuffix or "t0" in fileSuffix:
		plotOut=True
	print("ewave",fileSuffix)

	# CHECK IF THIS HAS ALREADY BEEN RUN (note layerwise!=0 len(probePositions)!=1 runs all layers and probe positions in parallel, then we save off individually, SO, we just have to look for l0 and/or p0)
	lookForFile=outDirec+"/psi_"+fileSuffix # suffix order is: e/t, l, p. e/t will have been passed by caller (fileSuffix)
	if layerwise!=0:
		lookForFile=lookForFile+"_l0"
	npts=1
	if semiAngle!=0 and len(probePositions)>1:
		npts=len(probePositions)
		lookForFile=lookForFile+"_p0"
	lookForFile=lookForFile+".npy" # "psi_[fileSuffix]_[lN]_[pN].npy"
	print("lookForFile",lookForFile)
	if os.path.exists(lookForFile):
		print("FIRST PSI-FILE FOUND FOR THIS CHUNK. SKIPPING",lookForFile)
		return

	# STEP 4, SIMULATE E WAVES
	# 1. create the potential from the atomic configuration(s)
	# 2. define the probe
	# 3. denote where the probe is parked (create the GridScan object)
	# 4. calculate the exit wave
	# 5. transform exit wave to diffraction plane

	# STEP 1.a CREATE ATOM ARRANGEMENT
	frozen_phonons=abtem.AtomsEnsemble(atomStack)
	# STEP 1.b CREATE POTENTIAL
	print("setting up wave sim")
	box=np.asarray(atomStack[0].cell).flat[::4] ; print("cell size: ",box)
	potential_phonon = abtem.Potential(frozen_phonons, sampling=.05) #,slice_thickness=.01) # default slice thickness is 1Å
	nLayers=1
	if layerwise!=0:
		nslices=potential_phonon.num_slices # 14.5 Å --> potential divided into 33 layers
		nth=nslices//layerwise # 33 layers, user asked for 10? 33//10 --> every 3rd layer --> 11 layers total. 
		nth=max(nth,1)		# user asked for 1000? 33//1000 --> 0! oops! just give them every layer instead. 
		potential_phonon = abtem.Potential(frozen_phonons, sampling=.05,exit_planes=nth)#,slice_thickness=.01) # default slice thickness is 1Å
		nLayers=potential_phonon.num_exit_planes
		layers=np.arange(nLayers)*nth ; np.save(outDirec+"/layers.npy",layers) # TODO not 100% sure these are truly the plane locations
		print("potential sliced into",nslices,"slices. keeping every",nth,"th layer, we'll have",nLayers,"exit waves")
		#with open(outDirec+"/nLayers.txt",'w') as f:
		#	f.write(str(nLayers))
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
	
	elif semiAngle!=0: # CONVERGENT PROBE INSTEAD OF PLANE WAVE (needs probe defined, and scan to denote where probe is positioned)
		probe = abtem.Probe(energy=100e3, semiangle_cutoff=semiAngle) # semiangle_cutoff is the convergence angle of the probe
		probe.grid.match(potential_phonon)
		print(probe.gpts)

		if plotOut:
			print("preview probe")
			probe.show() ; plt.savefig(outDirec+"/probe.png")

		# STEP 3, PROBE POSITION (user defined, or center of the sim volume)
		if len(probePositions)>0:
			custom_scan=abtem.CustomScan(probePositions) # TODO SHOULD PROBEPOSITIONS USE ABSOLUTE COORDINATES (Å), OR BE IN UNITS OF UNIT CELLS? 
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

	if plotOut:
		atoms=atomStack[0]
		abtem.show_atoms(atoms,plane="xy"); plt.title("beam view")
		if semiAngle!=0:
			if len(probePositions)>0:
				plt.plot(*np.asarray(probePositions).T)
			else:
				plt.plot(box[0]/2,box[1]/2)
		plt.savefig(outDirec+"/beam.png")
		if semiAngle!=0:
			pP=np.asarray(probePositions)
			if len(pP)==0:
				pP=np.asarray([[box[0]/2,box[1]/2]])
			plt.xlim([min(pP[:,0])-5,max(pP[:,0])+5])
			plt.ylim([min(pP[:,1])-5,max(pP[:,1])+5])
			plt.savefig(outDirec+"/beam2.png")
		abtem.show_atoms(atoms,plane="yz"); plt.title("side view") ; plt.savefig(outDirec+"/side.png")


	else: # PLANE WAVE INSTEAD OF CONVERGENT PROBE (just define the plane wave params)
		plane_wave = abtem.PlaneWave(energy=100e3, sampling=0.05)

	# STEP 4 CALCULATE EXIT WAVE
	print("calculating exit waves") ; start=time.time()
	if restartFrom:
		exit_waves = entrance_waves.multislice(potential_phonon).compute()
		print("np.shape(exit_waves.array)",np.shape(exit_waves.array))
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
	elif semiAngle!=0:
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

	# STEP 5 CONVERT TO DIFFRACTION SPACE
	print("converting exit wave to k space",end=" ")
	sys.stdout.flush() # https://stackoverflow.com/questions/35230959/printfoo-end-not-working-in-terminal
	start=time.time()
	diffraction_complex = exit_waves.diffraction_patterns(max_angle=maxrad,block_direct=False,return_complex=True)
	kx=diffraction_complex.sampling[-2]*diffraction_complex.shape[-2]/2 ; ky=diffraction_complex.sampling[-1]*diffraction_complex.shape[-1]/2
	kxs=np.linspace(-kx,kx,diffraction_complex.shape[-2]) ; kys=np.linspace(-ky,ky,diffraction_complex.shape[-1])
	if not os.path.exists(outDirec+"/kxs.npy"):
		np.save(outDirec+"/kxs.npy",kxs) ; np.save(outDirec+"/kys.npy",kys)
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

	if restartFrom and deleteInputWave:
		print("delete previous input wave file")
		os.remove(restartDirec+"/ewave_"+fileSuffix+".npy")
	#return zr # even if layerwise!=0, we'll end up with the last (thickest) psi. 

# END SHARED FUNCTIONS

# PZ-SPECIFIC FUNCTIONS: Paul Zeiger PRB 104, 104301 (2021), energy band pass filter displacements, run ewave for FP configs for each band, combine based on coherent/incoherent parts
# generateEnergyMasks - generate band-pass windows
# bandPassDisplacements - apply band-pass windows to displacements
# getFPconfigs -
# energyLoss - calculate Ψ(E,q,r) from coherent/incoherent parts

# performs FFT on displacements, and generates gaussians we'll use for energy masking
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

# 
def bandPassDisplacements(energyMask):
	print("calculate band pass on displacements")
	disp_bandpass=np.real(np.fft.ifft(dfft*energyMask[:,None,None],axis=0)) # ω,a,3 --> t,a,3 with band-pass filter applied to only select displacements associated with specific modes
	return disp_bandpass

def getFPconfigs(disp_bandpass,plotOut=False):

	os.makedirs(outDirec+"/configpics",exist_ok=True)
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
			if semiAngle!=0:
				if len(probePositions)>0:
					plt.plot(*np.asarray(probePositions).T)
				else:
					plt.plot(box[0]/2,box[1]/2)
			plt.savefig(outDirec+"/configpics/"+str(plotOut)+"_beam.png")
			if semiAngle!=0:
				pP=np.asarray(probePositions)
				if len(pP)==0:
					pP=np.asarray([[box[0]/2,box[1]/2]])
				plt.xlim([min(pP[:,0])-5,max(pP[:,0])+5])
				plt.ylim([min(pP[:,1])-5,max(pP[:,1])+5])
				plt.savefig(outDirec+"//configpics/"+str(plotOut)+"_beam2.png")
			abtem.show_atoms(atoms,plane="yz"); plt.title("side view") ; plt.savefig(outDirec+"/configpics/"+str(plotOut)+"_side.png")
	return atomStack

# given a 3D matrix (nFP,kx,ky), perform coherent/incoherent averaging
# COMPARING TO ZEIGER PRB 104, 104301 (2021) : They list:
# They use Ψ(q,r,R), where q is momentum, r is position of STEM probe, R is frozen-phonon configurations
#  Note the default exit-wave from abtem is real-space Ψ(R,rx,ry,x,y), so you can convert you can use return_complex arg for diffraction_patterns to get Ψ(R,rx,ry,ky,kx), and for sanity, we're going to use Ψ(R,kx,ky) ONLY (and if you have multiple probe positions rx,ry (coming from our customScan) or layers, pass them in separately).
def energyLoss(psi):
	nFP=len(psi) ; print("nFP",nFP)
	abpsisq=np.absolute(psi)**2 # incoherent = 1/N ⟨ | Ψ(q,r,Rₙ(ω)) |² ⟩ₙ (where "⟨ ⟩ₙ" means Σₙ)
	inco=1/nFP*np.sum(abpsisq,axis=0) # sum FP configs, keep kx axis
	sumpsi=1/nFP*np.sum(psi,axis=0) # coherent =  | 1/N ⟨ Ψ(q,r,Rₙ(ω)) ⟩ₙ | ²
	cohe=np.absolute(sumpsi)**2
	return inco-cohe

# END PZ-SPECIFIC FUNCTIONS

# POST PROCESSING / ANALYSIS FUNCTIONS: at this main_PZ and main_JACR should have created ivib*.npy files in the appropriate output directory. these are 3D (E,kx,ky) matrices and post-processing is just a matter of slicing up the 3D matrix in different ways. sum across kx,ky to get vDOS (states vs ω), take a slice across k to get the dispersion (ω vs k), take slices across ω to get an energy-resolved diffraction image (kx vs ky)

def eDOS():
	ws=np.load(outDirec+"/ws.npy")

	chunks=psiFileNames(prefix="ivib")
	print(chunks)
	Xs=[] ; Ys=[] ; lbls=[]
	for fs in chunks:
		if not os.path.exists(fs[0]): # e.g. we deleted early layers' files or something 
			continue
		psi=np.load(fs[0])
		Xs.append(ws) ; Ys.append(np.sum(np.absolute(psi),axis=(1,2)))
		lbls.append(fs[0])
	Ys=[ y[ws>1] for y in Ys ] ; Xs=[ x[x>1] for x in Xs ]
	#Ys=[ y/np.amax(y[ws>2]) for y in Ys ]
	plot(Xs,Ys,xlabel="frequency (THz)",ylabel="DOS (-)",labels=lbls,markers=rainbow(len(Xs)),filename=outDirec+"/eDOS.png",title="raw DOS")
	Ys=[ y/np.amax(y[Xs[0]>2]) for y in Ys ]
	plot(Xs,Ys,xlabel="frequency (THz)",ylabel="DOS (-)",labels=lbls,markers=rainbow(len(Xs)),filename=outDirec+"/eDOS_norm.png",title="peak normed")

def sliceE(nth=1):
	ws=np.load(outDirec+"/ws.npy")
	kxs=np.load(outDirec+"/kxs.npy")
	kys=np.load(outDirec+"/kys.npy")
	os.makedirs(outDirec+"/slices",exist_ok=True)

	chunks=psiFileNames(prefix="ivib")
	print(chunks)
	for fs in chunks:
		psi=np.load(fs[0])
		fo=fs[0].split("/") ; fo.insert(2,"slices") ; fo="/".join(fo) # outputs/inputfilename/ivib.npy --> outputs/inputfilename/slices/ivib.npy
		for i in tqdm(range(len(ws))):
			if i%nth!=0:
				continue
			contour(np.absolute(psi[i]).T,kxs,kys,filename=fo.replace(".npy","_e"+str(i)+".png"),xlabel="$\AA$^-1",ylabel="$\AA$^-1",title=str(ws[i])+" THz")#,zlim=zlim,cmap='Spectral')

#           | (1)	take a 2D slice out of the E,kx,ky data cube to get
#           o      (2)	ω vs k. arguments m, xi, yi define the slice via
# (3)       |    .'	point-slope form: y-yi=m*x-xi). xi,yi denote the
#  -----o---o---o-----	"origin" for k-space in the final dispersion.
#           | .'  	(1) m=np.inf xi=0 yi=0 would be the phonon dispersion
#   o   o   o'  o   o	    ω vs ky (Γ to X). beware selectivity rules: no
#         .'|   	    T modes will be shown!
#       o'  o   o	(2) m=1 xi=0 yi=0 would be Γ to M or something.
#     .'    |   	(3) m=0 xi=0 yi=4/a would be an offset horizontal slice,
#   .'      o   	    where selectivity rules will show T modes only in
#           |  		    the first BZ.
def dispersion(m=0,xi=0,yi=4/5.43729,xlim=[0,12/5.43729],ylim=[-np.inf,np.inf]):
	psi=np.load(outDirec+"/ivib.npy")
	ws=np.load(outDirec+"/ws.npy")
	kxs=np.load(outDirec+"/kxs.npy")
	kys=np.load(outDirec+"/kys.npy")

	ijs=[]
	if m<1:
		for i in range(len(kxs)):
			x=kxs[i] ; y=m*(x-xi)+yi
			if x<xlim[0] or x>xlim[1] or y<ylim[0] or y>ylim[1]:
				continue
			j=np.argmin(np.absolute(kys-y)) ; ijs.append([i,j])
	else:
		for j in range(len(kys)):
			y=kys[j] ; x=1/m*(y-yi)+xi
			if x<xlim[0] or x>xlim[1] or y<ylim[0] or y>ylim[1]:
				continue
			i=np.argmin(np.absolute(kxs-x)) ; ijs.append([i,j])
	
	overplot={"xs":[],"ys":[],"kind":"line","color":"r","linestyle":":"} 
	sliced=[] ; ks=[]
	for n,ij in enumerate(ijs):
		i,j=ij
		sliced.append(psi[:,i,j])
		distance=np.sqrt((kxs[i]-xi)**2+(kys[j]-yi)**2)
		sign=1
		if m>=0 and ( kxs[i]-xi < 0 or kys[j]-yi < 0 ):
			sign=-1
		ks.append(distance*sign)
		overplot["xs"].append(kxs[i]) ; overplot["ys"].append(kys[j])

	diff=np.load(outDirec+"/diff.npy")
	contour(np.log(diff).T,kxs,kys,filename=outDirec+"/disp_linescan.png",xlabel="$\AA$^-1",ylabel="$\AA$^-1",title="dispersion masked diffraction image",overplot=[overplot])
	contour(np.log(np.absolute(sliced)).T,ks,ws,xlabel="k ($\AA$^-1)",ylabel="frequency (THz)",title="dispersion",filename=outDirec+"/dispersion.png")

def diffraction():
	kxs=np.load(outDirec+"/kxs.npy")
	kys=np.load(outDirec+"/kys.npy")
	os.makedirs(outDirec+"/slices",exist_ok=True)

	# reading and grouping psi files (instead of ivib files) via psiFileNames assumes certain parameters depending on mode
	if mode=="PZ":
		global energyCenters ; energyCenters=np.load(outDirec+"/ws.npy")
	if mode=="JACR":
		global ts ; ts=np.arange(1e6)

	chunks=psiFileNames(prefix="psi")
	print(chunks)
	for fs in chunks:
		psis=[]
		for f in fs: # either psi_e or psi_t, either loop through energy, or time chunks
			psi=np.load(f) 
			if len(np.shape(psi))==2:
				psis.append(psi)
			else:
				for p in psi:	# for each energy or timechunk, loop through individual FP configs, or individual timesteps. 
					psis.append(p)
		fo=fs[0].replace("_t0","").replace("_e0","").replace(".npy","").replace("psi","diff")
		diff=np.sum(np.absolute(psis),axis=0)
		np.save(fo+".npy",diff)
		contour(np.log(diff).T,kxs,kys,filename=fo+".png",xlabel="$\AA$^-1",ylabel="$\AA$^-1",title="diffraction image")#,zlim=zlim,cmap='Spectral')
			#contour(np.log(np.absolute(sliced)),kxs,ws)

# python3 inputs/infile1.txt outputs/outdirec1 outputs/outdirec2 outputs/outdirec3...
# we will create a new output direc and stack up all like psi files
# def combinePZruns():
#	

# python3 inputs/infile1.txt
# we will create a new output direc and copy in and rename psi and ivib files (saveExitWave=True ; restartFrom="AlN_0m_01" ; numCycles=15 ; deleteInputWave=True) run over and over again will use one run's exit waves as the probe for subsequent runs, providing the ability to simulate super-thick volumes. we need to collect up the files from each cycle and rename them: e.g. "psi_e9_[p7].npy" ; "psi_t0_[p7].npy" ; "ivib_[p7].npy" --> "psi_e9_[l8]_[p7].npy" ; "psi_t0_[l8]_[p7].npy" ; "ivib_[l8]_[p7].npy"
def cyclesToLayers():

	importPositions()	# reads in globals: velocities,ts,types,avg,disp
	preprocessPositions()	# handles trimming/rotation, but NOT tiling

	# CREATE NEW OUTPUT DIRECTORY
	print(outDirec)
	newDirec=outDirec+"_combined"
	os.makedirs(newDirec,exist_ok=True)

	# GATHER UP FILES FROM INDIVIDUAL RUNS
	# reading and grouping psi files (instead of ivib files) via psiFileNames assumes certain parameters depending on mode
	if mode=="PZ":
		global energyCenters ; energyCenters=np.load(outDirec+"/ws.npy")
	if mode=="JACR":
		global ts ; ts=np.arange(1e6)

	outDirecs=[ outDirec ] + [ outDirec+"_cycle"+str(i) for i in range(numCycles) ]

	chunks=psiFileNames(prefix="psi")

	# COPY THEM IN (or use symlinks to avoid using so much space?)
	for l,OD in enumerate(outDirecs): # ["outputs/AlN_0m_01","outputs/AlN_0m_01_cycle0","outputs/AlN_0m_01_cycle1","outputs/AlN_0m_01_cycle2"...]
		# e.g. [[ "outputs/AlN_0m_01/psi_t0_p0.npy","outputs/AlN_0m_01/psi_t10_p0.npy","outputs/AlN_0m_01/psi_t20_p0.npy"...],[ "outputs/AlN_0m_01/psi_t0_p1.npy","outputs/AlN_0m_01/psi_t10_p1.npy","outputs/AlN_0m_01/psi_t20_p1.npy"...]...]
		for chunk in chunks: 
			for f in chunk: # e.g. "outputs/AlN_0m_01/psi_t98_p0.npy" or "outputs/AlN_0m_01/psi_t98.npy"
				pieces=f.replace(".npy","").split("_") # e.g. [..."01/psi","t98","p0"] or [..."01/psi","t98.npy"]
				if "t" in pieces[-1]: # no "p0"
					pieces.append("l"+str(l))
				else:
					pieces.insert(-1,"l"+str(l))
				fo="_".join(pieces)+".npy"
				f=f.replace(outDirec,OD)
				fo=fo.replace(outDirec,newDirec)
				#f=f.replace("
				print(f,"-->",fo)
				#shutil.copy(f,fo)
				if os.path.exists(fo):
					continue
				os.symlink("../../"+f,fo)

	# CREATE DUMMY INPUT FILE FOR COMBINED FOLDER (turn off restartFrom and numCycles, add layerwise, create new layers.npy file)
	lines=open(sys.argv[-1]).readlines()
	filtered=[]
	for l in lines:
		if "restartFrom" in l or "numCycles" in l:
			l="# "+l
		filtered.append(l)
	filtered.append("\nlayerwise=2\n")
	newInputFile=sys.argv[-1].replace(".txt","_combined.txt")
	with open(newInputFile,'w') as fo:
		for l in filtered:
			fo.write(l)

	layers=np.arange(numCycles+1)*nz*c
	np.save(newDirec+"/layers.npy",layers)
	
	for otherfile in ["ws.npy","kxs.npy","kys.npy"]:
		shutil.copy(outDirec+"/"+otherfile,newDirec)
	
	print("PLEASE RUN \"python3 abEELS.py "+newInputFile+"\" before attempting any post-processing (this creates required ivib files)")
	#print("YOU MAY NOW RUN POST-PROCESSING ON COMBINED RUN: e.g. \"python3 DOS inputs/"+sys.argv[-1].replace(".txt","_combined.txt")+"\"")

funcAliases={"DOS":"eDOS"}
# if called directly, run main(). does not run if imported "from abEELS import *"
if __name__=='__main__':

	infile=sys.argv[-1]
	readInputFile(infile)

	if len(sys.argv)==2:
		if mode=="JACR":
			main_JACR()
		else:
			main_PZ()
	else:
		fun=sys.argv[1] 
		if fun in funcAliases:
			fun = funcAliases[ fun ]
		command=fun+"()"
		exec(command)