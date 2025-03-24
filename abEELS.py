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
 
	#global t0
	#t0=np.random.randint(len(ts)-maxTimestep) # TODO WHAT IS THE PROPER WAY TO STACK? for PZ method we select random (frequency-binned) timesteps for stacking in z. and tiling in x,y. here though, the temporal continuity maybe matters? 
	#if restartFrom:
	#	offsets=np.linspace(0,len(ts)-maxTimestep,numCycles+2,endpoint=False,dtype=int)
	#	#np.random.seed(1) # ALWAYS USE THE SAME SEED SO EACH RUN STEPS THROUGH THE SAME RANDOMIZED LIST
	#	#np.random.shuffle(offsets)
	#	t0=offsets[cycleID]
		
	# run ewave concurrently for chunks of timesteps
	for t in np.arange(len(ts))[::concurrentTimesteps]:
		# don't bother constructing the atoms objects if we're just going to skip it inside ewave! moved lookForFile code to its own function to call here
		if os.path.exists(psiFileName("t"+str(t))): 
			print("skipping timestep",t)
			continue
		if t>=maxTimestep:
			break

		atomStack=[]
		for i in range(concurrentTimesteps):
			print("USING TIMESTEP",t+i+t0)
			if max(tiling)<=1:
				atoms = ase.Atoms(aseString,positions=avg+disp[t+i+t0,:,:],pbc=[True,True,True],cell=[lx,ly,lz])
			else:
				av=tile(avg) ; dis=tile(disp[t+i+t0,:,:]) ; cell=[lx*tiling[0],ly*tiling[1],lz*tiling[2]]
				aseStr="".join( [aseString]*tiling[0]*tiling[1]*tiling[2] )
				atoms = ase.Atoms(aseStr,positions=av+dis,pbc=[True,True,True],cell=cell)
			atomStack.append(atoms)
			if i==0 and t==0:
				# outToPositionsFile() expects: filename,pos,types,masses,a,b,c
				outToPositionsFile(outDirec+"/configuration0.pos",avg+disp[t+i+t0,:,:],types,list(range(1,len(set(types))+1)),lx,ly,lz)
		ewave(atomStack,fileSuffix="t"+str(t),plotOut=False)

	outfiles=psiFileNames(prefix="psi")
	for chunk in outfiles:
		# psi_t=[] # stack up chunked timesteps into one big list of timesteps 
		# problem: loading as a list uses more ram. we should create a 3D matrix first. easy to drop-in replace your appends with: "psi_t[ct]=... ; ct+=1"
		psi_t=np.zeros( ( maxTimestep,len(np.load(outDirec+"/kxs.npy")),len(np.load(outDirec+"/kys.npy")) ) ,dtype=complex) ; ct=0
		for f in tqdm(chunk): # psi_t0.npy,psi_t10.npy,psi_t20.npy...
			p=np.load(f) # ts,kx,ky
			if concurrentTimesteps==1:	# if one timestep at a time, output files are 2D kx,ky instead of 3D t,kx,ky
				#psi_t.append(p)
				psi_t[ct,:,:]=p ; ct+=1
			else:
				for i in range(concurrentTimesteps):
					#psi_t.append(p[i,:,:])
					psi_t[ct,:,:]=p[i,:,:] ; ct+=1
		# FFT along time axis: Ψ(t,q,r) --> Ψ(ω,q,r)		# NOT SURE I SAW IT IN THE PAPER, BUT NEED TO SUBTRACT 
		#psi=np.fft.fft(psi_t-np.mean(psi_t,axis=0),axis=0) 	#   OFF MEAN TO AVOID HIGH ZERO-FREQUENCY PEAK
		# problem: this means you need 2x as much ram. loop through each kx,ky (one column as a time) or through kx (one "sheet" at a time) and edit in-place
		for i in tqdm(range(np.shape(psi_t)[1])):
			psi_t[:,i,:]=np.fft.fft(psi_t[:,i,:]-np.mean(psi_t[:,i,:],axis=0),axis=0)
		psi=psi_t
		ws=np.fft.fftfreq(n=len(psi_t),d=ts[1]-ts[0])
		n=len(ws)//2 ; ws/=dt # convert to THz: e.g. .002 picosecond timesteps, every 10th timestep logged
		# kxs=np.load(outDirec+"/kxs.npy") ; kys=np.load(outDirec+"/kys.npy")
		# TODO NEED TO LOOP THROUGH LAYERS AND PROBE POSITIONS FOR SAVING
		if not os.path.exists(outDirec+"/ws.npy"):
			np.save(outDirec+"/ws.npy",ws)
		outname=chunk[0].replace("psi","ivib").replace("_t0","")
		if not os.path.exists(outname):
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
	global path, dumpfile, dumptype, useLattice, atomTypes, nx, ny, nz, a, b, c ,dt, trim, beamAxis, flipBeam, tile, maxFreq, nthTimestep
	global semiAngle, outDirec, probePositions, saveExitWave, restartFrom, restartDirec, addModes, modifyProbe, modifyAtoms, numFP , slice_thickness
	global kmask_xis,kmask_xfs,kmask_yis,kmask_yfs,kmask_cxs,kmask_cys,kmask_rad,kmask_lbl,deleteInputWave, cycleID , calculateForces
	global concurrentTimesteps, maxTimestep , orthogonalize , t0 , cropExitWave , shift , rotate, sampling
	concurrentTimesteps=10 ; maxTimestep=1000 ; numFP=1 ; deleteInputWave=False ; slice_thickness=1 ; shift=[0,0,0] ; rotate=False ; sampling=.05
	kmask_xis,kmask_xfs,kmask_yis,kmask_yfs,kmask_cxs,kmask_cys,kmask_rad,kmask_lbl=[],[],[],[],[],[],[],[]
	dumptype="qdump" ; probePositions=[] ; addModes={} ; useLattice=False ; modifyProbe=False ; modifyAtoms=False ; cycleID=0
	calculateForces=False ; orthogonalize=False ; saveExitWave=False ; restartFrom=False ; maxFreq=None ; t0=0 ; nthTimestep=1
	cropExitWave=False ; 
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
		restartDirec="outputs/"+restartFrom #; RD=restartDirec
		#if not os.path.exists(restartDirec):	# First run, restart direc might not exist (if so, turn off restartFrom)
		#	restartFrom=False ; cycleID=0
		#else:
		#	#if restartFrom!=infile.split("/")[-1].replace(".txt",""):
		#	restartDirec="outputs/"+restartFrom
		#	#else: # TODO numCycles CODE NEEDS TO BE REVAMPED. AN INCOMPLETE CYCLE SHOULD NOT JUMP AHEAD TO THE NEXT CYCLE JUST BECAUSE THE CURRENT CYCLE'S FOLDER EXISTS. SEE outputs/cycleGenerator.py. IT GENERATES A BUNCH OF INPUT FILES, AND A STRING OF PYTHON COMMANDS TO RUN, AND RERUNNING ANY OF THEM IS FINE BECAUSE THE PSI FILES ARE THERE, AND IT DOESN'T SKIP INAPPROPRIATELY. 
		#	#	for i in range(numCycles):	# new output appends "cycleN"
		#	#		OD=outDirec+"_cycle"+str(i) ; cycleID+=1
		#	#		if not os.path.exists(OD):	# if output direc already exists, that becomes our new restart direc
		#	#			outDirec=OD
		#	#			break
		#	#		else:
		#	#			RD=restartDirec+"_cycle"+str(i)
		#	#	else:
		#	#		print("NUM CYCLES",numCycles,"REACHED. QUITTING")
		#	#		#sys.exit()
		#	#	restartDirec=RD			# only update restartDirec once we've found the right N
		#	print("RESTART SET: outDirec:",outDirec,"restartDirec",restartDirec)
	os.makedirs(outDirec,exist_ok=True)

	if modifyProbe and len(probePositions)>1:
		print("UH OH! modifyProbe appears to be incompatible with probePositions! try setting up multiple runs with a probe position offset applied inside of modifyProbe! see run log \"abtemProbeVsModifyProbeAndPositions\" for evidence and more details")
		sys.exit()

	# e.g. your lammps timestep size was 0.0005 ps, dumped every 10th step (dt in your input file should be 0.0005*10) yielding a max frequency (1/dt/2) of 100THz. too high, waste of compute. use every-other-timestep, new dt will be 0.0005*10*2 (max frequency of 50 THz). double dt here, and importPositions will select every-other-timestep
	if nthTimestep!=1: 
		dt*=nthTimestep
	shutil.copy(infile,outDirec+"/")

def importPositions():
	# twp 2024-09-03 removing global "positions" since we shouldn't use it! use avg+disp instead! and velocities would only be used to calculate vDOS (which is w*dDOS, since u(t,x)=exp(iwt-ikx))
	global ts,types,avg,disp,dt #velocities,positions
	# read in from custom dump file, containing positions *and* velocities
	if dumptype=="qdump":
		if os.path.exists(path+"avg.npy"): # best case scenario: average positions / displacements npy files already exist: read them
			print("reading existing npy files: avg,disp,typ,ts")
			avg=np.load(path+"avg.npy")
			disp=np.load(path+"disp.npy")
			types=np.load(path+dumpfile+"_typ.npy")
			ts=np.load(path+dumpfile+"_ts.npy")
		elif os.path.exists(path+dumpfile+"_pos.npy"): # next-best: positions npy file exists: read it, then run averaging.
			print("reading existing npy files: pos,typ,ts")
			positions=np.load(path+dumpfile+"_pos.npy") # note: qdump() will read these anyway, but we can skip that if they already exist
			types=np.load(path+dumpfile+"_typ.npy")
			ts=np.load(path+dumpfile+"_ts.npy")
			print("averaging positions")
			avg,disp=avgPos(positions,nx*a,ny*b,nz*c) # nt, na, 3 # TODO I DON'T KNOW THAT avgPos CORRECTLY HANDLES NON-ORTHOGONALIZED CELLS
			np.save(path+"avg.npy",avg)
			np.save(path+"disp.npy",disp)
		else:					# last scenario: no npy files, so read qdump file
			print("reading dump file")
			positions,velocities,ts,types=qdump(path+dumpfile)#,convert=False) # this will also save off pos/types/etc npy files
			velocities=0 # don't waste RAM storing velocities
			print("averaging positions")
			avg,disp=avgPos(positions,nx*a,ny*b,nz*c) # nt, na, 3
			np.save(path+"avg.npy",avg)
			np.save(path+"disp.npy",disp)
		if useLattice:
			print("OVERWRITING AVERAGE POSITIONS FROM LATTICE FILE:",useLattice)
			avg,typ=scrapePos(path+useLattice)
			disp*=0
	elif dumptype=="positions":
		print("reading positions from pos file")
		pos,types=scrapePos(path+dumpfile) # qdump positions,velocities are [nt,na,3]. avg is [na,3] and disp is [nt,na,3]
		disp=np.zeros((maxTimestep,len(pos),3)) ; avg=pos ; ts=np.arange(maxTimestep)
	elif dumptype=="npydict":
		avg=np.load(path+dumpfile["avg"]) ; disp=np.load(path+dumpfile["disp"])
		types=np.load(path+dumpfile["types"]) ; ts=np.arange(len(disp))
		print(avg.shape,disp.shape,types.shape,ts.shape)
	print("found",len(ts),"timesteps")
	# TODO I THINK THIS CONFLICTS WITH HAIRCUT CHECKER: what if you have a non-orthogonal cell with danglers? chop x and slide right, then chop y and slide up (but you just slid danglers to the wrong place!)
	if orthogonalize: 
		for ijk,n,l in zip([0,1,2],[nx,ny,nz],[a,b,c]):
			mask=np.zeros(len(avg)) ; mask[avg[:,ijk]<0]=1
			avg[mask==1,ijk]+=n*l #; disp[:,xmask==1,ijk]+=n*l
			mask=np.zeros(len(avg)) ; mask[avg[:,ijk]>=n*l]=1
			avg[mask==1,ijk]-=n*l #; disp[:,xmask==1,0]+=nx*a

	if max(shift)>0:
		for i,dx in enumerate(shift):
			if dx>0:
				avg[:,i]+=dx

	# WRAPPING: imagine a plane of atoms initialized at z=0. these atoms jiggle through the PBC, and there's a 50% chance their average is at the top vs bottom of the simulation. then when a wave hits "dangling" top atoms, we'll find asymmetry where there is none, and other strange e-wave interactions
	# Consider the following. the 4th row of atoms has atom 0 "wrapped" around to +lx instead of being at zero where it "should be". 
	# |o     o     o     o     |	# this results in a "fuzzy" system with dangling atoms, and when thesde are at the top, we'll detect
	# | o    o    o       o    |	# asymmetry where there is none, and other strange e-wave interactions. 
	# |o    o      o     o     |	# to fix this, we infer the spacing of the atoms (collect x positions, sort them, calculate distance
	# |      o     o     o    o|	# between them, and look for the smallest distance which is not just a vibrational distance)
	# |o    o      o    o      |	# any atoms within l/2 of the right edge can then be tossed back around to the left. 
	# infer spacing 
	spacing=[]
	for i in range(3):
		xyzs=list(sorted(avg[:,i]))	# all positions in x
		dxyz=np.gradient(xyzs)		# spacing between atoms in x
		dxyz=dxyz[dxyz>.1]		# assume jiggling is less than .1, assume no interatomic spacing is less than .1
		if len(dxyz)==0:		# e.g. monolayer of atoms! 
			spacing.append(1)
		else:
			spacing.append(min(dxyz))
			print("SPACING",i,min(dxyz),max(dxyz))
	print("INFERRED INTERATOMIC SPACING",spacing,"(used for haircut)")
	#spacing=[ s/2 for s in spacing ]
	print("spacing",spacing,"nx,ny,nz",nx,ny,nz,"a,b,c",a,b,c)
	for i in range(3):
		mask=np.zeros(len(avg))
		l=[nx*a,ny*b,nz*c][i]
		mask[avg[:,i]>=l-spacing[i]/2]=1
		print("haircut applied to",len(mask[mask==1]),"atoms")
		avg[mask==1,i]-=l
	# ALSO SHIFT IN BEAM DIRECTION BY HALF A SPACING, TO ENSURE UNWRAPPED STUFF DOESN'T GET REWRAPPED BY ASE (dangling top/bottom atoms)
	avg[:,beamAxis]+=spacing[beamAxis]/2
	# outToPositionsFile() expects: filename,pos,types,masses,a,b,c
	outToPositionsFile(outDirec+"/haircutchecker.pos",avg,types,list(range(1,len(set(types))+1)),nx*a,ny*b,nz*c)

	# e.g. you ran your MD simulation with timesteps of 0.0005 ps dumping every 10, which is a max frequency of 1/(.0005*10)/2=100 THz and you don't care about frequencies above 50. if you process every timestep, you're wasting time (and need to process way more timesteps to get the same frequency resolution, e.g. 400 timesteps to get 0.5 THz resolution (-100 to 100 THz / 400 = 0.5 THz). Instead, use every-other timestep. easiest way to handle that is just select every-nth velocity/displacement and double the passed dt. now you have *effectively* run with 0.0005 ps timesteps dumping every 20, max frequency 1/(.0005*20)/2=50THz max, only need to process 200 (corrected) timesteps to get 0.5 THz resolution (-50 to 50 THz / 200 = 0.5 THz)
	if nthTimestep!=1:
		print("ts",ts)
		disp=disp[::nthTimestep] # ts=ts[::nthTimestep] # ts is just timestep indices 1,2,3,..., so DON'T grab every-other
		#dt*=nthTimestep # WE'RE ACTUALLY GOING TO DO THIS IN readInputFile(), SO OTHER FUNCTIONS LIKE sliceE WILL HEAR ABOUT THE NEW dt

	# all of the following contour matrix tomfoolery allows use of the same addWave function to visualize as we use for adding displacements! this minimizes the chances of having a bug somewhere that only affects viewing or displacing, meaning what we see if not what we end up with). 
	if "A" in addModes.keys() and "contourIndices" in addModes.keys():
		contourXs=np.linspace(min(avg[:,0]),max(avg[:,0]),100)		# x positions used for contour plot showing waves and atoms
		contourYs=np.linspace(min(avg[:,1]),max(avg[:,1]),101) 
		contourZs=np.zeros((maxTimestep,101*100,3))			# emulates the disp matrix (nt,na,xyz); that's what addWave expects
		contourXY=np.zeros((100*101,3))					# emulates the avg matrix (na,xyz); that's what addWave expects
		contourXY[:,0]=(contourXs[None,:]*np.ones(101)[:,None]).flat	# load the x positions (1D list) into the faked avg matrix
		contourXY[:,1]=(contourYs[:,None]*np.ones(100)[None,:]).flat

	if "A" in addModes.keys(): # addModes={"THz":[...],"iA":[...],"pdirec":[...],"vdirec":[...],"phase":[...]}
		pltX=np.linspace(0,20,100) ; pltY=[] ; pltlbl=[]
		xi=addModes.get("xi",0) ; yi=addModes.get("yi",0)
		for i in range(len(addModes["A"])):
			A,k,w,p,v,phi=[ addModes[key][i] for key in ["A","k","w","pdirec","vdirec","phase"] ]
			print("add wave A:",A,"k",k,"w",w,"x",p,"phi",phi)
			addWave(A,k,w,phi,p,v,disp,avg,xi=xi,yi=yi)
			pltY.append(A*np.sin(k*pltX*2*np.pi+phi))
			pltlbl.append("A: "+str(np.round(A,5))+", k: "+str(np.round(k,5))+", phi: "+str(np.round(phi,5)))
			#avg[:,:]+=added[:,:]
			#disp[:
			if "contourIndices" in addModes.keys() and i in addModes["contourIndices"]:
				# faked disp/avg matrices means we can call addWave to load displacements into contour Zs. 
				addWave(A,k,w,phi,p,0,contourZs,contourXY,xi=xi,yi=yi)	# set v=0 so displacments all end up in 0th xyz index (of nt,na,xyz)

		pltX=[pltX]*len(pltY) ; mkrs=['-']*len(pltY)
		#if "atomSpacing" in addModes.keys():
		#	pltX.append(np.arange(20//addModes["atomSpacing"])*addModes["atomSpacing"])
		#	pltY.append(pltX[-1]*0) ; mkrs.append(".") ; pltlbl.append("")
		plot(pltX,pltY,markers=mkrs,labels=pltlbl,filename=outDirec+"/addModes.png")
		
		os.makedirs(outDirec+"/addWaveContours",exist_ok=True)
		contourZs=contourZs[:,:,0].reshape((maxTimestep,101,100))			# unform the emulated disp matrix (nt,na,xyz) into a 2D grid ny,nx
		overplot=[{"xs":avg[:,0],"ys":avg[:,1],"kind":"scatter","marker":"."},
				{"xs":[xi],"ys":[yi],"kind":"scatter","c":"w","marker":"."}]	# add atoms to the contour as points
		for i in range(maxTimestep):
			contour(contourZs[i],contourXs,contourYs,overplot=overplot,aspect=1,figsize=(20,20),xlabel="x ($\AA$)",ylabel="x ($\AA$)",title="displacements for wave indices: "+",".join([ str(v) for v in addModes["contourIndices"]]),filename=outDirec+"/addWaveContours/previewModes-"+str(i)+".png",xlim=[xi-30,xi+30],ylim=[yi-30,yi+30])
			#contour(contourZs[i],contourXs,contourYs,overplot=overplot,aspect=1,figsize=(20,20),xlabel="x ($\AA$)",ylabel="x ($\AA$)",title="displacements for wave indices: "+",".join([ str(v) for v in addModes["contourIndices"]]),filename=outDirec+"/previewModes2.png",xlim=[xi-30,xi+30],ylim=[yi-30,yi+30])

def addWave(A,k,w,phi,p_xyz,v_xyz,disp,avg,xi=0,yi=0):
	na=len(avg)
	nt={"PZ":len(disp),"JACR":maxTimestep}[mode] # add to ALL timesteps for PZ method, since snapshots are taken randomly. only add to first N for JACR method (no point adding to the timesteps we don't even use. just wastes computation time)
	# VECTOR MATH RIPPED STRAIGHT OUTA lammpsScapers.py > SED()
	dxy=np.asarray([-xi,-yi,0])
	if isinstance(p_xyz,(int,float)): # 0,1,2 --> x,y,z
		xs=(avg+dxy[None,:])[:,p_xyz] # a,xyz --> a
	else:	# [1,0,0],[1,1,0],[1,1,1] and so on
		# https://math.stackexchange.com/questions/1679701/components-of-velocity-in-the-direction-of-a-vector-i-3j2k
		# project a vector A [i,j,k] on vector B [I,J,K], simply do: A•B/|B| (dot, mag)
		# for at in range(na): x=np.dot(avg[at,:],p_xyz)/np.linalg.norm(p_xyz)
		# OR, use np.einsum. dots=np.einsum('ij, ij->i',listOfVecsA,listOfVecsB)
		p_xyz=np.asarray(p_xyz)
		d=p_xyz[None,:]*np.ones((na,3)) # xyz --> a,xyz
		xs=np.einsum('ij, ij->i',avg[:,:]+dxy[None,:],d) # pos • vec, all at once
		xs/=np.linalg.norm(p_xyz)
	if isinstance(v_xyz,(int,float)):
		vs=np.zeros(3)
		vs[v_xyz]=1
	else:
		vs=v_xyz/np.sqrt(np.sum(np.asarray(v_xyz)**2))
	for ijk in range(3):
		disp[:nt,:,ijk]+=vs[ijk]*A*np.sin(k*xs[None,:]*2*np.pi+phi+w*ts[:nt,None]*dt*2*np.pi)

# TRIM, TILE, ROTATE, FLIP
#trim=[[],[],[]] # add two values (unit cell counts) to any dimension to trim to that size
#tile=[1,10,1]	# expanding the simulation in direction parallel to beam will lead to artifacts in reciprical space
#beamAxis=1	# abtem default is to look down z axis. we'll rotate the sim if you specify 0 or 1s
#flipBeam=False	# default is to look from +z towards -z, but we can mirror the simulation too (e.g. if we're more sensitivity to top layers)
# TODO TECHNICALLY OUR IMPLEMENTATION OF beamAxis AND flipBeam ARE WRONG. swapping axes is effectively mirroring across a 45° plane, and flibBeam mirrors across one of the three cartesian planes. mirroring is wrong though, think about chirality: you'll flip handedness
def preprocessPositions():
	global ts,types,avg,disp,a,b,c,nx,ny,nz,tiling,shift,probePositions
	#print("preprocessPositions, shapes: ts,types,avg,disp",np.shape(ts),np.shape(types),np.shape(avg),np.shape(disp))
	# TRIM A TOO-BIG VOLUME
	for ijk,lims in enumerate(trim):
		print("preprocessPositions: trim:",ijk,lims)
		if len(lims)!=2:
			continue
		if lims[0]<0:
			lims[0]=0
		n=[nx,ny,nz][ijk]
		if lims[1]>n:		# is user is a doofus (or mistypes) a tiling greater than the simulation size, just ignore it.
			lims[1]=n
		abc=[a,b,c][ijk]
		xs=avg[:,ijk] ; xi=lims[0]*abc ; xf=lims[1]*abc
		print("TRIM:",ijk,xi,xf)
		mask=np.zeros(len(xs))
		mask[xs>=xi]=1 ; mask[xs>=xf]=0
		#mask[xs>=xi]=1 ; mask[xs>=xi+2]=0 ; mask[xs>=xf-2]=1 ; mask[xs>=xf]=0 # CUSTOM MASK MOD TO CLEAR OUT ATOMS SO WE CAN VISUALIZE THE PROBE
		avg=avg[mask==1,:] ; disp=disp[:,mask==1,:] ; types=types[mask==1]
		avg[:,ijk]-=xi
		if ijk==0:
			nx=lims[1]-lims[0]
		if ijk==1:
			ny=lims[1]-lims[0]
		if ijk==2:
			nz=lims[1]-lims[0]
		for i in range(len(probePositions)):
			if ijk==2:
				continue
			probePositions[i][ijk]-=lims[0]*abc
		print("preprocessPositions: updated vals:","nx",nx,"ny",ny,"nz",nz,"probePositions",probePositions)
	if rotate:
		theta,axis=rotate								# [ c -s ] 
		R=np.asarray([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]) 	# [ s  c ] [ c  0 -s ]
		R=np.insert(R,axis,[0,0],axis=1)						#          [ s  0  c ] [ c  0 -s ]
		row=[0,0,0] ; row[axis]=1							#                      [ 0  1  0 ]
		R=np.insert(R,axis,row,axis=0)							#                      [ s  0  c ]
		dxyz=np.asarray([nx*a/2,ny*b/2,nz*c/2])
		avg=np.matmul(avg-dxyz,R)+dxyz
		disp=np.matmul(disp,R)

	# ROTATE SO BEAM AXIS IS Z
	if beamAxis!=2: # default is to look down z, so we need to flip axes if we want to look down a different direction
		rollby=[-1,1][beamAxis] # to look down x, roll -1 [x,y,z] --> [y,z,x]. to look down y, roll +1 [x,y,z] --> [z,x,y]
		print("preprocessPositions: adjust beam axis:",beamAxis,"rollby",rollby)
		avg=np.roll(avg,rollby,axis=1) # na,[x,y,z] --> na,[y,z,x]
		disp=np.roll(disp,rollby,axis=2) # nt,na,[x,y,z] --> nt,na,[y,z,x]
		a,b,c=np.roll([a,b,c],rollby)
		nx,ny,nz=np.roll([nx,ny,nz],rollby)
		tiling=np.roll(tiling,rollby)
		shift=np.roll(shift,rollby)
		# TODO probePositions might need to be axis-swapped depending on rolling? 
		print("preprocessPositions: updated vals:","a,b,c",a,b,c,"nx,ny,nz",nx,ny,nz,"tiling",tiling,"shift",shift)
	if flipBeam:
		print("preprocessPositions: flip beam: inverting avg and disp")
		avg[:,2]*=-1 ; avg[:,2]+=nz*c # # na,[x,y,z], flip z, shift position
		disp[:,:,2]*=-1 # # nt,na,[x,y,z], no shift for displacements or velo
	if modifyAtoms:
		modifyAtoms()
		# outToPositionsFile() expects: filename,pos,types,masses,a,b,c
		outToPositionsFile(outDirec+"/haircutchecker-postmod.pos",avg,types,list(range(1,len(set(types))+1)),nx*a,ny*b,nz*c)

	if len(probePositions)>0 and max(shift)>0:
		print("preprocessPositions: apply shift to probe positions")
		for i in range(len(probePositions)):
			probePositions[i][0]+=shift[0]
			probePositions[i][1]+=shift[1]
		print("preprocessPositions: updated vals:","probePositions",probePositions)
# replaced with np.squeeze
#def matstrip(ary): # strip all len==1 indices out of an N-D array. shape 2,1,3,4,1,7 turns into shape 2,3,4,7. useful for getting rid of spurious axes
#	shape=np.asarray(np.shape(ary))
#	ones=np.where(shape==1)[0]
#	for i in reversed(ones):
#		ary=np.sum(ary,axis=i)
#	return ary

def psiFileName(fileSuffix):
	lookForFile=outDirec+"/psi_"+fileSuffix # suffix order is: e/t, l, p. e/t will have been passed by caller (fileSuffix)
	if layerwise!=0:
		lookForFile=lookForFile+"_l0"
	npts=1
	if semiAngle!=0 and len(probePositions)>1:
		npts=len(probePositions)
		lookForFile=lookForFile+"_p0"
	lookForFile=lookForFile+".npy" # "psi_[fileSuffix]_[lN]_[pN].npy"
	return lookForFile

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
	global probePositions

	if "e0" in fileSuffix or "t0" in fileSuffix:
		plotOut=True
	print("ewave",fileSuffix)

	# CHECK IF THIS HAS ALREADY BEEN RUN (note layerwise!=0 len(probePositions)!=1 runs all layers and probe positions in parallel, then we save off individually, SO, we just have to look for l0 and/or p0)
	lookForFile=psiFileName(fileSuffix)
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
	potential_phonon = abtem.Potential(frozen_phonons, sampling=sampling,slice_thickness=slice_thickness) # default slice thickness is 1Å
	xs=np.linspace(0,potential_phonon._grid.extent[0],potential_phonon._grid.gpts[0],endpoint=False)
	ys=np.linspace(0,potential_phonon._grid.extent[1],potential_phonon._grid.gpts[1],endpoint=False)

	# POTENTIAL RESAMPLING BASED ON MAX K DESIRED. 
	# Could we propagate in reciprocal space (so we can "crop" k, keeping same k resolution, and be processing a much smaller voxel map?
	# maybe, but what about FFT, trim, iFFT, then propagate in real-space, without needing a different propagator (Fresnel is real-space)
	# maybe that too, BUT, this is the same as just specifying the real-space sampling to 1/maxk
	#potential_array=np.asarray( potential_phonon.to_images().array )
	#rpotential,ky,kx=fft2(potential_array,ys,xs,maxk=maxrad/100*2)
	#rrpotential,y2,x2=fft2(rpotential,ky,kx,inverse=True)
	#sampling=x2[1]-x2[0]
	#potential_phonon = abtem.Potential(frozen_phonons, sampling=sampling,slice_thickness=slice_thickness) # default slice thickness is 1Å
	#xs=np.linspace(0,potential_phonon._grid.extent[0],potential_phonon._grid.gpts[0],endpoint=False)
	#ys=np.linspace(0,potential_phonon._grid.extent[1],potential_phonon._grid.gpts[1],endpoint=False)
	#print("FOUND:",xs,sampling,maxrad)

	if not os.path.exists(outDirec+"/xs.npy"):
		np.save(outDirec+"/xs.npy",xs) ; np.save(outDirec+"/ys.npy",ys)
	nLayers=1 ; npts=1
	if layerwise!=0:
		nslices=potential_phonon.num_slices # 14.5 Å --> potential divided into 33 layers
		nth=nslices//layerwise # 33 layers, user asked for 10? 33//10 --> every 3rd layer --> 11 layers total. 
		nth=max(nth,1)		# user asked for 1000? 33//1000 --> 0! oops! just give them every layer instead. 
		potential_phonon = abtem.Potential(frozen_phonons, sampling=sampling,exit_planes=nth,slice_thickness=slice_thickness) # default slice thickness is 1Å
		nLayers=potential_phonon.num_exit_planes
		sliceZs=np.cumsum(potential_phonon._slice_thickness)		# 0.439,0.879,...14.5 (33 entries)
		sliceIDs=potential_phonon.exit_planes				# -1,11,22,33 (-1 implies the first exit plane is at z=0)
		sliceIDs=np.asarray(sliceIDs)+1					# 0,12,23,34 (get rid of weird "-1" to denote entrance wave)
		sliceZs=np.insert(sliceZs,0,0)					# 0,0.439,0.879,...14.5 (34 entries)
		layers=sliceZs[sliceIDs]
		np.save(outDirec+"/layers.npy",layers) # TODO not 100% sure these are truly the plane locations
		print("potential sliced into",nslices,"slices. keeping every",nth,"th layer, we'll have",nLayers,"exit waves")
		#sys.exit()
		#with open(outDirec+"/nLayers.txt",'w') as f:
		#	f.write(str(nLayers))
		# abtem/potentials/iam.py > class Potential() states: 
		# exit_planes : If 'exit_planes' is an integer a measurement will be collected every 'exit_planes' number of slices.
	if plotOut:
		print("preview potential")
		potential_phonon.show() ; plt.savefig(outDirec+"/potential.png")
		print("preview with atoms")
		potentialArray=np.mean(potential_phonon.to_images().array,axis=(0,1)) # sum over configurations and layers
		overplotAtoms=[]
		for t in set(types):
			c=["g","b","r","k","c","m","y"][t-1]
			#overplotAtoms.append( { "xs":avg[types==t,0]+disp[0,types==t,0] , "ys":avg[types==t,1]+disp[0,types==t,1] , "kind":"scatter" , "c":c , "marker":"."} )
			overplotAtoms.append( { "xs":avg[types==t,0] , "ys":avg[types==t,1] , "kind":"scatter" , "c":c , "marker":"."} )
		#contour(potentialArray.T,xs,ys,filename=outDirec+"/potential2.png",overplot=overplotAtoms)
		#fftd,ky,kx=fft2(potential_phonon.to_images.array,ys,xs,maxrad/100) # nicecontour > FFT2 expects y,x index ordering. 
		#contour(fftd.absolute.T,kx,ky,xlabel="kx ($\AA$^-1)",ylabel="ky ($\AA$^-1)",title="FFT(potential)",filename=outDirec+"/potential-reciprocal.png",aspect=1,heatOrContour="pix")
	print("potential extent:",potential_phonon.extent) #; sys.exit()

	# STEP 2 DEFINE PROBE (or plane wave)
	if restartFrom:
		wavefile=restartDirec+"/ewave_"+fileSuffix+".npy" # TODO FOR NOW, WE'RE NOT ALLOWING RESTART FROM npts>1 OR nLayers>1
		print("GET PROBE FROM EXIT WAVE",wavefile)
		ewave=np.load(wavefile)
		if len(np.shape(ewave))==2:
			entrance_waves=abtem.waves.Waves(ewave,  energy=100e3, sampling=sampling, reciprocal_space=False) 
		else:
			meta={'label': 'Frozen phonons', 'type': 'FrozenPhononsAxis' }
			meta=[abtem.core.axes.FrozenPhononsAxis(meta)]
			entrance_waves=abtem.waves.Waves(ewave,  energy=100e3, sampling=sampling, reciprocal_space=False,ensemble_axes_metadata=meta)

		# if we don't do this, we get extent=n*sampling, when really, the sampling should be more of a recommendation (real sampling = size/n)
		entrance_waves.grid.match(potential_phonon) 
	
	elif semiAngle!=0: # CONVERGENT PROBE INSTEAD OF PLANE WAVE (needs probe defined, and scan to denote where probe is positioned)
		probe = abtem.Probe(energy=100e3, semiangle_cutoff=semiAngle)#, C12=20, C10=100) # semiangle_cutoff is the convergence angle of the probe
		probe.grid.match(potential_phonon)
		print(probe.gpts)

		# TODO IF YOU HAVE A GIANT SIMULATION AREA, THIS (or subsequently calculating exit waves if you comment this out) WILL CRASH. I ASSUME BECAUSE THE PROBE IS "ZERO MAGNITUDE" FAR FROM THE CENTER?
		if plotOut:
			print("preview probe")
			#custom_scan=abtem.CustomScan([box[0]/2, box[1]/2])
			#probe.build(custom_scan,max_batch=1).compute()
			#probe.show(max_batch=1) ; plt.savefig(outDirec+"/probe.png") # IF YOU CRASH HERE WITH "integer modulo by zero" TRY TRIMMING YOUR SIM VOLUME
			custom_scan=abtem.CustomScan([box[0]/2, box[1]/2])
			probewave=probe.build(custom_scan,max_batch=1).compute()
			Z=probewave.array ; nx,ny=np.shape(Z)[-2:]
			LX,LY=probe.extent
			xpr=np.linspace(-LX/2,LX/2,nx) ; ypr=np.linspace(-LY/2,LY/2,ny)
			contour(np.absolute(Z).T,xpr,ypr,xlabel="x ($\AA$)",ylabel="y ($\AA$)",title="probe, real space",filename=outDirec+"/probe.png")

		# STEP 3, PROBE POSITION (user defined, or center of the sim volume)
		# Case 1: You are using abtem's probe
		if not modifyProbe:
			if len(probePositions)>0:	# multiple probe positions specified!
				custom_scan=abtem.CustomScan(probePositions)
				npts=len(probePositions)
			else:				# or default to the middle
				custom_scan=abtem.CustomScan([box[0]/2, box[1]/2])
				npts=1
		# Case 2: you are supplying your own probe (modifyProbe should be a function, which in-place modify's an array).
		# BEWARE: abtem will ignore custom_scan and simply overlay the pixel array of the probe over the pixel array of the potential! we will need to handle this by "rolling" the array you update, but you can only specify one probe position at a time! 
		else:
			# extract pixel array from probe, pass it to probePositions to modify
			custom_scan=abtem.CustomScan([box[0]/2, box[1]/2]) ; npts=1
			probewave=probe.build(custom_scan,max_batch=1).compute()
			Z=probewave.array ; nx,ny=np.shape(Z)[-2:]
			LX,LY=probe.extent
			# convention is: modifyProbe should receive positive/negative xs,ys (for consistent treatment with abtem probe)
			xpr=np.linspace(-LX/2,LX/2,nx) ; ypr=np.linspace(-LY/2,LY/2,ny)
			modifyProbe(Z,xpr,ypr)

			# preview custom probe BEFORE applying shifts to it:
			if plotOut:
				pr=probewave.array
				print("probewave",pr.shape)
				while len(pr.shape)>2:
					print("probewave",pr.shape)
					pr=pr[0,:,:]
					print("probewave",pr.shape)
					xlb="x ($\AA$)"
				kwargs={"xlabel":"x ($\AA$)","ylabel":"y ($\AA$)","aspect":1,"heatOrContour":"pix"}
				contour(np.real(pr).T,xpr,ypr,title="Re(probe)",filename=outDirec+"/probe_R.svg",**kwargs)
				contour(np.imag(pr).T,xpr,ypr,title="Im(probe)",filename=outDirec+"/probe_I.svg",**kwargs)
				contour(np.absolute(pr).T,xpr,ypr,title="Mag(probe)",filename=outDirec+"/probe_M.svg",**kwargs)
				contour(np.angle(pr).T,xpr,ypr,title="Phi(probe)",filename=outDirec+"/probe_A.svg",**kwargs)
				fftd,ky,kx=fft2(pr,ypr,xpr,maxrad/100) # nicecontour > FFT2 expects y,x index ordering. 
				kwargs["xlabel"]="kx ($\AA$^-1)" ; kwargs["ylabel"]="ky ($\AA$^-1)"
				kwargs["xlim"]=[-1,1] ; kwargs["ylim"]=[-1,1]
				contour(fftd.real.T,kx,ky,title="Re(probe^-1)",filename=outDirec+"/rprobe_R.svg",**kwargs)
				contour(fftd.imag.T,kx,ky,title="Im(rprobe^-1)",filename=outDirec+"/rprobe_I.svg",**kwargs)
				contour(np.absolute(fftd).T,kx,ky,title="Mag(rprobe^-1)",filename=outDirec+"/rprobe_M.svg",**kwargs)
				contour(np.angle(fftd).T,kx,ky,title="Phi(rprobe^-1)",filename=outDirec+"/rprobe_A.svg",**kwargs)

			# handle rolling based on probePositions
			if len(probePositions)>1:
				print("WARNING! If modifyProbe is set, you should only set one probePosition location! we will only be using your first probePosition. remove the remainder to suppress this warning")
				probePositions=[probePositions[0]]
			dx=probePositions[0][0]-(xpr[0]-xs[0]) # potential's coordinate system vs probe's coordinate system, + shift
			dy=probePositions[0][1]-(ypr[0]-ys[0])
			di=int(round( (dx)/(xs[1]-xs[0]) ))
			dj=int(round( (dy)/(ys[1]-ys[0]) ))
			rolled=np.roll(np.roll(Z,di,axis=0),dj,axis=1)
			Z*=0 ; Z+=rolled
			# set up remaining assumed parameters
			#custom_scan=abtem.CustomScan([box[0]/2, box[1]/2]) # doesn't actually matter, but the var is assumed later
			npts=1

	else: # PLANE WAVE INSTEAD OF CONVERGENT PROBE (just define the plane wave params)
		plane_wave = abtem.PlaneWave(energy=100e3, sampling=sampling)

	if plotOut:
		print("preview atoms")
		atoms=atomStack[0]
		abtem.show_atoms(atoms,plane="xy"); plt.title("beam view")
		if semiAngle!=0:
			if len(probePositions)>0:
				#plt.plot(*np.asarray(probePositions).T)
				for i,p in enumerate(probePositions):
					plt.plot([p[0]],[p[1]]) ;  plt.annotate(str(i),p)
			else:
				plt.plot(box[0]/2,box[1]/2)
		plt.savefig(outDirec+"/beam.png")

		if semiAngle!=0:
			pP=np.asarray(probePositions)
			if len(pP)==0:
				pP=np.asarray([[box[0]/2,box[1]/2]])
			plt.xlim([min(pP[:,0])-5,max(pP[:,0])+5])
			plt.ylim([min(pP[:,1])-5,max(pP[:,1])+5])
			for t in set(types):
				filtered=avg[types==t,:2]
				distToAtoms=np.sqrt(np.sum((filtered-pP[None,0,:])**2,axis=1))
				n=np.argmin(distToAtoms) ; xy=filtered[n]-np.asarray([.5,0])
				plt.annotate(atomTypes[t-1],xy)
			plt.savefig(outDirec+"/beam2.png")

		if semiAngle!=0 and len(probePositions)>0 and modifyProbe:
			for i,p in enumerate(probePositions):
				#print("i,p",i,p,np.shape(pr),np.shape(xs),np.shape(ys),pr,xs,ys)
				#plt.clf()
				#plt.xlim([min(pP[:,0])-5,max(pP[:,0])+5])
				#plt.ylim([min(pP[:,1])-5,max(pP[:,1])+5])
				#plt.contourf(xs+p[0],ys+p[1],np.absolute(pr).T,levels=50)
				#for t in set(types):
				#	xscatter=avg[types==t,0] ; yscatter=avg[types==t,1] ; c=["g","b","r","k","c","m","y"][t-1]
				#	plt.scatter(xscatter,yscatter,c=c)
				#plt.scatter(p[0],p[1],c="k")
				#plt.gca().set_aspect('equal')
				#plt.savefig(outDirec+"/beampos_"+str(i)+".png")
				i1=np.argmin(np.absolute(xs-(min(pP[:,0])-5)))
				i2=np.argmin(np.absolute(xs-(min(pP[:,0])+5)))
				j1=np.argmin(np.absolute(xs-(min(pP[:,1])-5)))
				j2=np.argmin(np.absolute(xs-(min(pP[:,1])+5)))
				contour(np.absolute(pr[i1:i2,j1:j2]).T,xs[i1:i2],ys[j1:j2],filename=outDirec+"/beampos_"+str(i)+"a.png",title="beam")#,xlim=[min(pP[:,0])-5,max(pP[:,0])+5],ylim=[min(pP[:,1])-5,max(pP[:,1])+5])
				contour(np.absolute(pr[i1:i2,j1:j2]).T,xs[i1:i2],ys[j1:j2],filename=outDirec+"/beampos_"+str(i)+"b.png",overplot=overplotAtoms,title="beam+atoms")#,xlim=[min(pP[:,0])-5,max(pP[:,0])+5],ylim=[min(pP[:,1])-5,max(pP[:,1])+5])
				contour(potentialArray[i1:i2,j1:j2].T,xs[i1:i2],ys[j1:j2],filename=outDirec+"/beampos_"+str(i)+"c.png",title="potential")#,xlim=[min(pP[:,0])-5,max(pP[:,0])+5],ylim=[min(pP[:,1])-5,max(pP[:,1])+5])

		abtem.show_atoms(atoms,plane="yz"); plt.title("side view") ; plt.savefig(outDirec+"/side.png")


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
			exit_waves=abtem.waves.Waves(ewave,  energy=100e3, sampling=sampling, reciprocal_space=False,ensemble_axes_metadata=meta)
			entrance_waves.grid.match(potential_phonon) 
	elif semiAngle!=0:
		if modifyProbe:
			exit_waves = probewave.multislice(potential_phonon).compute()
		else:
			exit_waves = probe.multislice(potential_phonon,scan=custom_scan,max_batch=1).compute()
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
		print("saveExitWave=True:",np.shape(exit_waves.array))
		for l in range(nLayers):
			for p in range(npts): # for psi, we do: psi_e9_[fp9]_[l8]_[p7].npy,
				wavefile=outDirec+"/ewave_"+fileSuffix +\
					{True:"_l"+str(l),False:""}[nLayers>1] +\
					{True:"_p"+str(p),False:""}[npts>1] + ".npy"
				waveout=exit_waves.array
				if nLayers>1:
					waveout=waveout[:,l,:,:]
				if npts>1:
					waveout=waveout[:,p,:,:]
				waveout=np.squeeze(waveout)
				if cropExitWave: # e.g. [[0,10],[0,10]] to crop to the first 10x10 unit cells in x,y when saving (reduces file sizes)
					(xi,xf),(yi,yf)=cropExitWave ; xi*=a ; xf*=a ; yi*=b ; yf*=b
					xmask=np.zeros(len(xs)) ; xmask[xs>=xi]=1 ; xmask[xs>xf]=0
					ymask=np.zeros(len(ys)) ; ymask[ys>=yi]=1 ; ymask[ys>yf]=0
					waveout=waveout[xmask==1,:][:,ymask==1]	# masking by rows and columns instead of a 2D mask means we keep it 2D
				np.save(wavefile,np.squeeze(waveout))

	# STEP 5 CONVERT TO DIFFRACTION SPACE
	print("converting exit wave to k space",end=" ")
	sys.stdout.flush() # https://stackoverflow.com/questions/35230959/printfoo-end-not-working-in-terminal
	start=time.time()
	#diffraction_complex = exit_waves.diffraction_patterns(max_angle=maxrad,block_direct=False,return_complex=True)
	#zr=diffraction_complex.array
	#kx=diffraction_complex.sampling[-2]*diffraction_complex.shape[-2]/2 ; ky=diffraction_complex.sampling[-1]*diffraction_complex.shape[-1]/2
	#kxs=np.linspace(-kx,kx,diffraction_complex.shape[-2]) ; kys=np.linspace(-ky,ky,diffraction_complex.shape[-1])
	zr,kys,kxs=fft2(exit_waves.array,ys,xs,maxrad/100) # using nicecontour's FFT2 function, which expects y,x indices

	if not os.path.exists(outDirec+"/kxs.npy"):
		np.save(outDirec+"/kxs.npy",kxs) ; np.save(outDirec+"/kys.npy",kys)
	print("took:",time.time()-start)

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
			zo=np.squeeze(zo)
			np.save(psifile,zo)
	print("took:",time.time()-start)

	if restartFrom and deleteInputWave:
		print("delete previous input wave file")
		os.remove(restartDirec+"/ewave_"+fileSuffix+".npy")
	#return zr # even if layerwise!=0, we'll end up with the last (thickest) psi. 

	# SEE OLD VERSION 0.20 FOR CALCULATE FORCES

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
		#vfft=np.fft.fft(velocities,axis=0) # FFT on displacements tells displacements associated with each frequency. THIS IS JUST EIGENVECTOR?
		ws=np.fft.fftfreq(len(ts),ts[1]-ts[0])
		n=int(len(ws)/2)
		ws/=dt # convert to THz: e.g. .002 picosecond timesteps, every 10th timestep logged
		#print(np.shape(dfft)) # nω, na, 3

		print("saving dDOS and vDOS")
		ddos=np.sum(np.absolute(dfft[:,:,1]),axis=1) # nω, na, 3 --select-Y--> nω, na --sum-a--> nω
		#vdos=np.sum(np.absolute(vfft[:,:,1]),axis=1) # no longer reading in velocities (potentially huge matrix) just to get vDOS. 
		vdos=ddos*ws	# u(x,t)=exp(i(ω*t-k*x)) --> v(x,t) = du(x,t)/dt = - i ω u(x,t)
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
		#plot(Xs_v,Ys_v,markers=markers,xlim=[0,Emax],ylim=[0,None],xlabel="frequency (THz)",ylabel="DOS",title="FFT(velo)",labels=[""]*len(Xs_v))#,filename=outDirec+"/DOS_v.png")

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

def tile(pos): # a,xyz
	lx,ly,lz=a*nx,b*ny,c*nz
	if max(tiling)<=1:
		return pos
	cloned=[]
	for i in range(tiling[0]):
		for j in range(tiling[1]):
			for k in range(tiling[2]):
				translate=np.asarray([i*lx,j*ly,k*lz])
				cloned.append(pos+translate)

	return np.reshape(cloned,(len(pos)*len(cloned),3)) # whichTile,whichAtom,xyz --> moreAtoms,xyz

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

def maskDOS():
	os.makedirs(outDirec+"/DOS",exist_ok=True)
	lines=open(outDirec+"/masks.txt")
	global shape,center,radius,length,width,omegascaling,title ; omegascaling=1 ; title=""
	kxs=np.load(outDirec+"/kxs.npy") ; kys=np.load(outDirec+"/kys.npy") ; mask=np.zeros((len(kxs),len(kys)))
	ct=-1
	Xs=[] ; Ys=[] ; lbls=[]
	for l in lines:
		if len(l)==0 or l[0]=="#":
			continue
		print(l) ; ct+=1 ; title=str(ct) ; mask*=0
		l=l.lstrip("#").strip()
		exec(l,globals())
		if shape=="round":
			radii=np.sqrt( (kxs[:,None]-center[0])**2 + (kys[None,:]-center[1])**2 )
			mask[radii<=radius]=1
		xs,ys,lbl=eDOS(mask,title={True:"mask"+str(ct),False:title}[len(title)==0],omegascaling=omegascaling)
		for i in range(len(xs)):
			Xs.append(xs[i]) ; Ys.append(ys[i]) ; lbls.append(lbl[i])
	plot(Xs,Ys,xlabel="frequency (THz)",ylabel="DOS (-)",markers=rainbow(len(Xs)),filename=outDirec+"/DOS/maskDOS.svg",title="raw DOS",xlim=[0,maxFreq],ylim=[0,None],heatOrContour="pix",labels=lbls)


def eDOS(mask=None,title="",omegascaling=1):
	print("PROCESSING eDOS: title=\""+str(title)+"\", omegascaling="+str(omegascaling))
	os.makedirs(outDirec+"/DOS",exist_ok=True)
	ws=np.load(outDirec+"/ws.npy")

	chunks=psiFileNames(prefix="ivib")
	print(chunks)
	Xs=[] ; Ys=[] ; lbls=[]
	for fs in chunks:
		if not os.path.exists(fs[0]): # e.g. we deleted early layers' files or something 
			continue
		psi=np.load(fs[0])
		if mask is not None:
			psi*=mask
		Xs.append(ws) ; Ys.append(np.sum(np.absolute(psi),axis=(1,2))*(ws**omegascaling))
		lbls.append(fs[0])
	Ys=[ y[ws>1] for y in Ys ] ; Xs=[ x[x>1] for x in Xs ]
	#Ys=[ y/np.amax(y[ws>2]) for y in Ys ]
	if mask is not None:
		 diffraction(mask,title)

	fo=outDirec+"/DOS/eDOS"+{True:"",False:"_"+title}[len(title)==0]+".png"
	plot(Xs,Ys,xlabel="frequency (THz)",ylabel="DOS (-)",labels=lbls,markers=rainbow(len(Xs)),filename=fo,title="raw DOS")
	out=np.zeros((len(Xs[0]),len(Ys)+1)) ; out[:,0]=Xs[0]
	for i,ys in enumerate(Ys):
		out[:,i+1]=ys
	np.savetxt(fo.replace(".png",".txt"),out,header="w\ty0...")
	#Ys=[ y/np.amax(y[Xs[0]>2]) for y in Ys ]
	#plot(Xs,Ys,xlabel="frequency (THz)",ylabel="DOS (-)",labels=lbls,markers=rainbow(len(Xs)),filename=outDirec+"/eDOS_norm.png",title="peak normed")
	#Ys=[ y*ws[ws>1]**2 for y in Ys ]
	#Ys=[ y/np.amax(y[Xs[0]>2]) for y in Ys ]
	#plot(Xs,Ys,xlabel="frequency (THz)",ylabel="DOS (-)",labels=lbls,markers=rainbow(len(Xs)),filename=outDirec+"/eDOS_wscale.png",title="scaled by $\omega$")
	return Xs,Ys,lbls

def sliceE(nth=1):
	ws=np.load(outDirec+"/ws.npy")
	kxs=np.load(outDirec+"/kxs.npy")
	kys=np.load(outDirec+"/kys.npy")
	os.makedirs(outDirec+"/slices",exist_ok=True)
	smearOmega=0
	if os.path.exists(outDirec+"/smearOmega.txt"):
		smearOmega=float( open(outDirec+"/smearOmega.txt",'r').readlines()[0] )

	chunks=psiFileNames(prefix="ivib")
	print(chunks)
	for fs in chunks:
		psi=np.load(fs[0])
		if smearOmega>0:
			from scipy.ndimage import gaussian_filter
			psi=gaussian_filter(psi,(smearOmega,0,0))

		#radii=np.sqrt(kxs[:,None]**2+kys[None,:]**2)
		#psi[:,radii<.05]=0

		fo=fs[0].split("/") ; fo.insert(2,"slices") ; fo="/".join(fo) # outputs/inputfilename/ivib.npy --> outputs/inputfilename/slices/ivib.npy
		for i in tqdm(range(len(ws))):
		#for i in [74]:
			if i%nth!=0:
				continue
			if (maxFreq is not None) and ( ws[i]>maxFreq or ws[i]<-1*maxFreq ):
				continue
			#out=np.absolute(psi[i])
			i1=max(i-1,0) ; i2=i+2				# ARE YOUR FREQUENCY BINS TOO FINE AND YOUR IMAGES TOO DIM/PIXELATED/CRISPY?
			out=np.sum(np.absolute(psi[i1:i2]),axis=0) 	# TRY A ROLLING MEAN A FEW BINS WIDE
			#out=np.arctan2(np.real(psi[i]),np.imag(psi[i]))
			out=np.sqrt(np.sqrt(out))
			contour(out.T,kxs,kys,filename=fo.replace(".npy","_e"+str(i)+".png"),xlabel="$\AA$^-1",ylabel="$\AA$^-1",title=str(np.round(ws[i],2))+" THz",aspect=1,heatOrContour="pix")#,zlim=zlim,cmap='Spectral')
			#contour(out.T**2,kxs,kys,filename=fo.replace(".npy","_e"+str(i)+"_2.png"),xlabel="$\AA$^-1",ylabel="$\AA$^-1",title=str(np.round(ws[i],2))+" THz",aspect=1,heatOrContour="pix")#,xlim=[-1,1],ylim=[-1,1])#,zlim=zlim,cmap='Spectral')

def sliceE2(nth=1):
	ws=np.load(outDirec+"/ws.npy")
	kxs=np.load(outDirec+"/kxs.npy")
	kys=np.load(outDirec+"/kys.npy")
	os.makedirs(outDirec+"/slices2",exist_ok=True)

	chunks=psiFileNames(prefix="ivib")
	print(chunks)
	for fs in chunks:
		psi=np.load(fs[0])
		fo=fs[0].split("/") ; fo.insert(2,"slices2") ; fo="/".join(fo) # outputs/inputfilename/ivib.npy --> outputs/inputfilename/slices/ivib.npy
		for i in tqdm(range(len(ws))):
			if i%nth!=0:
				continue
			if ws[i]<0:
				continue
			if (maxFreq is not None) and ( ws[i]>maxFreq or ws[i]<-1*maxFreq ):
				continue
			n=np.argmin(np.absolute(ws+ws[i]))
			p=np.absolute(psi[i])+np.absolute(psi[n])
			contour(p.T,kxs,kys,filename=fo.replace(".npy","_e"+str(i)+".png"),xlabel="$\AA$^-1",ylabel="$\AA$^-1",title=str(ws[i])+" THz",aspect=1)#,zlim=zlim,cmap='Spectral')



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
def loopDispersions():
	lines=open(outDirec+"/dispersions.txt")
	global m,xi,yi,xlim,ylim,title,includeNegatives,omegaScaling,smearOmega
	for l in lines:
		includeNegatives=True ; omegaScaling=False
		smearOmega=0									# defaulting sequence: 0, smearOmega.txt, dispersions.txt (line in dispersions.txt overrides all)
		if os.path.exists(outDirec+"/smearOmega.txt"):
			smearOmega=float( open(outDirec+"/smearOmega.txt",'r').readlines()[0] )
		l=l.split("#")[0].strip()
		if len(l)<=10:
			continue
		print("loopDispersions: processing line:",l)
		exec(l,globals())
		dispersion(m,xi,yi,xlim,ylim,title,includeNegatives,omegaScaling,smearOmega)
	# how does this work? copy and past something like below into a "dispersions.txt" file and we'll loop through each line
	# m=0      ; xi=0   ; yi=2/b ; xlim=[0,6/a] ; ylim=[-np.inf,np.inf] ; title="Xo" 	# AlN, Xo
	# m=0      ; xi=0   ; yi=0   ; xlim=[0,6/a] ; ylim=[-np.inf,np.inf] ; title="Xc" 	# AlN, Xc
	# m=np.inf ; xi=0   ; yi=0   ; xlim=[0,6/a] ; ylim=[0,6/b]          ; title="Mc" 	# AlN, Xc
	# m=np.inf ; xi=2/a ; yi=0   ; xlim=[0,6/a] ; ylim=[0,6/b]          ; title="Mo" 	# AlN, Xc

loadedChunks={}
def dispersion(m=None,xi=None,yi=None,xlim=None,ylim=None,title=None,includeNegatives=True,omegaScaling=0,smearOmega=0):

	os.makedirs(outDirec+"/dispersions",exist_ok=True)
	
	#global a,b,c,beamAxis
	# copied from preprocessPositions, since we need lattice parameters a,b,c to line up with our diffraction pattern
	#if beamAxis!=2: # default is to look down z, so we need to flip axes if we want to look down a different direction
	#	rollby=[-1,1][beamAxis] # to look down x, roll -1 [x,y,z] --> [y,z,x]. to look down y, roll +1 [x,y,z] --> [z,x,y]
	#	a,b,c=np.roll([a,b,c],rollby)
	#	beamAxis=2 # update so we don't reroll if loopDispersions is calling this multiple times?? (TODO technically i don't think we should be rolling here....
	#print(a,b,c) ; sys.exit()

	if m is None:
		# TODO hardcoding linescan here, but should figure out a way to make them passable via command-line args or something
		#m=0 ; xi=0 ; yi=4/b ; xlim=[0,12/a] ; ylim=[-np.inf,np.inf] ; title="Xo"	# Si, Xo
		#m=0 ; xi=0 ; yi=0 ; xlim=[0,12/a] ; ylim=[-np.inf,np.inf] ; title="Xc"		# Si, Xc
		#m=1 ; xi=0 ; yi=0 ; xlim=[0,6/a] ; ylim=[-np.inf,np.inf] ; title="Kc" 		# Si, Kc
		#m=1 ; xi=-2/a ; yi=2/b ; xlim=[-2/a,4/a] ; ylim=[-np.inf,np.inf] ; title="Ko"	# Si, Ko
		#m=0 ; xi=0 ; yi=2/b ; xlim=[0,6/a] ; ylim=[-np.inf,np.inf] ; title="Xo" 	# AlN, Xo
		m=0 ; xi=0 ; yi=0 ; xlim=[-6/a,6/a] ; ylim=[-np.inf,np.inf] ; title="Xc" 	# AlN, Xc
		#m=np.inf ; xi=0 ; yi=0 ; xlim=[0,6/a] ; ylim=[0,6/b] 	; title="Mc" 		# AlN, Xc
		#m=np.inf ; xi=2/a ; yi=0 ; xlim=[0,6/a] ; ylim=[0,6/b] 	; title="Mo" 	# AlN, Xc

	#psi=np.load(outDirec+"/ivib.npy")
	#ws=np.load(outDirec+"/ws.npy")
	kxs=np.load(outDirec+"/kxs.npy")
	kys=np.load(outDirec+"/kys.npy")

	chunks=psiFileNames(prefix="ivib")
	print(chunks)

	ijs=[]
	if m<1:
		for i in tqdm(range(len(kxs))):
			x=kxs[i] ; y=m*(x-xi)+yi
			if x<xlim[0] or x>xlim[1] or y<ylim[0] or y>ylim[1]:
				continue
			j=np.argmin(np.absolute(kys-y)) ; ijs.append([i,j])
	else:
		for j in tqdm(range(len(kys))):
			y=kys[j] ; x=1/m*(y-yi)+xi
			if x<xlim[0] or x>xlim[1] or y<ylim[0] or y>ylim[1]:
				continue
			i=np.argmin(np.absolute(kxs-x)) ; ijs.append([i,j])

	global loadedChunks # Why are we loading ivibs into a global dict? loading and reloading when generating many dispersions (loopDispersions) takes forever, wastes ram
	for fs in chunks:
		if fs[0] not in loadedChunks.keys():
			print("load ivib/diff",fs[0])
			psi=np.load(fs[0])
			#from scipy.ndimage import gaussian_filter
			loadedChunks[fs[0]]=psi # gaussian_filter(psi,1)
		else:
			print("reusing previously loaded ivib/diff",fs[0])
			psi=loadedChunks[fs[0]]
		filelabel=fs[0].split("/")[-1].replace("ivib","").replace(".npy","")
		diff=np.load(outDirec+"/diff"+filelabel+".npy")
		ws=np.load(outDirec+"/ws.npy") # reload ws each time inside the loop just in case we trimmed it last cycle
		#print(np.amax(psi),np.nanmax(psi),np.amin(psi),np.nanmin(psi))
		print("dispersions: title:",title)

		#radii=np.sqrt(kxs[:,None]**2+kys[None,:]**2)
		#psi[:,radii<.05]=0

		overplot={"xs":[],"ys":[],"kind":"line","color":"r","linestyle":":"} 
		#overplot={"xs":[],"ys":[],"kind":"scatter","color":"r","marker":"."} 
		sliced=[] ; ks=[]
		for n,ij in enumerate(tqdm(ijs)):
			i,j=ij
			sliced.append(psi[:,i,j])
			distance=np.sqrt((kxs[i]-xi)**2+(kys[j]-yi)**2)
			sign=1
			if (m>=0 and m<=1) and ( kxs[i]-xi < 0 ): 	#    .-'  / (both positive
				sign=-1					# -'     /   slope, but 
			if (m>1) and ( kys[j]-yi < 0 ):			#   vs  /   check x or y)
				sign=-1
			if (m<0 and m>=-1) and ( kxs[i]-xi < 0 ):
				sign=-1
			if (m<-1) and ( kys[j]-yi < 0 ):
				sign=-1
			ks.append(distance*sign)
			overplot["xs"].append(kxs[i]) ; overplot["ys"].append(kys[j])
		
		sliced=np.asarray(sliced)
		# blurring takes eons! 
		# sliced=gaussianBlur([sliced,np.linspace(0,1,len(ks)),np.linspace(0,1,len(ws))],.01)
		# from scipy.ndimage import gaussian_filter
		# sliced=gaussian_filter(sliced,(1,1))

		# perform scaling and trimming post-slice! this way we don't need to copy (a potentially giant) psi in memory
		if maxFreq is not None:
			sliced=sliced[:,ws<=maxFreq] ; ws=ws[ws<=maxFreq]
			sliced=sliced[:,ws>=-maxFreq] ; ws=ws[ws>=-maxFreq]
		if not includeNegatives:
			sliced=sliced[:,ws>=0] ; ws=ws[ws>=0]

		for i in range(omegaScaling):
			sliced[:,ws>=0]*=ws[None,ws>=0]
			sliced[:,ws<0]*=-1*ws[None,ws<0]



		#print(np.amax(sliced),np.amin(sliced))
		sliced=np.absolute(sliced)
		if smearOmega>0:
			from scipy.ndimage import gaussian_filter
			sliced=gaussian_filter(sliced,(0,smearOmega))
		sliced[sliced==0]=np.amin(sliced[sliced>0])
		#print(np.amax(sliced),np.amin(sliced))
		contour(np.log(diff).T,kxs,kys,filename=outDirec+"/dispersions/"+title+"_disp_linescan"+filelabel+".png",xlabel="$\AA$^-1",ylabel="$\AA$^-1",title="dispersion masked diffraction image",overplot=[overplot],heatOrContour="pix")
		contour(np.sqrt(sliced).T,ks,ws,xlabel="k ($\AA$^-1)",ylabel="frequency (THz)",title="dispersion",filename=outDirec+"/dispersions/"+title+"_dispersion"+filelabel+".png",heatOrContour="pix") # i prefer sqrt scale to log scale, as it sort of "equalizes" big numbers
		np.save(outDirec+"/dispersions/"+title+"_dispersion"+filelabel+"_Zs.npy",sliced)
		np.save(outDirec+"/dispersions/"+title+"_dispersion"+filelabel+"_ks.npy",ks)
		np.save(outDirec+"/dispersions/"+title+"_dispersion"+filelabel+"_ws.npy",ws)


def diffraction(mask=None,title=""):
	kxs=np.load(outDirec+"/kxs.npy")
	kys=np.load(outDirec+"/kys.npy")
	#os.makedirs(outDirec+"/slices",exist_ok=True)

	# reading and grouping psi files (instead of ivib files) via psiFileNames assumes certain parameters depending on mode
	if mode=="PZ":
		global energyCenters ; energyCenters=np.load(outDirec+"/ws.npy")
	if mode=="JACR":
		global ts ; ts=np.arange(1e6)

	chunks=psiFileNames(prefix="psi")
	print(chunks)
	for fs in chunks:
		fo=fs[0].replace("_t0","").replace("_e0","").replace(".npy","").replace("psi","diff")
		if os.path.exists(fo+".npy"):
			print("REIMPORT DIFF")
			diff=np.load(fo+".npy")
		else:
			print("ASSEMBLE DIFF FROM PSIS")
			diff=np.zeros((len(kxs),len(kys)))
			for f in tqdm(fs): # either psi_e or psi_t, either loop through energy, or time chunks
				psi=np.load(f)
				if len(np.shape(psi))==2:
					diff+=np.absolute(psi)
				else:
					for p in psi:	# for each energy or timechunk, loop through individual FP configs, or individual timesteps. 
						diff+=np.absolute(p)

			np.save(fo+".npy",diff)
		logdiff=np.log(diff)
		if mask is not None:
			logdiff*=mask
			fo=fo.split("/") ; fo.insert(2,"DOS") ; fo="/".join(fo)
		contour(logdiff.T,kxs,kys,filename=fo+{True:"",False:"_"+title}[len(title)==0]+".png",xlabel="$\AA$^-1",ylabel="$\AA$^-1",title="diffraction image",aspect=1)#,heatOrContour="pix")


#,zlim=zlim,cmap='Spectral')
			#contour(np.log(np.absolute(sliced)),kxs,ws)

def nukeHighFreqs(maxF=50):
	ws=np.load(outDirec+"/ws.npy")
	#print(ws)
	mask=np.ones(len(ws))
	mask[ws>maxF]=0
	mask[ws<-1*maxF]=0
	vibes=glob.glob(outDirec+"/ivib*.npy")
	for f in vibes:
		print(f)
		ivib=np.load(f)
		ivib=ivib[mask==1]
		np.save(f,ivib)
	print("ws")
	ws=ws[mask==1]
	np.save(outDirec+"/ws.npy",ws)

# python3 inputs/infile1.txt outputs/outdirec1 outputs/outdirec2 outputs/outdirec3...
# we will create a new output direc and stack up all like psi files
# def combinePZruns():
#

# Use one run's exit wave as the input wave for the subsequent run. we'll update globals, set saveExitWave=True (so we can use the real-space exit wave as the next run's input wave), and loop through calling main() functions. inspired by cycleGenerator.py, which solved the issues of the previous cycling code (which used to be inside readInputFile, and would cause problems if one run failed partway through)
# This has couple of use cases. In AlN_5m_01, we simulate an insanely thick structure from a thin MD run by repeately running over the same structure. In SiGe_5m_02, we ran an MD run with a huge structure, but RAM consumption is too high when we try to run abEELS over it, so each cycle trims to a different layer of the whole. 
# optionally, place a file titled "cycleDefinitions.txt" in the run's output folder, containing a dict of globals you want set for each run.
# For example, AlN_5m_02's cycleDefinitions.txt might contain: 
# cycleDefinitions={ "trim":[ [[0,15],[],[0,10]] , [[15,30],[],[0,10]] , [[30,45],[],[0,10]] ] , "t0":[0,200,400] }
# where the first run trims in x 0-15 and y 0-10 and starts at timestep 0, the second run trims 15-13 and 0-10 and starts and timestep 200 and so on. this is to avoid artifacts from re-using the same set of atom's and/or the same timesteps for the pretend subsequent layers. 
# Similarly, for SiGe_5m_02, cycleDefinitions.txt contains: 
# cycleDefinitions={ "trim":[ [[15,35],[15,35],[0,2]] , [[15,35],[15,35],[2,4]] , [[15,35],[15,35],[4,6]] , [[15,35],[15,35],[6,8]] , ... ] }
# where the first run uses the top 2 unit cells' worth of layers, second run picks up at layers 2 and 3, and so on. no pretending of subsequent layers here, but we blow up ram if we try to use the entire 50x50x20 UC volume
def runCycles():
	global saveExitWave,deleteInputWave,restartFrom,restartDirec,outDirec,cycleDefinitions
	cycleDefinitions={}
	print(outDirec)
	if os.path.exists(outDirec+"/cycleDefinitions.txt"):
		lines=open(outDirec+"/cycleDefinitions.txt").readlines()
		exec("".join(lines),globals())
	print("RUNCYCLES >>>","cycleDefinitions:",cycleDefinitions)
	letters="abcdefghijklmnopqrstuvwxyz"
	letters=[ letters[i]+letters[j] for i in range(26) for j in range(26) ]
	nameOriginal=outDirec.replace("outputs/","") ; name=nameOriginal
	for n in range(numCycles):
		# reinitialize all globals every cycle! can't have things like axis rolling for beamDirection messing us up (swapping a/b/c etc)
		readInputFile(infile)
		saveExitWave=True ; deleteInputWave=True
		if n!=0:
			restartFrom=name
			restartDirec="outputs/"+restartFrom
			name=nameOriginal+letters[n]
		for k in cycleDefinitions.keys():
			val=cycleDefinitions[k][n]
			exec(k+"="+str(val),globals())
			print("RUNCYCLES >>>",trim,t0)
		outDirec="outputs/"+name
		os.makedirs(outDirec,exist_ok=True)
		print("RUNCYCLES >>>","restartFrom",restartFrom,"saveTo",outDirec)
		if mode=="JACR":
			main_JACR()
		else:
			main_PZ()
		if n%5==0:
			outDirec="outputs/"+nameOriginal
			nukeCyclePsis()

# if you ran runCycles, you're probably left with a bunch of maybe-useless psi files inside each output folder. these take up space, but if you have no plans for them, then they're useless. the ivib file exists, so you probably don't need the psi files. so we'll loop through, delete them, and replace them with an empty file of the same name, which means subsequent runs of runCycles won't try to reprocess if it finds a missing psi file.
# BEWARE! if you run this while runCycles is still running, you'll nuke a psi file which will later be needed to construct the ivib!  
def nukeCyclePsis():
	print("NUKE PSIS FROM",outDirec,"AND CHILDREN")
	letters="abcdefghijklmnopqrstuvwxyz"
	letters=[ letters[i]+letters[j] for i in range(26) for j in range(26) ]
	for n in range(numCycles):
		for t in range(maxTimestep):
			if not os.path.exists( outDirec+letters[n]+"/ivib.npy" ):
				continue
			fo=outDirec+letters[n]+"/psi_t"+str(t)+".npy"
			if os.path.exists(fo):
				print("n",n,"t",t)
				os.remove(fo)
				np.save(fo,[])

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

def JACRAverager():
	direcsIn=["Si_5m_09a","Si_5m_09b","Si_5m_09c","Si_5m_09d","Si_5m_09e","Si_5m_09f","Si_5m_09g","Si_5m_09h","Si_5m_09i","Si_5m_09j"]
	direcOut="Si_5m_09aj"
	global outDirec ; outDirec="outputs/"+direcOut				# 
	os.makedirs(outDirec,exist_ok=True)					# create output folder
	shutil.copy("inputs/"+direcsIn[0]+".txt","inputs/"+direcOut+".txt")	# create dummy input file (from first direcIn)
	for f in [ "kxs.npy","kys.npy","ws.npy","xs.npy","ys.npy"]:		# copy in files we'll want for processing later
		shutil.copy("outputs/"+direcsIn[0]+"/"+f,outDirec)
	ivibs=[ np.load("outputs/"+d+"/ivib.npy") for d in direcsIn ]			# read in each ivib data cube (ws,kx,ky)
	ivib=np.mean(ivibs,axis=0)
	np.save(outDirec+"/ivib.npy",ivib)

# IF YOU RUN OUT OF RAM AT THE END OF JACR (for example) BECAUSE YOU HAVE A LOT OF HUGE PSIS, USE THIS TO TRIM IN K SPACE
def trimPsisToIvib():
	kxi=-1 ; kxf=1 ; kyi=-1 ; kyf=1
	kx=np.load(outDirec+"/kxs.npy") ; ky=np.load(outDirec+"/kys.npy")
	shutil.copy(outDirec+"/kxs.npy",outDirec+"/kxs_original.npy")
	shutil.copy(outDirec+"/kys.npy",outDirec+"/kys_original.npy")
	i1=len(kx[kx<=kxi])-1 ; i2=len(kx[kx>=kxf])-1
	j1=len(ky[ky<=kyi])-1 ; j2=len(ky[ky>=kyf])-1
	kx=kx[i1:-i2] ; ky=ky[j1:-j2]
	np.save(outDirec+"/kxs.npy",kx) ; np.save(outDirec+"/kys.npy",ky)
	diff=np.load(outDirec+"/diff.npy")
	shutil.copy(outDirec+"/diff.npy",outDirec+"/diff_original.npy")
	diff=diff[i1:-i2,j1:-j2]
	np.save(outDirec+"/diff.npy",diff)
	psi_t=np.zeros((maxTimestep,len(kx),len(ky)),dtype=complex)
	for i in tqdm(range(maxTimestep)):
		psi_t[i,:,:]=np.load(outDirec+"/psi_t"+str(i)+".npy")[i1:-i2,j1:-j2]
	psi=np.fft.fft(psi_t-np.mean(psi_t,axis=0),axis=0)
	ws=np.fft.fftfreq(n=len(psi_t))
	print(dt)
	ws/=dt # convert to THz: e.g. .002 picosecond timesteps, every 10th timestep logged
	if not os.path.exists(outDirec+"/ws.npy"):
		np.save(outDirec+"/ws.npy",ws)
	outname=outDirec+"/ivib.npy"
	np.save(outname,psi)


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
		# you can pass a comma-delimited list of functions. e.g. "python3 abEELS main_JACR,diffraction,sliceE inputs/inputFile.txt",
		funs=sys.argv[1].split(",") 
		for fun in funs:
			if fun in funcAliases:
				fun = funcAliases[ fun ]
			command=fun+"()"
			exec(command)