import numpy as np
import glob,os
from tqdm import tqdm

def scrapePos(filename):
	lines=open(filename,'r').readlines()
	for i,l in enumerate(lines):
		numCols=len(l.split(" "))
		hasNonNumChars=( False in [ c in "0123456789.e- \n" for c in l ])
		if hasNonNumChars:
			continue
		if numCols==6 or numCols==5:
			break
	#print("scrapePos: data starts on line",i)
	data=np.loadtxt(filename,delimiter=" ",skiprows=i)
	#print(data)
	if numCols==5:
		pos=data[:,2:] ; types=data[:,1].astype(int)
	else:
		pos=data[:,3:] ; types=data[:,2].astype(int)
	return pos,types

def bboxFromPos(filename):
	lines=open(filename,'r').readlines()
	bbox=[]
	for l in lines:
		if "xlo xhi" in l or "ylo yhi" in l or "zlo zhi" in l:
			l=l.split()
			bbox.append([float(l[0]),float(l[1])])
		if len(bbox)==3:
			return np.asarray(bbox)

def scrapeLayers(filename):		# # Chunk-averaged data for fix flux_m2 and group all	# comments
	ts=[] ; Ts=[] ; X=[]		# # Timestep Number-of-chunks Total-count
	f=open(filename, 'r')		# # Chunk Coord1 Ncount temp
	while True:			# 16020050 240 12000					# line containing timestep number
		l=f.readline()		#   1 0.679661 50 0
		if not l: 	# EOF	#   2 2.03898 50 0
			break		#   3 3.39831 76 -0.00920895
		if l[0]=="#": # comment	#   4 4.75763 48 -0.00131646
			continue	#   5 6.11695 46 0.00744407
		elif l[0]!=" ":	# step	#   6 7.47627 53 -0.0087254
			l=l.split()
			ts.append(int(l[0])) ; Ts.append([])
		else:			# line contains: layer ID, position, Natoms/chunk , Temperature
			l=l.split()
			if len(l)!=4:
				continue
			Ts[-1].append(float(l[3]))
			if len(Ts)==1:
				X.append(float(l[1]))
	if len(Ts[-1])!=len(Ts[0]): # partial line, incomplete run
		Ts=Ts[:-1] ; ts=ts[:-1]
	print("(numpy-izing)")
	return np.asarray(Ts),np.asarray(X),np.asarray(ts)

def scrapeOutput(filename,columnNames=""):
	# if we were given the columns the user wants, then we can search for processed output files. if the user didn't specify, we have no way(?) of knowing what columns exist in the output file (across all thermo/runs!) without actually scanning through it (some subset of columns may have been exported previously, so we can't just infer from what columns were exported). so we *must* scan through all. so only check for exported files if columnNames was passed
	fname=filename.split("/")[-1]
	if len(columnNames)!=0:
		# look to see how many of each column's files were exported
		exported=[ glob.glob(filename.replace(fname,"out_"+n+"_*.txt")) for n in columnNames ]
		lens=[ len(e) for e in exported ]
		# if all have have at least 1 file exported...
		if 0 not in lens:
			exported=sum(exported,[]) # flatten list-of-lists-of-filenames (one list per columnName) into list-of-filenames
			roundIDs=[ int(fi.split("_")[-1].replace(".txt","")) for fi in exported ]
			data=[]
			for i in range(min(roundIDs)-1,max(roundIDs)):
				data.append({})		# same datastructure as if we were reading in the output file fresh: one dict per thermo/run
				for n in columnNames:
					# OOPS, THIS FAILS IF YOU CALL FROM ANOTHER DIRECTORY, AND FILES IN 'exported' HAVE DIRECTORY PREPENDED
					#if "out_"+n+"_"+str(i+1)+".txt" in exported: # if the file exists, include it in the dict
					#	data[-1][n]=[]
					#	out=np.loadtxt("out_"+n+"_"+str(i+1)+".txt")
					#	data[-1]["Step"]=out[:,0]
					#	data[-1][n]=out[:,1]
					for f in exported:
						if "out_"+n+"_"+str(i+1)+".txt" in f:
							data[-1][n]=[]
							out=np.loadtxt(f)
							data[-1]["Step"]=out[:,0]
							data[-1][n]=out[:,1]

			return data

	f=open(filename,'r') ; readingData=False ; data=[]
	while True:
		line=f.readline() # use this instead of lines=f.readlines() because outfile may be huge
		if not line:
			break
		line=line.strip()
		#print(line)
		# out file has lots of junk, with timestep data between a header (with column labels, first of which is "Step"), and a footer (reporting a loop time)

		# Each thermo/run results in a section, with a columns header such as "Step \t Temp \t Press ...", so if Step is present, it's the start of a new section. at the end, you'll have a line saying "Loop time of nnnn on nnnn procs for nnnn steps with nnnn atoms"
		if line[:4]=="Step":
			data.append({}) # new dict for new series
			names=line.split()
			readingData=True
			for n in names: #prepare columns for runSections[whichSection][valueType]
				if n in columnNames or len(columnNames)==0 or n=="Step":
					data[-1][n]=[] # empty list per each column name
		elif line[:4]=="Loop":
			readingData=False
		elif readingData:
			line=line.split()
			if len(line)!=len(names): # if a simulation is still running, it's plausible a partial line was logged to file
				print("badLen")
				continue
			for n,v in zip(names,line):
				if n in columnNames or len(columnNames)==0 or n=="Step":
					data[-1][n].append(float(v))
	for i in range(len(data)):
		for n in data[i].keys():
			if n=="Step":
				continue
			out=np.zeros((len(data[i][n]),2))
			out[:,0]+=data[i]["Step"] ; out[:,1]+=data[i][n]
			np.savetxt(filename.replace(fname,"out_"+n+"_"+str(i+1)+".txt"),out)
	return data

def scrapeDump(filename,trange=":",arange="::1"):
	if os.path.exists(filename+"_t.npy"):
		print("scrapeDump previous exported pos,ts: rereading them in from .npy files")
		ts=np.load(filename+"_t.npy") ; pos=np.load(filename+"_p.npy")
		return pos,ts
	print("scraping output file", filename)
	f=open(filename, 'r')
	timesteps=[] ; lineIDs=[] ; i=-1 ; nP=0 ; maxt=trange.split(":")[1]
	print("running through file once, counting timesteps")
	while True:
		if len(maxt)>0 and len(timesteps)>int(maxt):
			break
		line=f.readline() ; i=i+1 #read lines one at a time, instead of loading entire (potentially huge) file into RAM.
		if ( not line ): # or ( ( "TIMESTEP" in line.upper() ) and ( len(pos)>=maxsteps ) ):
			break
		if "Timestep" in line: #regular dump file contains single line: "Timestep: t"
			t=int(line.split()[2])
		elif "TIMESTEP" in line: #custom dump file contains 2 lines: "ITEM: TIMESTEP\n t"
			t=int(f.readline()) ; i=i+1
		if "TIMESTEP" in line.upper():
			timesteps.append(t)
			lineIDs.append([0,0])
			#lineIDs[-1][0]=i+1
		elif len(line.split())<3 or "ITEM" in line:
			continue
		else:
			if nP==0:
				nP=len(line.split())
			if lineIDs[-1][0]==0:
				lineIDs[-1][0]=i #first line with data
			lineIDs[-1][1]=i #potentially last line with data on it
	if lineIDs[-1][0]==0: #sometimes there's a nefarious empty "TIMESTEP" label in the qwdump files? idk why
		del lineIDs[-1] ; del timesteps[-1]
	print("found "+str(len(timesteps))+" timesteps, "+str(lineIDs[0][1]-lineIDs[0][0]+1)+" atoms, with "+str(nP)+" datapoints each")
	timesteps=eval("timesteps["+trange+"]") ; lineIDs=eval("lineIDs["+trange+"]")
	#timesteps=timesteps[-lastN:] ; lineIDs=lineIDs[-lastN:]
	#timesteps=timesteps[:maxsteps] ; lineIDs=lineIDs[:maxsteps]
	nt=len(timesteps) ; nA=lineIDs[0][1]-lineIDs[0][0]+1

	As=np.arange(nA) ; As=eval("As["+arange+"]") ; nA=len(As)
	minA=min(As) ; maxA=max(As) ; nthA=As[1]-As[0]

	pos=np.zeros((nt,nA,nP)) #; ids=np.zeros(nA) ; types=np.zeros(nA)
	l=-1
	f.seek(0)

	for t,idpair in enumerate(tqdm(lineIDs)): #for each timestep
		while l<idpair[0]-1: #fast forward to next line with data on it
			line=f.readline() ; l=l+1
		a=-1
		while l<idpair[1]: #and then scroll through all lines with data on them, and record
			line=f.readline() ; l=l+1 ; a=a+1
			#if a not in As: 
			if a<minA or a>maxA or a%nthA!=minA: # much much much faster than "if a not in As"
				continue
			#print(line,l,a,idpair)
			pos[t,As==a,:]=list(map(float,line.split()))[:]
	np.save(filename+"_t.npy",timesteps) ; np.save(filename+"_p.npy",pos)
	return pos,np.asarray(timesteps) # [nS,nA,xyz]

def qdump(filename,timescaling=1,convert=True,safemode=False): # OBSCENELY FAST IN COMPARISON TO scrapeDump()
	if os.path.exists(filename+"_ts.npy"):
		print("ignoring qdump, reading npy files instead")
		ts=np.load(filename+"_ts.npy")
		pos=np.load(filename+"_pos.npy")
		vel=np.load(filename+"_vel.npy")
		types=np.load(filename+"_typ.npy")
		return pos,vel,ts,types

	from ovito.io import import_file # TODO WEIRD BUG, AFTER THIS RUNS, WE CAN'T PLOT STUFF WITH NICEPLOT. WE GET THE FOLLOWING ERROR: ImportError: Cannot load backend 'TkAgg' which requires the 'tk' interactive framework, as 'qt' is currently running  WHY TF DOES OVITO LOAD qt AND HOW DO WE UNLOAD IT??
	# TESTING: 
	# ase.io.read can also read qdumps, BUT, it takes significantly longer: 
	# start=time.time() ; loaded=qdump("projects/Si_SED_09b/NVE.qdump"); print(time.time()-start) # takes 6s
	# start=time.time(); loaded=aseread("NVE.qdump",index=":") ; print(time.time()-start) # takes 39s
	print("reading qdump")
	pipeline = import_file(filename)
	nt=pipeline.source.num_frames
	data=pipeline.compute(0)
	na,nxyz=np.shape(data.particles.positions.array)
	pos=np.zeros((nt,na,3))
	vel=np.zeros((nt,na,3))
	types=data.particles.particle_type.array
	ts=np.arange(nt)*timescaling
	for n in tqdm(range(nt)):
		if safemode:
			try:
				data=pipeline.compute(n)
			except:
				print("safemode == True. failure on timestep",n)
				continue
		else:
			data=pipeline.compute(n)
		pos[n,:,:]=data.particles.position.array
		vel[n,:,:]=data.particles.velocities.array
	if convert:
		print("ts,pos,vel,typ",ts.shape,pos.shape,vel.shape,types.shape)
		np.save(filename+"_ts.npy",ts)
		np.save(filename+"_pos.npy",pos)
		np.save(filename+"_vel.npy",vel)
		np.save(filename+"_typ.npy",types)
	return pos,vel,ts,types

#where=np.where(pos[:,:,2]<.25*c)
#pos[where,2]+=lz

def avgPos(pos,sx,sy,sz,alpha=90,beta=90,gamma=90): # takes "pos" as from: pos,timesteps=scrapeDump(trange="-"+str(avgOver)+":"), [nS,nA,xyz]
	nS,nA,na=np.shape(pos)
	displacements=np.zeros((nS,nA,na))
	for t in tqdm(range(nS)):
		# distance between initial position and position at time t (for each atom, along each axis), including wrapping
		displacements[t,:,:]=dxyz(pos[0,:,:],pos[t,:,:],sx,sy,sz,alpha,beta,gamma) 
	# average position = initial position + average of all displacements away from initial position
	# time-dependent displacements *from that average position* also requires subtracting mean(displacements)
	return pos[0,:,:]+np.mean(displacements[:,:,:],axis=0),displacements-np.mean(displacements[:,:,:],axis=0)

def getWrapAndRange(size,axis):
	wrap=np.zeros(3)
	if size is None:
		wrap[axis]=1 ; r=[0]
	else:
		wrap[axis]=size ; r=[-1,0,1]
	return wrap,r

# TODO currently we only handle gamma unskewing, BUT, if we worked in fractional coordinates (unskew before calculating, and i'm sure code exists to calculate the transformation matrix based on all alpha beta gamma) then we wouldn't need skewing in the shifting. 
def dxyz(pos1,pos2,sx,sy,sz,alpha=90,beta=90,gamma=90): # given two snapshots of positions [[xa,ya,za],[xb,yb,zb],...] x2, don't just do dxyz=xyz1-xyz2. must include wrapping!
	dxyz_0=pos2-pos1
	# we use these for wrapping
	wx,i_range=getWrapAndRange(sx,0) ; wy,j_range=getWrapAndRange(sy,1) ; wz,k_range=getWrapAndRange(sz,2)
	
	# for pos_1 in the 27 surrounding positions (original, and 26 neighbors), keep only the smallest (absolute) distance found
	for i in i_range:
		for j in j_range:
			for k in k_range:

				# e.g. [.1,.2,-.1] + 1*[25,0,0]+0*[0,10,0]+0*[0,0,10] # to wrap +x for a hypothetical 25x10x10 simulation volume
				shift_xyz=i*wx+j*wy+k*wz
				if gamma!=90: # TODO WE SHOULD BE ABLE TO HANDLE NON-90 ALPHA AND BETA TOO
					skew=np.eye(3) ; skew[0,1]=-np.sin(gamma*np.pi/180-np.pi/2) ; skew[1,1]=np.cos(gamma*np.pi/180-np.pi/2)
					shift_xyz=np.matmul(skew,shift_xyz)
				#print(i,j,k,shift_xyz)

				dxyz_w=pos2+shift_xyz-pos1 # if an atom crossed the x axis (from +x to -x ie L) it'll register as closer if we take (x0+L)-xf
				dxyz_0=absMin(dxyz_0,dxyz_w)
	return dxyz_0

def absMin(dxyz_a,dxyz_b): # use this for getting absolute distances with wrapping. Nx3 [[1dx,1dy,1dz],[2dx,2dy,2dz],...] vs [[1dx,1dy,1dz],[2dx,2dy,2dz],...] with different wrapping, we want to keep the closest (not max, not min)
	abs1=np.absolute(dxyz_a) # absolute distances (still in x,y,z separately) for first comparison
	abs2=np.absolute(dxyz_b) # and second comparison (Wrapped)
	minabs=np.minimum(abs1,abs2) # lowest distances, between two comparisons (wrapped and unwrapped)
	keeps=np.zeros((len(dxyz_a),3)) # next, we'll "select" distances from the approriate dxyz*, including sign, by comparing minabs (lowests) vs each comparison's
	keeps[abs1==minabs]=dxyz_a[abs1==minabs]
	keeps[abs2==minabs]=dxyz_b[abs2==minabs]
	return keeps

# Spectral Energy Density: phonon dispersions!
# avg - average positions [a,xyz] (import using scrapeDump or qdump. average via avgPos)
# velocities - time-dependent atom velocities [t,a,xyz] (as imported via scrapeDump or qdump)
# p_xyz - 0,1,2 indicating if we should track positions in x,y or z (this is your wave-vector direction)
# v_xyz - like v_xyz, but for which velocities to track. L vs T modes
# a - this is your specified periodicity (or lattice constant for crystals)
# nk - resolution in k-space. note your resolution in ω is inherited from ts
# bs - optional: should be a list of atom indices to include. this allows the caller to sum over crystal cell coordinates (see discussion on Σb below)
# TODO: currently k_max=π/a. this is convention. so if you want your x axis to be wavelength⁻¹, you need to divide by π? should we do this for you? idk
# TODO: ditto for ω, which is rad/timestep. you need to scale it accordingly (timesteps to time units) and include 2π to get to Hz vs rad/s
def SED(avg,velocities,p_xyz,v_xyz,a,nk=100,bs='',perAtom=False,ks='',keepComplex=False):
	nt,na,nax=np.shape(velocities)
	if len(bs)==0:
		bs=np.arange(na)
	else:
		na=len(bs)
	nt2=int(nt/2) #; nt2=nt # this is used to trim off negative frequencies
	if len(ks)==0:
		ks=np.linspace(0,np.pi/a,nk)
	else:
		nk=len(ks)

	ws=np.fft.fftfreq(nt)[:nt2] ; Zs=np.zeros((nt2,nk))
	if keepComplex:
		Zs=np.zeros((nt2,nk),dtype=complex)

	# Φ(k,ω) = Σb | ∫ Σn u°(n,b,t) exp( i k r̄(n,0) - i ω t ) dt |² # https://github.com/tyst3273/phonon-sed/blob/master/manual.pdf
	# b is index *within* unit cell, n is index *of* unit cell. pairs of n,b
	# can be thought of as indices pointing to a specific atom.
	# u° is the velocity of each atom as a function of time. r̄(n,b=0) is the
	# equilibrium position of the unit cell. (if we assume atom 0 in a unit
	# cell is at the origin, we can use x̄(n,b=0)). looping over n inside the 
	# integral, but not b, means we are effectively picking up "every other 
	# atom", which means short-wavelength optical modes will register as 
	# their BZ-folded longer-wavelength selves*. using r̄(n,b=0) rather than
	# x̄(n,b) means a phase-shift is applied for b≠0 atoms. this means if we
	# ignore b,n ("...Σb | ∫ Σn...") and sum across all atoms instead
	# ("... | ∫ Σi..."), we lose BZ folding. if we don't care (fine, so long
	# as a is small enough / k is large is small enough to "unfold" the 
	# dispersion), we simplify summing, and can use x̄ instead. we thus no
	# longer require a perfect crystal to be analyzed. the above equation
	# also simplified to:
	# Φ(k,ω) = | ∫ Σn u°(n,t) exp( i k x̄(n) - i ω t ) dt |²
	# and noting the property: e^(A+B)=e^(A)*e^(B)
	# Φ(k,ω) = | ∫ Σn u°(n,t) exp( i k x̄(n) ) * exp( - i ω t ) dt |²
	# and noting that the definition of a fourier transform:
	# F(w) = ∫ f(t) * exp( -i 2 π ω t ) dt
	# we can reformulate the above eqaution as:
	# f(t) = u°(n,t) * exp( i k x )
	# Φ(k,ω) = | FFT{ Σn f(t) } |²
	# of course, this code still *can* analyze crystals with folding: the 
	# user should simply call this function multiple times, passing the "bs"
	# argument with a list of atom indices for us to use

	# TODO NEED VALIDATION OF THESE. CHECK OUT /media/Alexandria/U Virginia/Research/MD/projects/nanotubeBN/ AND COMPARE TO hBN_07
	if isinstance(p_xyz,str): # "theta" and "radius" are both allowed! we will do polar coordinates! (psuedo-angular momentum)
		if p_xyz=="theta":
			xs=np.arctan2(avg[bs,1],avg[bs,0])
			ks=np.linspace(0,a,nk)	# IF THETA, "a" SHOULD BE NUMBER OF UNIT CELLS AROUND THE CIRCUMFERENCE (previously, system is length L, divided by n, where a=L/n, and ks goes to 2π/a. here, θ goes 0-2π (this is L), so π/(2π/n) --> n
		elif p_xyz=="radius":
			xs=np.sqrt(avg[bs,1]**2+avg[bs,0]**2)
		else:
			print("ERROR, UNRECOGNIZED STRING PASSED FOR p_xyz",p_xyz,"(try \"theta\" or \"radius\", an axis index, or a vector)")
	elif isinstance(p_xyz,(int,float)): # 0,1,2 --> x,y,z
		xs=avg[bs,p_xyz] # a,xyz --> a
	else:	# [1,0,0],[1,1,0],[1,1,1] and so on
		# https://math.stackexchange.com/questions/1679701/components-of-velocity-in-the-direction-of-a-vector-i-3j2k
		# project a vector A [i,j,k] on vector B [I,J,K], simply do: A•B/|B| (dot, mag)
		# for at in range(na): x=np.dot(avg[at,:],p_xyz)/np.linalg.norm(p_xyz)
		# OR, use np.einsum. dots=np.einsum('ij, ij->i',listOfVecsA,listOfVecsB)
		p_xyz=np.asarray(p_xyz)
		d=p_xyz[None,:]*np.ones((na,3)) # xyz --> a,xyz
		xs=np.einsum('ij, ij->i',avg[bs,:],d) # pos • vec, all at once
		xs/=np.linalg.norm(p_xyz)

	# TODO mongo ram usage for rotation step vs simply using a reference to the existing matrix, if velocities is huge, e.g. simulations with a 100000 atoms, e.g. 50x50x5 UC silicon (8 atoms per UC), as required for 110 SED
	if isinstance(v_xyz,str): # "theta" and "radius" are both allowed! we will do polar coordinates! (psuedo-angular momentum)
		if v_xyz=="theta":
			vs=np.arctan2(velocities[:,bs,1],velocities[:,bs,0])
		elif v_xyz=="radius":
			vs=np.sqrt(velocities[:,bs,1]**2+velocities[:,bs,0]**2)
		else:
			print("ERROR, UNRECOGNIZED STRING PASSED FOR p_xyz",p_xyz,"(try \"theta\" or \"radius\", an axis index, or a vector)")
	elif isinstance(v_xyz,(int,float)):
		vs=velocities[:,bs,v_xyz] # t,a,xyz --> t,a
	else: 
		# for handling velocities, there's just one more step from above: "flattening" first two axes t,a,xyz --> t*a,x,y,z
		vflat=np.reshape(velocities[:,bs,:],(nt*na,3)) # t,a,xyz --> t*a,xyz
		v_xyz=np.asarray(v_xyz)
		d=v_xyz[None,:]*np.ones((nt*na,3))
		vs=np.einsum('ij, ij->i',vflat,d)
		vs=np.reshape(vs,(nt,na)) # and unflattening at the end: t*a,xyz --> t,a,xyz
		vs/=np.linalg.norm(v_xyz)

	if perAtom:
		Zs=np.zeros((nt2,nk,na),dtype=complex)
		for j,k in enumerate(tqdm(ks)):	
			# f(t) = u°(a,t) * exp( i k x̄ )
			F=vs[:,:]*np.exp(1j*k*xs[None,:]) # t,a
			# Σn u°(a,t) * exp( i k x̄ )
			# F=np.sum(F,axis=1)/na # t,a --> t
			# ∫ { Σn u°(a,t) exp( i k x) } * exp( - i ω t ) dt AKA FFT{ Σn u°(a,t) exp( i k x) }
			integrated=np.fft.fft(F,axis=0)[:nt2,:] # t,a --> ω,t. trim off negative ω
			# | ∫ Σn u°(a,t) exp( i k x) * exp( - i ω t ) dt |²
			Zs[:,j,:]+=integrated # np.absolute(integrated)**2
		return Zs,ks,ws

	for j,k in enumerate(tqdm(ks)):
		# f(t) = u°(a,t) * exp( i k x̄ )
		F=vs[:,:]*np.exp(1j*k*xs[None,:]) # t,a
		# Σn u°(a,t) * exp( i k x̄ )
		F=np.sum(F,axis=1)/na # t,a --> t
		# ∫ { Σn u°(a,t) exp( i k x) } * exp( - i ω t ) dt AKA FFT{ Σn u°(a,t) exp( i k x) }
		integrated=np.fft.fft(F)[:nt2] # t --> ω. trim off negative ω
		# | ∫ Σn u°(a,t) exp( i k x) * exp( - i ω t ) dt |²
		if keepComplex:
			Zs[:,j]+=integrated
		else:
			Zs[:,j]+=np.absolute(integrated)**2
	return Zs,ks,ws

def addDumpColumn(dumpfile,columnvals,outfile):
	f1=open(dumpfile,'r')
	f2=open(outfile,'w')
	ct=-1
	while True:
		line=f1.readline() #read lines one at a time, instead of loading entire (potentially huge) file into RAM. 
		#print(line,ct)
		if not line: #EOF
			break
		if "Timestep" in line: #TIMESTEP LINE
			print(line)
			ct=-1
		elif len(line.split())>=4: #LINES WITH DATA FOR ATOMS
			ct=ct+1
			vals=columnvals[ct]
			line=line.replace('\n','\t')
			if not isinstance(vals,(float,int)):
				vals='\t'.join([ str(v) for v in vals ])
			line=line+'\t'+str(vals)+'\n'
		f2.write(line)

# save positions (na,xyz) to a positions file. useful for visualizing avg and stuff
def outToPositionsFile(filename,pos,types,masses,a,b,c,alpha=90,beta=90,gamma=90,fractional=False):
	alpha,beta,gamma = [ v*np.pi/180 for v in [ alpha,beta,gamma ] ]
	# https://docs.lammps.org/Howto_triclinic.html
	lx = a                   ;   xy = b*np.cos(gamma)            ; xz = c*np.cos(beta)
	ly = np.sqrt(b**2-xy**2) ;   yz = (b*c*np.cos(alpha)-xy*xz)/ly ; lz = np.sqrt(c**2-xz**2-yz**2)
	if fractional: # USER MAY ELECT TO PASS POSITIONS IN FRACTIONAL COORDINATES (NO SKEWNING APPLIED FOR ALPHA/BETA/GAMMA, SO WE SHOULD DO IT FOR THEM
		print("fractional conversion:",pos[-1])
		M=np.asarray([[lx,xy,xz],[0,ly,yz],[0,0,lz]])
		pos=[ np.matmul(M,p) for p in pos ]
		print("fractional conversion:",pos[-1])
	import datetime
	now=datetime.datetime.now()
	lines=["########### lammpsScapers.py > outToPositionsFile() "+now.strftime("%Y/%m/%d %H:%M:%S")+" ###########"]
	lines=lines+[str(len(pos))+" atoms","",str(len(masses))+" atom types",""]
	for s,xyz in zip([lx,ly,lz],["x","y","z"]):
		lines.append("0.0 "+str(s)+" "+xyz+"lo "+xyz+"hi")
	if np.amax(np.absolute([xy,xz,yz]))>1e-5:
		lines.append(str(xy)+" "+str(xz)+" "+str(yz)+" xy xz yz")
	lines=lines+["","Masses",""]+[ str(i+1)+" "+str(m) for i,m in enumerate(masses) ]+["","Atoms",""]
	for i,(t,xyz) in enumerate(zip(types,pos)):
		t=int(t) ; atxyz=[ str(v) for v in [i+1,1,t,*xyz] ]
		lines.append(" ".join(atxyz))
		#lines[-1]=lines[-1].replace(" -0.0 "," 0.0 ") 
	with open(filename,'w') as f:
		for l in lines:
			f.write(l+"\n")

# positions (nt,na,nxyz) is written to a psuedo-dump file. useful for visualizing avg and stuff
def outToQdump(filename,pos,types,sx,sy,sz):
	import datetime
	now=datetime.datetime.now()
	lines=[]#"########### lammpsScapers.py > outToPositionsFile() "+now.strftime("%Y/%m/%d %H:%M:%S")+" ###########"]
	nt,na,nxyz=np.shape(pos)
	for t in tqdm(range(nt)):
		lines=lines+["ITEM: TIMESTEP",str(t+1),"ITEM: NUMBER OF ATOMS",str(na),"ITEM: BOX BOUNDS pp pp pp"]
		for s in [sx,sy,sz]:
			lines.append("0 "+str(s))
		lines.append("ITEM: ATOMS id type x y z")
		for a in range(na):
			l=[a+1,types[a],*pos[t,a,:]]
			lines.append(" ".join([str(v) for v in l ]))
	with open(filename,'w') as f:
		for l in lines:
			f.write(l+"\n")

# pos should be average atomic positions. you'll need to call avgPos yourself
# neighborLocations should be coordinates where you expect to find neighbors
# sx,sy,sz is used for wrapping for neighbor-finding. "None" is acceptable to disable wrapping in that axis
def procrustes(positions,sx,sy,sz,neighborLocations,rotation=True,scaling="aniso"):
	import itertools
	nA=len(positions) ; nn=len(neighborLocations)
	pcds,rots,scxs,scys,sczs,scis=np.zeros(nA),np.zeros(nA),np.zeros(nA),np.zeros(nA),np.zeros(nA),np.zeros(nA)
	RMS=np.sqrt(np.sum(neighborLocations**2)) # d1=√(dx1²+dy1²+dz1²), RMS=√(d1²+d2²+d3²+d4²)=√(Σ(dnj))
	RMS1D=[ np.sqrt(np.sum(neighborLocations[:,i]**2)) for i in range(3) ]
	print("calcating procrustes, looping through atoms")
	for a,xyz in enumerate(tqdm(positions)): # for each atom, find 4 neighbors
		xyz=np.ones(nA)[:,None]*xyz[None,:] # Nx3 for easy calculation of vectors to every other atom
		neighborVecs=dxyz(xyz,positions,sx,sy,sz) # x,y,z distance to every other atom, including wrapping
		distances=np.sum(neighborVecs**2,axis=1)**.5 # absolute distance to every other atom
		# use argsort: https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
		neighbors=np.argsort(distances)[1:nn+1] # N closest real neighbors' indices (excluding index 0 in argsort, since that's "this atom")
		neighlocs=neighborVecs[neighbors] # relative coordinates of neighbor, real units
		permutations=list(itertools.permutations(range(nn))) # permutations of indices [[0,1,2,3],[0,1,3,2],[0,2,1,3],...]
		pcd=[] ; rot=[] ; scx=[] ; scy=[] ; scz=[] ; sci=[]
		# cycle through all combinations of "neighbors 1,2,3 compared to lattice sites a,b,c" or "neighbor 1,2,3 compared to lattice sites a,c,b". this just "shuffles" the neighbor locations: since any order might be compared to the perfect lattice
		for i,perm in enumerate(permutations):
			#for each rotation, iso scale, anisoscale, the 3 steps are: "compute it" "apply it" "log it"
			#ROTATION: https://en.wikipedia.org/wiki/Kabsch_algorithm and https://github.com/charnley/rmsd/blob/master/rmsd/calculate_rmsd.py
			P=neighlocs[perm,:]
			Q=neighborLocations ; R=np.identity(3) # "attempting rotation of P onto Q"
			H=np.dot(np.transpose(P),Q) #"covariance matrix"
			U,S,V=np.linalg.svd(H)
			if (np.linalg.det(U) * np.linalg.det(V)) < 0.0:
				S[-1]=-S[-1]
				U[:,-1]=-U[:,-1]
			R=np.dot(U,V)
			if rotation:
				P=np.dot(P,R)
			rot.append( np.arccos((np.trace(R)-1)/2)*180/np.pi%30 ) #https://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_and_to_axis%E2%80%93angle
			rot[-1]=min(abs(rot[-1]-30),rot[-1])

			#SCALING: RMS distance from each point to the origin, set to 1. (total distance for isotropic scaling, or x,y,z separately and 3 scaling factors, for aniso)
			RMSdist=np.sqrt(np.sum(P**2)) ; RMSx=np.sqrt(np.sum(P[:,0]**2)) ; RMSy=np.sqrt(np.sum(P[:,1]**2)) ; RMSz=np.sqrt(np.sum(P[:,2]**2))
			if scaling=="iso":
				P*=RMS/RMSdist
			if scaling=="aniso":
				P[:,0]*=RMS1D[0]/RMSx ; P[:,1]*=RMS1D[1]/RMSy ; P[:,2]*=RMS1D[2]/RMSz
			scx.append(RMS/RMSx) ; scy.append(RMS/RMSy) ; scz.append(RMS/RMSz) ; sci.append(RMS/RMSdist)
			# CALCULATE PROCRUSTES
			pcd.append( min(np.sum((P-Q)**2),np.sum((P+Q)**2)) ) #use both ±neighsLattice to compare against (in case no rotation was performed and it's the mirrored atom we're examining)
		p=np.argmin(pcd)
		pcds[a]=pcd[p]
		rots[a]=rot[p]
		scxs[a]=scx[p]
		scys[a]=scy[p]
		sczs[a]=scz[p]
		scis[a]=sci[p]
	return pcds,rots,scxs,scys,sczs,scis
 
def binning(array,n,axis=0):
	shape=np.asarray(np.shape(array),dtype=int) ; shape[axis]//=n	# scale appropriate axis (e.g. down by 10
	new=np.zeros(shape)						# create empty new binned array
	print(shape)
	for i in range(n):	# "starting at 0, every 10th. then starting at 1, every 10th" and so on. trailing values ignored
		if axis==0:
			new+=array[i:i+n*new.shape[0]:n]
		else:
			new+=array[:,i::n]
	new/=n
	return new

# returns a function which can be passed a triplet of three atoms' positions, [ijk,xyz], and return the potential energy
def potentialFunc_SW(swfile):
	r_cut,potential2Body,potential3Body=SW(swfile)
	print("r_cut",r_cut)
	def potential(pos):
		vij=pos[1,:]-pos[0,:]
		vik=pos[2,:]-pos[0,:]
		rij=lVec(vij) ; rik=lVec(vik)
		if rij>r_cut or rik>r_cut:
			return 0
		tijk=angBetween(vij,vik)
		return potential3Body(rij,rik,tijk)+potential2Body(rij)
	return potential

#E=ΣᵢΣⱼϕ₂(rᵢⱼ)+ΣᵢΣⱼΣₖϕ₃(rᵢⱼ,rᵢₖ,θᵢⱼₖ)
#ϕ₂ is 2-body component, ϕ₂(rᵢⱼ)=A ϵ *[ B*(σ/r)^p-(σ/r)^q ] * exp( σ / r-a*σ )
#ϕ₃ is 3-body component, ϕ₃(rᵢⱼ,rᵢₖ,θᵢⱼₖ)=λ ϵ (cos(θᵢⱼₖ)-cos₀)² exp( γᵢⱼσᵢⱼ / r-aᵢⱼ*σᵢⱼ ) exp( γᵢₖσᵢₖ / r-aᵢₖ*σᵢₖ )
#calculate potential energy E for each atom in A as a result of its interaction with each atom in B.
def SW(swfile):
	sw=readSW(swfile)
	e=sw["e"];s=sw["s"];a=sw["a"];l=sw["l"];g=sw["g"];c=sw["c"];A=sw["A"];B=sw["B"];p=sw["p"];q=sw["q"];t=sw["t"]
	r_cut=a*s*.99 #BEWARE: check this if you change sw potentials! use sw2LJ()
	def potential2Body(rij):
		return A*e*(B*(s/rij)**p-(s/rij)**q)*np.exp(s/(rij-a*s))
	def potential3Body(rij,rik,tijk):
		return l*e*(np.cos(tijk)-c)**2*np.exp(g*s/(rij-a*s))*np.exp(g*s/(rik-a*s))
	return r_cut,potential2Body,potential3Body

def readSW(swfile):
	f=open(swfile,'r',errors='ignore') #some versions of python choke on readlines() if there are unicode characters in the file being read (eg, umlauts because some germans developed your SW potential)
	entries={}
	lines=f.readlines()
	for l in lines:
		if len(l)<1 or l[0]=="#" or len(l.split())<14:
			continue
		#print(l)			#                  0       1     2 3      4     5         6 7 8 9 10
		l=list(map(float,l.split()[3:]))	#elem1,elem2,elem3,epsilon,sigma,a,lambda,gamma,costheta0,A,B,p,q,tol
		names="e,s,a,l,g,c,A,B,p,q,t"
		for n,v in zip(names.split(','),l):
			entries[n]=float(v)
		return entries	

def lVec(v1):
	return np.sqrt( np.sum( v1**2 ) )
def angBetween(v1,v2): # cos(θ)=(vᵢⱼ•vᵢₖ)/(|vᵢⱼ|*|vᵢₖ|)
	return np.arccos(np.dot(v1,v2)/(lVec(v1)*lVec(v2))) #cos(θ)=(vᵢⱼ•vᵢₖ)/(|vᵢⱼ|*|vᵢₖ|)

#combining calculated Fx(t), Fy(t), Fz(t) with exported Vx(t), Vy(t), Vz(t): Q_LA(ω)=ΣᵢFxᵢ(ω)*Vxᵢ(ω), Q_TA(ω)=ΣᵢFyᵢ(ω)*Vyᵢ(ω), Q_TA(ω)=ΣᵢFzᵢ(ω)*Vzᵢ(
# Qᴬᴮ(t)=ΣᵢΣⱼ⟨ dUᵢ/drᵢⱼ * vᵢ - dUⱼ/dⱼᵢ * vⱼ ⟩ PRB95,144309 eq 27, Q(ω)=FFT(Q(t)) (post-summing), Work = Force * Velocity, "work on A by B, minus work on B by A", note cross-correlation ⟨-⟩, since FFT(f(t))*FFT(g(t))≠FFT(f(t)*g(t)), but rather, FFT(⟨f(t),g(t)⟩)
# need pairwise forces: Force=dEnergy/dx (change in energy as you move an atom, so compute E, perturb atom 'm' by dx, recompute E, Fₘ=(Eoₘ-Eₘ)/dx for force on 'm'
# two body term: force between 'm' and 'p' are pairwise, ie, Fₘₚ₂=-Fₚₘ₂, "force on m applied by p" vs "force on p applied by m"
# three body term: we can compute "force on m", "force on p", "force on u", but the potential does not prescribe which forces came from where. "how much of the 
#    force on p came from m vs came from u?", we aren't told. but, this means we are allowed to choose. so we can just say that the central atom acts on both 
#    outer atoms, and there is no interaction between the two outer atoms. This means "force on m" is "force on m applied by p" and "force on u" is "force on u 
#    applied by p". This then allows a return to pairwise "equal and opposite" interactions, ie, Fₘₚ₃=-Fₚₘ₃, Fᵤₚ₃=-Fₚᵤ₃
# finally, given that the potential is additive (sum these terms across all sets of atoms), so is heat flux. for each set of atoms, we can simply add up each 
#    Qₘₚ=Fₘₚ*Vₘ-Fₚₘ*Vₚ
#Fx=dE/dx, Fy=dE/dy, Fz=dE/dz



	# need pairwise forces: Force=dEnergy/dx (change in energy as you move an atom, so compute E, perturb atom 'm' by dx, recompute E, Fₘ=(Eoₘ-Eₘ)/dx for force on 'm'
	# two body term: force between 'm' and 'p' are pairwise, ie, Fₘₚ₂=-Fₚₘ₂, "force on m applied by p" vs "force on p applied by m"


# Power = Force * velocity , Fₓ=dU/dx. for two atoms interacting: Qᵢⱼ=dUᵢ/drᵢⱼ *  vᵢ - dUⱼ/dⱼᵢ * vⱼ , "net work: power on i by j minus power on j by i"
# In the time domain: Qᴬᴮ(t)=ΣᵢΣⱼ( dUᵢ(t)/drᵢⱼ * vᵢ(t) - dUⱼ(t)/dⱼᵢ * vⱼ(t) )
# In the frequency domain: Qᴬᴮ(ω)=ΣᵢΣⱼ( dUᵢ(ω)/drᵢⱼ * vᵢ(ω) - dUⱼ(ω)/dⱼᵢ * vⱼ(ω) )
# Note that these are NOT the same! ℱ[ f(t) * g(t) ] ≠ ℱ[ f(t) ] * ℱ[ g(t) ] (or f(ω)*g(ω))
# another way to think about this is: our fundamental question is "what (frequency of) oscillatory forces result in energy flow", which is a slightly separate question from simply "what oscillatory forces are there" (or oscillatory velocities, as in vDOS). Imagine the simplest case where F=-sin(ωt) and v=cos(ωt). The force "leads" the velocity slightly, and this is a case where the there is there clearly ought to be a net Q at ω. mathematically though, Q(t)=sin(ωt)*cos(ωt) which is a function with *half* the periodicity (function is positive when F and v are both positive, or when F and v are both negative). clearly we want F(ω) and v(ω) separate, i.e. ℱ[ F(t) ] * ℱ[ v(t) ]
# So if ℱ[ f(t) * g(t) ] ≠ ℱ[ f(t) ] * ℱ[ g(t) ], then what IS ℱ[ f(t) ] * ℱ[ g(t) ] in the time domain?
# ℱ[ f(t) ] * ℱ[ g(t) ] = ℱ[ ⟨ f(t),g(t) ⟩ ] (where ⟨-⟩ that's a cross-correlation) 
# So Q(ω) is NOT simply ℱ[ Q(t) ], but either ℱ[ F(t) ] * ℱ[ v(t) ] or ℱ[ ⟨ F(t),v(t) ⟩ ]
# And expanding this to "power between sides A and B", we sum over atoms in A and B, only including instances where i is in A and j is in B
# And practically in the code, how do we get forces? and what about the case of a 3-body potential?
# Consider atoms j-i-k where i is the central atom. 
# Let's start with "forces on j" (Fⱼ). compute energy (Eₒ), perturn atom j by dx and recalculate (Eₚⱼ). Fⱼₓ=-(Eₚⱼ-Eₒ)/dx (if Eₚⱼ > Eₒ, force is in negative direction). If we perturbed j, this is net force on j (Fⱼ)
# For a many-body potential, we will make the assumption that satellite atoms only experience a force from the central atom: Fⱼₖ=0, thus Fⱼᵢ=Fⱼ ("force on j by i is the same as the net force on j"). repeat for Fₖᵢ=Fₖ=(Eₚₖ-Eₒ)/dx. For forces on i, we can say interactions are "equal and opposite", Fᵢⱼ=-Fⱼᵢ, Fᵢ=-Fₖᵢ, and then for net force on i, we can sum: Fᵢ=Fᵢⱼ+Fᵢₖ
# Is this assumption of Fⱼₖ=Fₖⱼ=0 allowed?? the potential does not define forces, it defines energy. so we can make whatever statements we want so long as the energy expressions are satisfied! 
# How does this code work? pass average positions, displacements, velocities, where you want to slice, and your stillinger weber file. 
# we'll iterate through all triplets of atoms j-i-k (treating i as the central atom). skip triplets with atoms too far apart. we'll perturb atom j and k so we can separate Fji, Fki, Fij, Fik. repeat for each timestep. and write these off to a file. (no need to recalculate forces on subsequent runs!). with forces known, we run the FFT(correlate(F,v)) stuff above, taking note that, for example, Fj only contributes if atoms i,j are on opposite sides and so on.
def SHF_SW(avg,disp,velocities,sliceX,swfile):
	# why take avg+disp? this will include wrapping (ensuring an atom is always on the same side of the simulation)
	positions=avg+disp
	# read SW potential. create potentialNBody functions which we pass interatomic radii / bond angles into
	r_cut,potential2Body,potential3Body=SW(swfile) ; print("r_cut",r_cut)
	# pre-trim to only keep atoms two r_cuts away from the interface! 
	xmask=np.zeros(len(avg))+1 ; xmask[avg[:,0]<sliceX-2*r_cut]=0 ; xmask[avg[:,0]>sliceX+2*r_cut]=0
	avg=avg[xmask==1,:] ; positions=positions[:,xmask==1,:] ; velocities=velocities[:,xmask==1,:]
	na=len(avg) ; nt=len(positions)

	# displacements we're going to apply to atoms (no displacements, dx,dy,dz)
	dxyzs=np.asarray([ [0,0,0],[.0001,0,0],[0,.0001,0],[0,0,.0001] ])

	# calculate distances between pairs of atoms (allows quick and easy filtering, e.g. skipping triplets of atoms all on the same side)
	print("calculating dBetween")
	#dBetween=positions[:,:,None,:]-positions[:,None,:,:]	# t,i,j,xyz will tell us distance (at each time) between atoms i and j
	#dBetween=np.sum( dBetween**2, axis=3 )			# dx,dy,dz --> distance between
	#dBetween=np.amin( dBetween, axis=0)			# minimum distance through course of simulation
	dBetween=np.zeros((na,na))
	for i in tqdm(range(na)):				# loop instead (positions might be huge and we'll blow up RAM if we do above)
		for j in range(na):
			if i==j:
				dBetween[i,j]=np.inf ; continue	# don't let self-self distance be set to zero...
			v=positions[:,j,:]-positions[:,i,:] 	# t,a,xyz --> t,xyz (3D vector for each timestep)
			d=np.sqrt( np.sum( v**2, axis=1) )	# length of said 3D vector
			dBetween[i,j]=np.amax(d)		# furthest distance two atoms are ever apart (e.g. exclude atoms that sometimes just stray outside the cutoff distance)

	print("calculate forces (or check saved forces files)")
	for i in tqdm(range(na)):
		xi=avg[i,0]
		for j in range(na):
			if j==i:
				continue
			if dBetween[i,j]>r_cut-.001:
				continue
			xj=avg[j,0]
			for k in range(na):
				if k==j or k==i:
					continue
				if dBetween[i,k]>r_cut-.001:
					continue
				xk=avg[k,0]
				if xi < sliceX and xj < sliceX and xk < sliceX:
					continue
				if xi > sliceX and xj > sliceX and xk > sliceX:
					continue

				logfile="SHF/Fi_"+str(i)+"-"+str(j)+"-"+str(k)+".txt"
				if os.path.exists(logfile):
					Fi_xyz=list( np.loadtxt(logfile) ) # column for Fx,Fy,Fz --> nt,3
				else:
					Fi_xyz=[]
				logfile="SHF/Fj_"+str(i)+"-"+str(j)+"-"+str(k)+".txt"
				if os.path.exists(logfile):
					Fj_xyz=list( np.loadtxt(logfile) )
				else:
					Fj_xyz=[]
				logfile="SHF/Fk_"+str(i)+"-"+str(j)+"-"+str(k)+".txt"
				if os.path.exists(logfile):
					Fk_xyz=list( np.loadtxt(logfile) ) # column for Fx,Fy,Fz --> nt,3
				else:
					Fk_xyz=[]

				for t in range(len(Fi_xyz),nt):
					vik=positions[t,k,:]-positions[t,i,:]
					#if lVec(vik)>r_cut-.001:
					#	continue
					Ej=np.zeros(4) # used for force on atom i (perturning position of atom i)
					for n,dxyz in enumerate(dxyzs):
						vij=(positions[t,j,:]+dxyz)-positions[t,i,:]		# vector "from i, to j"
						vik=positions[t,k,:]-positions[t,i,:]
						rij=lVec(vij) ; rik=lVec(vik) ; tijk=angBetween(vij,vik)
						Ej[n] = potential3Body(rij,rik,tijk)+potential2Body(rij)
						#print("vij",vij,"vik",vik,"rij",rij,"rik",rik,"tijk",tijk,"Ei["+str(n)+"]",Ei[n])
					Fji=-(Ej[1:]-Ej[0])/.0001 # force on i (assume force from j is zero)
					#print("Fi",Fi)
					#time.sleep(1)
					Ek=np.zeros(4) # used for force on atom k (perturbing position of atom k)
					for n,dxyz in enumerate(dxyzs):
						vij=positions[t,j,:]-positions[t,i,:]
						vik=(positions[t,k,:]+dxyz)-positions[t,i,:]
						rij=lVec(vij) ; rik=lVec(vik) ; tijk=angBetween(vij,vik)
						Ek[n] = potential3Body(rij,rik,tijk)+potential2Body(rij)
					Fki=-(Ek[1:]-Ek[0])/.0001 # force on k (assume force from j is zero)
					Fik=-1*Fki				# F on i by k is equal-and-opposite F on k by i
					Fij=-1*Fji
					Fi=Fij+Fik 				# Fi=Fij+Fik. F on i, from j, is total F on i (Fi) minus force on i by k
					# atom i always contributes to energy exchange, since it's central
					#if xi>sliceX:			# atom i to the right of the interface, net Q is in + direction across
					#	Q+=Fi*velocities[t,i,:] # Fx,Fy,Fz times Vx,Vy,Vz
					#else:				# or vice versa, can have net Q in - direction across
					#	Q-=Fi*velocities[t,i,:]
					# atom j only contributes if i,j on opposing sides of the interface
					#if ( xi<sliceX and xj>sliceX ):	# j is right of the interface, receiving heat
					#	Q+=Fji*velocities[t,j,:]
					#if ( xi>sliceX and xj<sliceX ): # j is on left. any energy j "gains" is a "negative flow" across the interface
					#	Q-=Fji*velocities[t,j,:]
					# atom k only contributes if i,k on opposing sides of the interface
					#if ( xi<sliceX and xk>sliceX ): 
					#	Q+=Fji*velocities[t,j,:]
					#if ( xi>sliceX and xk<sliceX ): 
					#	Q-=Fji*velocities[t,j,:]
					Fi_xyz.append(Fi) ; Fj_xyz.append(Fji) ; Fk_xyz.append(Fki)

				np.savetxt("SHF/Fi_"+str(i)+"-"+str(j)+"-"+str(k)+".txt",Fi_xyz)
				np.savetxt("SHF/Fj_"+str(i)+"-"+str(j)+"-"+str(k)+".txt",Fj_xyz)
				np.savetxt("SHF/Fk_"+str(i)+"-"+str(j)+"-"+str(k)+".txt",Fk_xyz)

				if os.path.exists("killSHF"):
					sys.exit()
	
	#	Qt.append(Q)
	#return Qt

	print("calculate Q from F and v")
	Qt=np.zeros((nt,3)) ; Qw=np.zeros((nt,3),dtype=complex)
	for i in tqdm(range(na)):
		xi=avg[i,0]
		for j in range(na):
			if j==i:
				continue
			if dBetween[i,j]>r_cut-.001:
				continue
			xj=avg[j,0]
			for k in range(na):
				if k==j or k==i:
					continue
				if dBetween[i,k]>r_cut-.001:
					continue
				xk=avg[k,0]
				logfile="SHF/Fi_"+str(i)+"-"+str(j)+"-"+str(k)+".txt"
				if not os.path.exists(logfile):
					continue
				logfile="SHF/Fi_"+str(i)+"-"+str(j)+"-"+str(k)+".txt"
				Fi_xyz=np.loadtxt(logfile)
				logfile="SHF/Fj_"+str(i)+"-"+str(j)+"-"+str(k)+".txt"
				Fj_xyz=np.loadtxt(logfile)
				logfile="SHF/Fk_"+str(i)+"-"+str(j)+"-"+str(k)+".txt"
				Fk_xyz=np.loadtxt(logfile)

				# "LLR" for j--i-|-k for example
				LR={True:"L",False:"R"}[xi<sliceX] + {True:"L",False:"R"}[xj<sliceX] + {True:"L",False:"R"}[xk<sliceX]
				for xyz in [0,1,2]:
					if LR=="LRR": # both j,k on opposite side
						Qt[:,xyz]-=np.correlate(Fi_xyz[:,xyz],velocities[:,i,xyz],mode="same")	# i on left gaining energy is net
						Qt[:,xyz]+=np.correlate(Fk_xyz[:,xyz],velocities[:,k,xyz],mode="same")	# "left" flow, convention negative
					if LR=="RLL": # same as above but flip signs
						Qt[:,xyz]+=np.correlate(Fi_xyz[:,xyz],velocities[:,i,xyz],mode="same")
						Qt[:,xyz]-=np.correlate(Fk_xyz[:,xyz],velocities[:,k,xyz],mode="same")
					if LR=="LLR": # j is on the same side as i, so subtract the "force on i by j" Fij
						Fi=Fi_xyz[:,xyz]+Fj_xyz[:,xyz] # Fij=-Fji, Fik=Fi-Fij
						Qt[:,xyz]-=np.correlate(Fi,velocities[:,i,xyz],mode="same")
						Qt[:,xyz]+=np.correlate(Fk_xyz[:,xyz],velocities[:,k,xyz],mode="same")
					if LR=="RRL":
						Fi=Fi_xyz[:,xyz]+Fj_xyz[:,xyz]
						Qt[:,xyz]+=np.correlate(Fi,velocities[:,i,xyz],mode="same")
						Qt[:,xyz]-=np.correlate(Fk_xyz[:,xyz],velocities[:,k,xyz],mode="same")
					if LR=="LRL": # now k is on the same side as i. like LLR but swap j,k
						Fi=Fi_xyz[:,xyz]+Fk_xyz[:,xyz] # Fik=-Fki, Fij=Fi-Fik
						Qt[:,xyz]-=np.correlate(Fi,velocities[:,i,xyz],mode="same")
						Qt[:,xyz]+=np.correlate(Fj_xyz[:,xyz],velocities[:,k,xyz],mode="same")
					if LR=="RLR":
						Fi=Fi_xyz[:,xyz]+Fk_xyz[:,xyz]
						Qt[:,xyz]+=np.correlate(Fi,velocities[:,i,xyz],mode="same")
						Qt[:,xyz]-=np.correlate(Fj_xyz[:,xyz],velocities[:,k,xyz],mode="same")



				#for xyz in [0,1,2]:
				#	# TODO THIS ISN'T QUITE RIGHT. WE ONLY CARE ABOUT F ACROSS THE INTERFACE. Fi MIGHT HAVE SOME FORCE FROM j ON THE SAME SIDE FOR EXAMPLE		#
				#	# atom i always contributes to energy exchange, since it's central
				#	if xi>sliceX:			# atom i to the right of the interface, net Q is in + direction across
				#		Qt[:,xyz]+=np.correlate(Fi_xyz[:,xyz],velocities[:,i,xyz],mode="same") # Fx,Fy,Fz times Vx,Vy,Vz
				#	else:				# or vice versa, can have net Q in - direction across
				#		Qt[:,xyz]-=np.correlate(Fi_xyz[:,xyz],velocities[:,i,xyz],mode="same")
				#	# atom j only contributes if i,j on opposing sides of the interface
				#	if ( xi<sliceX and xj>sliceX ):	# j is right of the interface, receiving heat
				#		Qt[:,xyz]+=np.correlate(Fj_xyz[:,xyz],velocities[:,j,xyz],mode="same")
				#	if ( xi>sliceX and xj<sliceX ): # j is on left. any energy j "gains" is a "negative flow" across the interface
				#		Qt[:,xyz]-=np.correlate(Fj_xyz[:,xyz],velocities[:,j,xyz],mode="same")
				#	# atom k only contributes if i,k on opposing sides of the interface
				#	if ( xi<sliceX and xk>sliceX ): 
				#		Qt[:,xyz]+=np.correlate(Fk_xyz[:,xyz],velocities[:,k,xyz],mode="same")
				#	if ( xi>sliceX and xk<sliceX ): 
				#		Qt[:,xyz]-=np.correlate(Fk_xyz[:,xyz],velocities[:,k,xyz],mode="same")
				
				#if xi>sliceX:			# atom i to the right of the interface, net Q is in + direction across
				#	Qt[:,:]+=Fi_xyz[:,:]*velocities[:,i,:] # Fx,Fy,Fz times Vx,Vy,Vz
				#else:				# or vice versa, can have net Q in - direction across
				#	Qt[:,:]-=Fi_xyz[:,:]*velocities[:,i,:]
				# atom j only contributes if i,j on opposing sides of the interface
				#if ( xi<sliceX and xj>sliceX ):	# j is right of the interface, receiving heat
				#	Qt[:,:]+=Fj_xyz[:,:]*velocities[:,j,:]
				#if ( xi>sliceX and xj<sliceX ): # j is on left. any energy j "gains" is a "negative flow" across the interface
				#	Qt[:,:]-=Fj_xyz[:,:]*velocities[:,j,:]
				# atom k only contributes if i,k on opposing sides of the interface
				#if ( xi<sliceX and xk>sliceX ): 
				#	Qt[:,:]+=Fk_xyz[:,:]*velocities[:,k,:]
				#if ( xi>sliceX and xk<sliceX ): 
				#	Qt[:,:]-=Fk_xyz[:,:]*velocities[:,k,:]

				#for xyz in [0,1,2]:
				#	# atom i always contributes to energy exchange, since it's central
				#	Qi=Fi_xyz[:,xyz]*velocities[:,i,xyz]
				#	Qj=Fj_xyz[:,xyz]*velocities[:,j,xyz]
				#	Qk=Fk_xyz[:,xyz]*velocities[:,k,xyz]
				#	if xi>sliceX:			# atom i to the right of the interface, net Q is in + direction across
				#		Qt[:,xyz]+=np.correlate(Qi,Qi,mode="same") # Fx,Fy,Fz times Vx,Vy,Vz
				#	else:				# or vice versa, can have net Q in - direction across
				#		Qt[:,xyz]-=np.correlate(Qi,Qi,mode="same")
				#	# atom j only contributes if i,j on opposing sides of the interface
				#	if ( xi<sliceX and xj>sliceX ):	# j is right of the interface, receiving heat
				#		Qt[:,xyz]+=np.correlate(Qj,Qj,mode="same")
				#	if ( xi>sliceX and xj<sliceX ): # j is on left. any energy j "gains" is a "negative flow" across the interface
				#		Qt[:,xyz]-=np.correlate(Qj,Qj,mode="same")
				#	# atom k only contributes if i,k on opposing sides of the interface
				#	if ( xi<sliceX and xk>sliceX ): 
				#		Qt[:,xyz]+=np.correlate(Qk,Qk,mode="same")
				#	if ( xi>sliceX and xk<sliceX ): 
				#		Qt[:,xyz]-=np.correlate(Qk,Qk,mode="same")

				#if xi>sliceX:			# atom i to the right of the interface, net Q is in + direction across
				#	Qw[:,:]+=np.fft.fft(Fi_xyz[:,:]*velocities[:,i,:],axis=0) # Fx,Fy,Fz times Vx,Vy,Vz
				#else:				# or vice versa, can have net Q in - direction across
				#	Qw[:,:]+=np.fft.fft(Fi_xyz[:,:]*velocities[:,i,:],axis=0)
				# atom j only contributes if i,j on opposing sides of the interface
				#if ( xi<sliceX and xj>sliceX ):	# j is right of the interface, receiving heat
				#	Qt[:,:]+=Fj_xyz[:,:]*velocities[:,j,:]
				#if ( xi>sliceX and xj<sliceX ): # j is on left. any energy j "gains" is a "negative flow" across the interface
				#	Qt[:,:]-=Fj_xyz[:,:]*velocities[:,j,:]
				# atom k only contributes if i,k on opposing sides of the interface
				#if ( xi<sliceX and xk>sliceX ): 
				#	Qt[:,:]+=Fk_xyz[:,:]*velocities[:,k,:]
				#if ( xi>sliceX and xk<sliceX ): 
				#	Qt[:,:]-=Fk_xyz[:,:]*velocities[:,k,:]

	return np.fft.fft(Qt,axis=0)
	#return Qw

# For each atom i, find atoms j,k,l,m,n,o within the radii limits
# avgs - [na,xyz]
def findNeighbors(avgs,r_min=0,r_max=np.inf):
	neighbors=[]
	na=len(avgs) ; indices=np.arange(na)
	print("calculating neighbor distances")
	d0=np.sqrt( np.sum( (avgs[:,None,:]-avgs[None,:,:])**2 , axis=2) ) # √(dx²+dy²+dz²), yields an na x na "lookup" matrix of distances
	print("iterating")
	for i in range(na):
		mask=np.zeros(na)+1 ; mask[d0[i,:]<r_min]=0 ; mask[d0[i,:]>r_max]=0 ; mask[i]=0
		neighbors.append([i]+list(indices[mask==1])) # ensure atom i is first in the list!
	return neighbors

# only keep neighbors when at least one is in both groups
# neighbors - a potentially-ragged list of lists, each list containing atom IDs
# As, Bs - lists of atom IDs
def filterNeighborsByGroup(neighbors,As,Bs):
	filtered=[]
	for ijk in neighbors:
		inA=[ i for i in ijk if i in As ]
		inB=[ i for i in ijk if i in Bs ]
		if len(inA)>0 and len(inB)>0:
			filtered.append(ijk)
	return filtered

# positions - [nt,na,xyz]
# potential - a function which can be passed a list of atom's positions [na,xyz] and return the potential energy
# atomSets - lists of atomIDs to feed into potential. e.g. i,j,k triplets for a 3-body potential
def calculateInteratomicForces(positions,potential,atomSets,perturbBy=.0001):	#   B     H	Perturb B, assume 
	os.makedirs("calculateInteratomicForces",exist_ok=True)			#    '-. /.- G	no force on B by
	nt=len(positions)							#  C----A'.  	C, ergo, Fb=Fba
	nBody=len(atomSets[0])							#     .' \ '-F	repeat for C-G, 
	#dxyz=np.zeros((nBody,nBody)) ; dxyz.flat[::nBody+1]=perturbBy		#    D    E	yields pairwise. 
	for ijk in tqdm(atomSets):						# Don't worry, later, B will be 
		for xyz in range(3):						# the central atom to capture Fbc etc
			fileout="calculateInteratomicForces/F"+xyzString+"_"+ijkString+".npy"
			if os.path.exists(fileout):
				continue
			forces=np.zeros((nt,nBody)) # will hold force at each timestep: total on i, then contribution from j,k,etc
			for t in range(nt):
				atoms=positions[t,ijk,:]
				Vo=potential(atoms)					# potential energy for unperturbed atomic configuration
				for j in range(nBody):					# for each atom in the set
					dx=np.zeros((nBody,3)) ; dx[j,xyz]=perturbBy	# perturbation matrix: perturb the jth atom, in x or y or z
					#print("xyz",xyz,"j",j,"dx",dx)
					Vi=potential(atoms+dx)				# recalculate potential, perturbing atom j in x y or z
					forces[t,j]=-(Vi-Vo)/perturbBy			# Fx=-dE/dx. if Vp > Vo, Force is negative!
			xyzString=["x","y","z"][xyz]
			ijkString=",".join([ str(j) for j in ijk ])
			#fileout="calculateInteratomicForces/F"+xyzString+"_"+ijkString+".txt"
			#np.savetxt(fileout,forces)

			np.save(fileout,forces)

def SHF(velocities,As,Bs):
	forceFiles=glob.glob("calculateInteratomicForces/*")
	nt=len(velocities)
	Qt=np.zeros((nt,3))
	for f in tqdm(forceFiles):
		xyz=["Fx","Fy","Fz"].index(f.split("/")[-1].split("_")[0])
		ijk=f.split("/")[-1].split("_")[-1].replace(".txt","").split(",")
		ijk=[ int(i) for i in ijk ]
		inA=[ i for i in ijk if i in As ]
		inB=[ i for i in ijk if i in Bs ]
		forces=np.loadtxt(f)		# columns are: Fi, Fj, Fk. these came from perturbing atoms i,j,k and so on.
		# if atom i on the left, j,k,etc gaining energy is +Q and i gaining energy is -Q. flipped if i on right
		i=ijk[0]
		sign={True:1,False:-1}[ i in As ]
		# we're going to sum up forces acting on by only atoms on the other side of the boundary
		Fi=np.zeros(nt)
		for c,j in enumerate(ijk):
			if c==0:
				continue
			if ( i in As and j in Bs ) or ( i in Bs and j in As ): # i on left, j on right, or vice-versa. IF SO, Fj*v*j counts towards Q
				# Fj was found by perturbing j, so this is F on j by i. 
				Qt[:,xyz]+=sign*np.correlate(forces[:,c],velocities[:,j,xyz],mode="same")
				Fi-=forces[:,c] # F on i by j is equal-and-opposite
				#print(types[i],types[j])
		Qt[:,xyz]-=sign*np.correlate(Fi,velocities[:,i,xyz],mode="same")
	return np.fft.fft(Qt,axis=0)


