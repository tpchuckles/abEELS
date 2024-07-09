import numpy as np
import glob,os
from tqdm import tqdm

def scrapeLayers(filename):
	ts=[] ; Ts=[] ; X=[]
	f=open(filename, 'r')
	while True:
		l=f.readline()
		if not l:		# EOF
			break
		if l[0]=="#":		# commented line (ignore)
			continue
		elif l[0]!=" ":		# line contains timestep number
			l=l.split()
			ts.append(int(l[0])) ; Ts.append([])
		else:			# line contains layer index, position and temperature
			l=l.split()
			Ts[-1].append(float(l[3]))
			if len(Ts)==1:
				X.append(float(l[1]))
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
					if "out_"+n+"_"+str(i+1)+".txt" in exported: # if the file exists, include it in the dict
						data[-1][n]=[]
						out=np.loadtxt("out_"+n+"_"+str(i+1)+".txt")
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

def qdump(filename,timescaling=1,convert=True): # OBSCENELY FAST IN COMPARISON TO scrapeDump()
	if os.path.exists(filename+"_t.npy"):
		print("ignoring qdump, reading npy files instead")
		ts=np.load(filename+"_t.npy")
		pos=np.load(filename+"_p.npy")
		vel=np.load(filename+"_v.npy")
		return pos,vel,ts

	from ovito.io import import_file # TODO WEIRD BUG, AFTER THIS RUNS, WE CAN'T PLOT STUFF WITH NICEPLOT. WE GET THE FOLLOWING ERROR: ImportError: Cannot load backend 'TkAgg' which requires the 'tk' interactive framework, as 'qt' is currently running
	# WHY TF DOES OVITO LOAD qt AND HOW DO WE UNLOAD IT??
	print("reading qdump")
	pipeline = import_file(filename)
	nt=pipeline.source.num_frames
	data=pipeline.compute(0)
	na,nxyz=np.shape(data.particles.positions.array)
	pos=np.zeros((nt,na,3))
	vel=np.zeros((nt,na,3))
	ts=np.arange(nt)*timescaling
	for n in tqdm(range(nt)):
		data=pipeline.compute(n)
		pos[n,:,:]=data.particles.position.array
		vel[n,:,:]=data.particles.velocities.array
	if convert:
		np.save(filename+"_t.npy",ts)
		np.save(filename+"_p.npy",pos)
		np.save(filename+"_v.npy",vel)
	return pos,vel,ts


def avgPos(pos,sx,sy,sz): # takes "pos" as from: pos,timesteps=scrapeDump(trange="-"+str(avgOver)+":"), [nS,nA,xyz]
	nS,nA,na=np.shape(pos)
	displacements=np.zeros((nS,nA,na))
	for t in tqdm(range(nS)):
		# distance between initial position and position at time t (for each atom, along each axis), including wrapping
		displacements[t,:,:]=dxyz(pos[0,:,:],pos[t,:,:],sx,sy,sz) 
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

def dxyz(pos1,pos2,sx,sy,sz): # given two snapshots of positions [[xa,ya,za],[xb,yb,zb],...] x2, don't just do dxyz=xyz1-xyz2. must include wrapping!
	dxyz_0=pos2-pos1
	# we use these for wrapping
	wx,i_range=getWrapAndRange(sx,0) ; wy,j_range=getWrapAndRange(sy,1) ; wz,k_range=getWrapAndRange(sz,2)
	# for pos_1 in the 27 surrounding positions (original, and 26 neighbors), keep only the smallest (absolute) distance found
	for i in i_range:
		for j in j_range:
			for k in k_range:
				dxyz_w=pos2+i*wx+j*wy+k*wz-pos1 # if an atom crossed the x axis (from +x to -x ie L) it'll register as closer if we take (x0+L)-xf
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
def SED(avg,velocities,p_xyz,v_xyz,a,nk=100,bs='',perAtom=False,ks='',keepComplex=False):
	nt,na,nax=np.shape(velocities)
	if len(bs)==0:
		bs=np.arange(na)
	else:
		na=len(bs)
	nt2=int(nt/2)
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
	# F(w) = ∫ f(t) * exp( -i 2 π w t ) dt
	# we can reformulate the above eqaution as:
	# f(t) = u°(n,t) * exp( i k x )
	# Φ(k,ω) = | FFT{ Σn f(t) } |²
	# of course, this code still *can* analyze crystals with folding: the 
	# user should simply call this function multiple times, passing the "bs"
	# argument with a list of atom indices for us to use

	if isinstance(p_xyz,(int,float)): # 0,1,2 --> x,y,z
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

	if isinstance(v_xyz,(int,float)):
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
			Q=neighborLocations ; R=np.identity(3) #"attempting rotation of P onto Q"
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
	shape=np.asarray(np.shape(array),dtype=int) ; shape[axis]/=n
	new=np.zeros(shape)
	for i in range(n):
		if axis==0:
			new+=array[i::n]
		else:
			new+=array[:,i::n]
	new/=n
	return new


