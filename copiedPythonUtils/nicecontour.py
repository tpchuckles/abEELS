import matplotlib,os
import matplotlib.pyplot as plt #https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.html
import matplotlib.tri as tri
import matplotlib.cm
import numpy as np
from niceplot import processText

defaultcmap=matplotlib.cm.inferno
params={'font':{ 'family':'arial' , 'weight':'regular' , 'size':16 },
	'axes':{ 'titlesize':16 , 'autolimit_mode':'round_numbers' },
	'figure':{ 'figsize':(8,6) , 'dpi':192 }}

def setContRC(key,para):
	matplotlib.rc(key, **para)
for key in params:
	setContRC(key,params[key]) # replaces "matplotlib.rc(key, **params[key])"

ZXY=[]
# zvals should be an NxM matrix, ordered [j,i], xvals should be length M [j], yvals is length N [i], following the convention of numpy (row index is first) or PIL (y index of pixel is first, equivalent to numpy). you should pass Zs.T if your Zs matrix is ordered x,y instead of y,x 
def contour(zvals,xvals,yvals,filename='',heatOrContour="heat",useLast=False,extras=[],**kwargs):

	global fig,ax,CS,cbar,ticks,UB,LB

	# 1D DATASET PASSED, USE TRIANGULATION INTERPOLATION
	if isinstance(zvals[0],int) or isinstance(zvals[0],float):
		print("triangulating")
		# Create grid values first.
		xi = np.linspace(min(xvals), max(xvals), 1000)
		yi = np.linspace(min(yvals), max(yvals), 1000)
		# Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
		triang = tri.Triangulation(xvals, yvals)
		interpolator = tri.LinearTriInterpolator(triang, zvals)
		Xi, Yi = np.meshgrid(xi, yi)
		zi = interpolator(Xi, Yi)
		#plt.tricontourf(xvals,yvals,zvals) ; plt.show()
		xvals=Xi[0,:] ; yvals=Yi[:,0] ; zvals=zi
		if kwargs.get("returnTriangulated",False):
			print("return!")
			return zvals,xvals,yvals
		#print(xvals)
	else: 
		lens=[ len(z) for z in zvals ]
		# 2D RAGGED DATASET, USE 1D INTERPOLATION TO GRIDIFY
		if len(set(lens))!=1 or np.shape(zvals)==np.shape(xvals):
			from scipy.interpolate import interp1d
			Zs=[]
			minx=max([ min(xs) for xs in xvals ])
			maxx=min([ max(xs) for xs in xvals ])
			xs=np.linspace(minx,maxx,1000)
			for z,x in zip(zvals,xvals):
				f=interp1d(x,z)
				Zs.append(f(xs))
			zvals=np.asarray(Zs) ; xvals=xs


	if np.amin(zvals)==np.amax(zvals):
		print("nicecontour: bad z bounds. exiting")
		print(np.amin(zvals),np.amax(zvals))
		return
	# if we're saving files, use the "Agg" backend. (and save off whatever the current backend is, and restore it after we save it, to prevent messing up the user's python environment). if we're showing, just default to whatever the user's default is
	#print(xvals,yvals,len(xvals),len(yvals),np.shape(zvals))
	backend=matplotlib.get_backend()
	if len(filename)!=0:# and filename!="PLOTOBJ":
		matplotlib.use("Agg") # https://stackoverflow.com/questions/31156578/matplotlib-doesnt-release-memory-after-savefig-and-close



	#if not useLast: # TODO this was commented out before. idk why??? (we need it now to allow surface plots to be overlaid)
	# ah. ALWAYS create new fig,ax, but always populate it with the full list (which goes into ZXY
	fig,ax=plt.subplots()
	#	#ax.plot([0,1], [0,1])
	plt.clf() # need this or you get duplicate cbars if you generate multiple plots! 
	#else:
	#	#ax=plt.gcf().get_axes()
	#	ax.patch.set_facecolor([0.1,0.2,0.3,0.4])
	#	print("useLast is True",CS.collections,filename)
	#print("USELAST",useLast)
	global ZXY ; xyz=[zvals,xvals,yvals]
	#if 
	if not useLast:	
		ZXY=[xyz]
	else:
		ZXY.append(xyz)
	

	#if len(filename)==0:
	#	# if macos error, try: "export MPLBACKEND=TKAgg" https://stackoverflow.com/questions/55811545/importerror-cannot-load-backend-tkagg-which-requires-the-tk-interactive-fra
	#	matplotlib.use("TkAgg") # https://stackoverflow.com/questions/56656777/userwarning-matplotlib-is-currently-using-agg-which-is-a-non-gui-backend-so
	#else:
	#	matplotlib.use("Agg") # https://stackoverflow.com/questions/31156578/matplotlib-doesnt-release-memory-after-savefig-and-close
	LB,UB=np.nanmin(zvals),np.nanmax(zvals)
	if "xlim" in kwargs.keys() or "ylim" in kwargs.keys():
		mask=np.ones(np.shape(zvals))
		xlim=kwargs.get("xlim",[min(xvals),max(xvals)])
		ylim=kwargs.get("ylim",[min(yvals),max(yvals)])
		if xlim[0] is None:
			xlim[0]=min(xvals)
		if xlim[1] is None:
			xlim[1]=max(xvals)
		if ylim[0] is None:
			ylim[0]=min(yvals)
		if ylim[1] is None:
			ylim[1]=max(yvals)

		mask[:,xvals<xlim[0]]=0 ; mask[:,xvals>xlim[1]]=0
		mask[yvals<ylim[0],:]=0 ; mask[yvals>ylim[1],:]=0
		cropped=zvals[mask==1]
		LB,UB=np.nanmin(cropped),np.nanmax(cropped)

	if "zlim" in kwargs.keys():
		zlim=kwargs["zlim"] ; LB={True:LB,False:zlim[0]}[zlim[0] is None] ; UB={True:UB,False:zlim[1]}[zlim[1] is None]
	nticks=kwargs.get("nticks",10)
	#print(LB,UB)
	ticks=kwargs.get("zticks",np.linspace(LB,UB,nticks))
	
	# for heatmaps, you can use tricontourf, but that won't work for contours. need to follow https://matplotlib.org/stable/gallery/images_contours_and_fields/irregulardatagrid.html


	if heatOrContour in ["heat","both"]:
		CS=plt.contourf(xvals,yvals,zvals,levels=np.linspace(LB,UB,500),cmap=kwargs.get("cmap",defaultcmap))
		#print(np.amin(zvals),np.amax(zvals))
		#cbar=plt.colorbar(ticks=ticks)
		for c in CS.collections:
			c.set_edgecolor("face")
		#	c.set_rasterized(True)
		#nDecimals=max(0,int(1-np.floor(np.log(UB-LB)/np.log(10)))) # 0.35-0 --> -0.4559319556497244 --> -1 --> could be represented at 3.5e-1. if it was 35, we'd want 0 decimals. if it was 3.5 we'd want 1 decimal. -1 we want 2. 350, we still want 0 decimals
		#print(nDecimals)
		#nDecimals=max(nDecimals,0) # https://stackoverflow.com/questions/19986662/rounding-a-number-in-python-but-keeping-ending-zeros
		#ticks=cbar.get_ticks()
		#ticks=[ format(v,'.'+str(nDecimals)+'f') for v in ticks]
		#cbar.ax.set_yticklabels(ticks)
		#cbar.ax.set_title(kwargs.get("zlabel","ZTITLE"))
		addcbar(kwargs)
	# TODO should contours have cbars? no need, if inline labels are used...
	# TODO beware: useLast=True -> no plt.clf() -> if heatmap is used, duplicative cbars will result. this might be okay though? because overlapping a heatmap seems like nonsense?
	if heatOrContour in ["contour","both"]:
		levels=kwargs.get("levels",np.linspace(LB,UB,20))
		contourKwargs={"levels":levels,"linestyles":kwargs.get("linestyle","-"),"linewidths":kwargs.get("linewidth",1)}
		color=kwargs.get("linecolor","black")
		try:
			colorOrMap={True:"cmap",False:"colors"}[ color in matplotlib.colormaps.keys() ]
		except:
			colorOrMap={True:"cmap",False:"colors"}[ color in matplotlib.cm.cmaps_listed.keys() ]
		contourKwargs[colorOrMap]=color
		#CS=plt.contour(xvals,yvals,zvals, **contourKwargs)
		for z,x,y in ZXY:
			CS=plt.contour(x,y,z, **contourKwargs)
		#if heatOrContour!="both":
		#	cbar=plt.colorbar()
		if kwargs.get("inline",False):
			plt.clabel(CS, inline=1)
	if heatOrContour in ["surface"]:
		if "plot_surface" not in dir(ax):
			print("NO PLOT_SURFACE IN DIR(AX), REGENERATING")
			fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
		# Plot the surface
		x=xvals[None,:]*np.ones(np.shape(zvals)) ; y=yvals[:,None]*np.ones(np.shape(zvals))

		surf = ax.plot_surface(x,y,zvals, linewidth=1, antialiased=False,cmap=kwargs.get("cmap",defaultcmap),alpha=.1)
		ax.contour3D(x,y,zvals,levels=[.025])
		#plt.show()
	if heatOrContour in ["pix"]: # shows unsmoothed raw data as pixels. creates substantially smaller svg files too! 
		print(np.shape(zvals))
		aspect=kwargs.get("aspect","auto")
		#if yvals[0]<yvals[1]:		# BEWARE: imshow displays with origin in upper-left. and imshow takes extent, not the actual
		#	zvals=zvals[::-1,:]	# col-by-col and row-by-row values. So if you have ascending yvals or descending xvals, heat or 
		#if xvals[0]>xvals[1]:		# contour modes would be correct, but the pix map would be flipped. so we need to manually
		#	zvals=zvals[:,::-1]	# detect and flip zvals as appropriate DOING IT BASED ON SINGLE PIXEL PAIRS IS BAD THOUGH
		zvals[zvals<LB]=np.nan ; zvals[zvals>UB]=np.nan
		# AUTO SORTING OF ROWS AND COLUMNS? 
		# Suppose the user passed: xvals=[1,2,3,4,5,-5,-4,-3,-2,-1] (common if it's frequencies that came from np.fft.fftfreq!). imshow simply shows the image, but we need to reorder the columns!
		# calculate ordering of rows/columns
		x_indices=np.arange(len(xvals)) ; xvals, x_indices = zip(*sorted(zip(xvals, x_indices)))	# x_indices will be: 5,6,7,8,9,0,1,2,3,4
		# apply indices. previously we did this: (but why?)
		#for i,ii in enumerate(x_indices):								# i=0, ii=5
		#	zvals_sorted[:,i]=zvals[:,ii]								# put values in column 5 into column 0
		#zvals=zvals_sorted
		zvals=zvals[:,x_indices]
		y_indices=np.arange(len(yvals)) ; yvals, y_indices = zip(*sorted(zip(yvals, y_indices)))
		zvals=zvals[y_indices,:]
		# BEWARE: imshow displays with origin in upper-left. and imshow takes extent, not the actual col-by-col and row-by-row values. We just sorted zvals to have ascending yvals and ascending xvals, so now we need to flip the pix map 
		zvals=zvals[::-1,:]
		
		plt.imshow(zvals,extent=(min(xvals),max(xvals),min(yvals),max(yvals)),cmap=kwargs.get("cmap",defaultcmap),aspect=aspect)
		#cbar=plt.colorbar(ticks=ticks)
		#if len(np.shape(zvals))<3:
		addcbar(kwargs)
	#if useLast:
	#	print(CS.collections)

	# GENERAL
	plt.title( processText( kwargs.get("title","TITLE") ) )
	plt.xlabel( processText( kwargs.get("xlabel","XLABEL") ) )
	plt.ylabel( processText( kwargs.get("ylabel","YLABEL") ) )
	plt.xlim( kwargs.get("xlim",None) )
	plt.ylim( kwargs.get("ylim",None) )
	#plt.clim( kwargs.get("zlim",None) )
	if "aspect" in kwargs.keys():
		plt.gca().set_aspect(kwargs["aspect"])

	if "overplot" in kwargs.keys(): # it's possible to pass a list of dicts of xs,ys,markers, to be plotted over top of the contour/heatmap
		for dataset in kwargs["overplot"]:
			#print("overplotting",dataset)
			xs=dataset["xs"] ; ys=dataset["ys"] ; kind=dataset["kind"] # these are the only 3 required keys! all else is kwargs
			kw={ k:dataset[k] for k in dataset.keys() if k not in ["xs","ys","kind"] }
			if kind=="scatter":
				plt.scatter(xs,ys,**kw)	
			elif kind=="line":
				plt.plot(xs,ys,**kw,marker="")
			elif kind=="text":		# TODO consider using annotate instead? https://stackoverflow.com/questions/14432557/scatter-plot-with-different-text-at-each-data-point this buys you arrows, and you can still do plt.annotate
				text=kw["text"]		# text-type overplotting has one more required key: "text"
				kw={ k:kw[k] for k in kw.keys() if k!="text" }
				for x,y,t in zip(xs,ys,text):
					plt.text(x,y,t,**kw)

	if "flip" in kwargs.keys():
		if "x" in kwargs["flip"]:
			plt.gca().invert_xaxis()
		if "y" in kwargs["flip"]:
			plt.gca().invert_yaxis()

	for extra in extras:
		extra(plt)
	if len(filename)>0:
		if ".svg" in filename:
			matplotlib.rc("svg", **{'fonttype':'none'}) # ensures text in svg files is saved as text (for later editing)
		plt.savefig(filename)
	else:
		plt.show()

	matplotlib.use(backend)
	#return CS

def addcbar(kwargs):
	global cbar,ticks
	cbar=plt.colorbar(ticks=ticks)
	nDecimals=max(0,int(1-np.floor(np.log(UB-LB)/np.log(10)))) # 0.35-0 --> -0.4559319556497244 --> -1 --> could be represented at 3.5e-1. if it was 35, we'd want 0 decimals. if it was 3.5 we'd want 1 decimal. -1 we want 2. 350, we still want 0 decimals
	#print(nDecimals)
	#nDecimals=max(nDecimals,0) # https://stackoverflow.com/questions/19986662/rounding-a-number-in-python-but-keeping-ending-zeros
	ticks=cbar.get_ticks()
	ticks=[ format(v,'.'+str(nDecimals)+'f') for v in ticks]
	cbar.ax.set_yticklabels(ticks)
	cbar.ax.set_title(kwargs.get("zlabel","ZTITLE"))


def invertContourColors(plt):
	global fig,CS,cbar ; ax=plt.gca()		# where niceplot's invertColors is passed global fix,ax objects, we must retreive ax via plt.gca
	ax.set_facecolor("black")			# inside area of plot (only relevant for contour-style (not heatmap) --> black
	fig.set_facecolor("black")			# outside area of plot --> black
	for s in ["bottom","top","left","right"]:	# border lines around plot --> white
		ax.spines[s].set_color("white")
	ax.xaxis.label.set_color('white')		# x axis label text --> white
	ax.tick_params(axis='x', colors='white')	# x axis tick marks --> white
	ax.yaxis.label.set_color('white')
	ax.tick_params(axis='y', colors='white')
	ax.title.set_color('white')			# both title texts --> white
	if "cbar" in globals():
		cbar.outline.set_color("white")			# border lines around cbar --> white
		cbar.ax.title.set_color('white')		
		cbar.ax.tick_params(axis='y', colors='white')	# cbar ticks --> white
	#except:
	#	pass

def fillHoles(Zs):
	where=np.where(np.isnan(Zs)) ; N,M=np.shape(Zs) ; s=2
	for y,x in zip(*where):
		#if x==0 or x==sx-1 or y==0 or y==sy-1:
		#	continue
		#neighborsOK=True
		#for i in [-1,1]:
		#	for j in [-1,1]:
		#		if np.isnan(Zs[y+j,x+i]):
		#			neighborsOK=False
		#if neighborsOK:
		#	Zs[y,x]=np.nanmean(Zs[y-1:y+2,x-1:x+2])
		nanMask=np.isnan(Zs[y-s:y+s+1,x-s:x+s+1]) ; okMask=~nanMask # 5x5 matrix of True/False, centered around x,y "~" flips True/False. 
		goodNeighbors=len(np.where(okMask)[0]) # count how many are not nans
		if goodNeighbors>=s*8: # outer border
			Zs[y,x]=np.nanmean(Zs[y-s:y+s+1,x-s:x+s+1])
		#Zs[y,x]=0

def tileQuadrant(Zs,xs,ys):
	nx=len(xs) ; ny=len(ys)
	# new area is 4x the size, don't duplicate center line?
	Znew=np.zeros((ny*2-1,nx*2-1)) ; xnew=np.zeros(nx*2-1) ; ynew=np.zeros(ny*2-1)
	# 1st quadrant: flip data for first half
	xnew[:nx]=-1*xs[::-1] ; ynew[:ny]=-1*ys[::-1]
	Znew[:ny,:nx]=Zs[::-1,::-1]
	# 3rd quadrant: don't flip data for second half
	xnew[nx-1:]=xs[:] ; ynew[ny-1:]=ys[:]
	Znew[ny-1:,nx-1:]=Zs[:,:]
	# 2nd and 4th quadrants are combinations of flipped/unflipped x/y
	Znew[:ny,nx-1:]=Zs[::-1,:]
	Znew[ny-1:,:nx]=Zs[:,::-1]
	return Znew,xnew,ynew


def untiltZs(Zs,xs,ys,twist=True):
	print(np.shape(Zs),np.shape(xs),np.shape(ys))
	slopesX=np.mean(np.gradient(Zs,xs[1]-xs[0],axis=1),axis=1) # calculate slope of each row. simple mean of slope between each point
	slopesY=np.mean(np.gradient(Zs,ys[1]-ys[0],axis=0),axis=0) # calculate slope of each column
	xs=np.asarray(xs) ; ys=np.asarray(ys)
	Znew=np.zeros(np.shape(Zs)) ; Znew+=Zs
	if twist:
		Znew-=xs[None,:]*slopesX[:,None] # offset each point by the slope in x
		Znew-=ys[:,None]*slopesY[None,:]
	else:
		Znew-=xs[None,:]*np.mean(slopesX)[None,None]
		Znew-=ys[:,None]*np.mean(slopesY)[None,None]

	return Znew

def unbendZs(Zs,xs,ys,axis="xy",saveframes=""):
	def quad(x,a,b,c):
		return a*x**2+b*x+c
	from scipy.optimize import curve_fit
	from niceplot import plot
	Znew=np.zeros(np.shape(Zs)) ; Znew+=Zs
	if "y" in axis:
		for i in range(len(xs)):			# a quadratic function is fitted to each column of data
			parm,cov=curve_fit(quad,ys,Znew[:,i])
			if len(saveframes):
				plot([ys,ys],[Znew[:,i],quad(ys,*parm)],filename=saveframes+"y_"+str(i)+".png")
			Znew[:,i]-=quad(ys,*parm)		# and the Zs are offset by each quadratic
	if "x" in axis:
		for j in range(len(ys)):
			parm,cov=curve_fit(quad,xs,Znew[j,:])
			if len(saveframes):
				plot([xs,xs],[Znew[j,:],quad(xs,*parm)],filename=saveframes+"x_"+str(j)+".png")
			Znew[j,:]-=quad(xs,*parm)
	return Znew

def interp(Zs,x,y,nx,ny=0,method='cubic'):
	if ny==0:
		ny=nx
	from scipy.interpolate import RegularGridInterpolator
	interp=RegularGridInterpolator((y,x),Zs,method=method)
	xs=np.linspace(min(x),max(x),nx) ; ys=np.linspace(min(y),max(y),ny)
	ym,xm=np.meshgrid(ys,xs)
	interpolated=interp((ym,xm)) # idk how, but somehow through interpolation, our indices get switched. 
	return interpolated.T,xs,ys

# NAH, THIS SUCKS. Reds/Blues/Greens go from white to black through the given color. that's not what we really want. 
def ZNChannel(Z): # pass a 3 x ny x nx matrix. we'll return a RGB object (ny nx 4) where each layer from the original is mapped to a color
	nl,nx,ny=np.shape(Z)
	Znew=np.zeros((ny,nx,4)) # colormap object: pix-y, pix-x, [R,G,B,A]
	for l,cm in zip(range(nl),[matplotlib.cm.Reds,matplotlib.cm.Blues,matplotlib.cm.Greens,matplotlib.cm.Oranges,matplotlib.cm.Purples]):
		layer=cm(Z[l]) ; layer[:,:,3]=alpha(Z[l])[:,:,3]
		#layer=cm(Z[l],zlim=(0,np.amax(Z[l])*2)) ; layer[:,:,3]=alpha(Z[l])[:,:,3]
		Znew[:,:,:]+=layer
	#Znew[:,:,:]+=matplotlib.cm.Reds(Z[0]) # each element gets a color
	#Znew[:,:,:]+=matplotlib.cm.Blues(Z[1])
	#Znew[:,:,:]+=matplotlib.cm.Greens(Z[2]) # gaussian with radius of 10
	return Znew[:,:,:3]/nl

#def prism(n) # 0 to 1, red to red. red: 255,0,0, orange: 255,122,0, yellow: 255,255,0, green: 0,255,0, blue: 0,0,255, violet: 122,0,255
from matplotlib.colors import LinearSegmentedColormap
roygbvr = [(1,0,0,1), (1,.5,0,1), (1,1,0,1), (0,1,0,1), (0,0,1,1), (.5,0,1,1),(1,0,0,1) ]
roygbvr = LinearSegmentedColormap.from_list("roygbvr", roygbvr)
nroygbvr = [(0,1,1,1), (0,.5,1,1), (0,0,1,1), (1,0,1,1), (1,1,0,1), (.5,1,0,1),(0,1,1,1) ] # THIS IS WHAT YOU SUBTRACT TO GET ROYGBVR
nroygbvr = LinearSegmentedColormap.from_list("nroygbvr", nroygbvr)
rogbvr = [(0,(1,0,0,1)), (.1666666,(1,.5,0,1)), (.333333,(0,1,0,1)), (.66666667,(0,0,1,1)), (.83333333,(.5,0,1,1)),(1,(1,0,0,1)) ]
rogbvr = LinearSegmentedColormap.from_list("rogbvr", rogbvr)
ro = LinearSegmentedColormap.from_list("ro", [ (1,0,0), (1,.5,0) ])
oy = LinearSegmentedColormap.from_list("oy", [ (1,.5,0), (1,1,0) ])
alpha=LinearSegmentedColormap.from_list("roy", [ (0,0,0,0), (0,0,0,1) ])
#arctic_sun = cbkry

def Z3ChannelRGB(Z):
	nl,nx,ny=np.shape(Z)	
	Znew=np.zeros((ny,nx,4)) #; red=np.asarray([255,0,0]) ; yellow=np.asarray([255,255,0]) ; blue=np.asarray([0,0,255])
	Z=[ z/np.amax(z) for z in Z ] ; sumZ=np.sum(Z,axis=0)
	prox_red=Z[0]/sumZ
	prox_yel=Z[1]/sumZ
	prox_blu=Z[2]/sumZ
	Znew[:,:,0]+=prox_red[:,:]*255
	Znew[:,:,1]+=prox_yel[:,:]*255
	Znew[:,:,2]+=prox_blu[:,:]*255
	print(np.amax(Znew))
	#Znew/=3
	a=alpha(sumZ)
	print(np.amax(a))
	Znew[:,:,3]=a[:,:,3]
	#Znew[:,:,3]=255
	return Znew

def colorwheel(xs,ys): # pass a 2D matrix of xs and ys between -1 and 1, we'll return colors for each point (goal is to have a radially-smooth color function usable for Z3Channel
	ny,nx=np.shape(xs)
	Z=np.zeros((ny,nx,4))#+1
	# color is linear with angle about the origin
	thetas=np.arctan2(ys,xs)
	thetas[thetas<0]+=2*np.pi # default -pi to pi, but wheel should start and end with red, so go 0 to 2*pi, then normalize to 1 
	colors=roygbvr(thetas/np.pi/2) # red/yellow/blue with mixing to make oranges/greens/violets
	#colors=matplotlib.cm.hsv(thetas) # red/green/blue with mixing to make cyan and magenta
	#colors=roygbvr(thetas) # red/green/blue with mixing to make orange/green/violets
	Z[:,:,:3]+=colors[:,:,:3]
	#colors=nroygbvr(thetas/np.pi/2)
	#Z[:,:,:3]-=colors[:,:,:3]
	# looks great, but there's a singularity at radius==0. we should fade to black there
	radii=np.sqrt(xs**2+ys**2)
	gauss=np.exp(-radii**2/.1**2)
	#trirad=(1-np.cos(3*thetas))/2
	Z+=( np.ones((ny,nx,4))-Z )*gauss[:,:,None] # MAKE WHITE ("how much does each pixel have left in each color channel")
	#Z-=Z*gauss[:,:,None]			# MAKE BLACK ("how much does each pixel have in each color channel")
	Z[:,:,3]=1 # ensure alpha channel is 1
	#zmax=np.maximum(Z[0],Z[1]) ; zmax=np.maximum(zmax,Z[2])
	#alphas=alpha(zmax)
	#print("alphas",np.amax(alphas[:,:,3]),np.amin(alphas[:,:,3]))
	#Znew[:,:,3]+=alphas[:,:,3]#*255
	#Znew[:,:,3]=255
	#print(Znew)
	return Z

# Convert a 3,ny,nx matrix into R,Y,B color channels, with correct mixing (R+Y=O, Y+B=G,R+B=V)
def Z3Channel(Z,alphalims=[0,1]):
	nl,nx,ny=np.shape(Z)	
	#Znew=np.zeros((ny,nx,4)) #; red=np.asarray([255,0,0]) ; yellow=np.asarray([255,255,0]) ; blue=np.asarray([0,0,255])
	maxs=np.amax(Z,axis=(1,2)) ; maxs[maxs==0]=1
	Z=[ z/m for z,m in zip(Z,maxs) ]
	xs=[ z*np.cos(t) for z,t in zip(Z,[0,2*np.pi/3,4*np.pi/3])]
	COMx=np.sum(xs,axis=0)#/np.sum(xs,axis=0)
	ys=[ z*np.sin(t) for z,t in zip(Z,[0,2*np.pi/3,4*np.pi/3])]
	COMy=np.sum(ys,axis=0)#/np.sum(ys,axis=0)
	#print("xs",np.amax(xs),np.amin(xs))
	#print("ys",np.amax(ys),np.amin(ys))
	#print("COMx",np.amax(COMx),np.amin(COMx))
	#print("COMy",np.amax(COMy),np.amin(COMy))
	#thetas=np.arctan2(COMy,COMx) ; radii=np.sqrt(COMx**2+COMy**2)
	#thetas[thetas<0]+=2*np.pi ; thetas/=np.pi*2
	#print("radii",np.amax(radii),np.amin(radii))
	#print("thetas",np.amax(thetas),np.amin(thetas))
	#print("t,r",np.shape(thetas),np.shape(radii))
	#colors=roygbvr(thetas) # red/yellow/blue with mixing to make oranges/greens/violets
	#colors=matplotlib.cm.hsv(thetas) # red/green/blue with mixing to make cyan and magenta
	#colors=roygbvr(thetas) # red/green/blue with mixing to make orange/green/violets
	#print("colors",colors)
	#Znew[:,:,:3]+=colors[:,:,:3]
	Znew=colorwheel(COMx,COMy)
	zmax=np.maximum(Z[0],Z[1]) ; zmax=np.maximum(zmax,Z[2])		# base alpha on the most intense color channel
	#print(np.amax(zmax))
	zmax/=np.amax(zmax)	# 0-N --> 0-1
	zmax[zmax<alphalims[0]]=alphalims[0]	# 0-1, but want only 0.25-0.75
	zmax[zmax>alphalims[1]]=alphalims[1]
	zmax-=np.amin(zmax) ; zmax/=np.amax(zmax)	# rescale 0.25 to 0 and 0.75 to 1
	alphas=alpha(zmax)

	#zmax=np.sum(Z,axis=0)
	#alphas=alpha(zmax/np.amax(zmax))
	print("alphas",np.amax(alphas[:,:,3]),np.amin(alphas[:,:,3]))
	Znew[:,:,3]=alphas[:,:,3]#*255
	#Znew[:,:,3]=255
	#print(Znew)
	#ij=np.where(Z[0]==np.amax(Z[0]))
	#print(ij)
	#i,j=50,0 # RED SWIGGLE LEFT/RIGHT
	#i,j=0,45 # VERTICAL SWIGGLE
	#i,j=99,33
	#print("Z",[ z[i,j] for z in Z ],"Cxy",COMx[i,j],COMy[i,j],"t,r",thetas[i,j],radii[i,j],"color",colors[i,j],alphas[i,j])
	#Znew[i,j,:3]=0 ; Znew[i,j,3]=1
	#Znew*=255	
	return Znew
	

	

def getContObjs():
	#print(id(ax),id(fig),id(plt)) # from gui.py > TDTR_fitting.py, plt ends up being shared between niceplot.py and nicecontour.py
	return ax,fig
def getCS():
	return CS
def setContObjs(a,f):
	global ax,fig ; ax,fig=a,f


# Why? np.fft.fft2 assumes index 0 means x=0! not always true, so here's a function which handles all the rolling/etc for you. if x₀ is messed up, the phase will be incorrect (even if the magnitude is correct). also, np.fft.fftfreq returns frequencies "out of order" (positives first) so we apply shift to that too. 
def fft2(Zs,xs,ys,maxk=np.inf,inverse=False): # expect Zs to have y,x indexing (just like contour function above). which means for non-2D, x is -1, y is -2 indices
	i=np.argmin(np.absolute(xs)) ; j=np.argmin(np.absolute(ys))
	Zs=np.roll(np.roll(Zs,-i,axis=-1),-j,axis=-2)			# np.fft.fft2 assumes index 0 means x=0! so we must roll
	#contour(Zs,np.arange(len(xs)),np.arange(len(ys)))
	fft=np.fft.fft2(Zs)						
	dx=xs[1]-xs[0] ; dy=ys[1]-ys[0]					# ℱ(ξ) = ∫ f(x) exp(-i2πξx) dx....
	fft*=dx*dy							# but numpy does not integrate! it sums! 
	kxs=np.fft.fftfreq(len(xs),xs[1]-xs[0])
	kys=np.fft.fftfreq(len(ys),ys[1]-ys[0])
	i=len(kxs)//2 ; j=len(kys)//2
	fft=np.roll(np.roll(fft,i,axis=-1),j,axis=-2)			# kx and ky include negative frequencies, so we must unroll! 
	kxs=np.roll(kxs,i) ; kys=np.roll(kys,j)
	#from niceplot import plot
	#plot([np.arange(len(kxs)),np.arange(len(kys))],[kxs,kys])
	ix=np.arange(len(kxs)) ; iy=np.arange(len(kys))			# I generally prefer masks as [0,0,0,0,1,1,1,1,1,1,0,0,0]
	ix=ix[kxs>=-maxk]  ; kxs=kxs[kxs>=-maxk]			# for the elements i want to keep, then kept=original[mask==1]
	ix=ix[kxs<=maxk]  ; kxs=kxs[kxs<=maxk]				# but here, we don't want to make assumptions on the shape.
	iy=iy[kys>=-maxk]  ; kys=kys[kys>=-maxk]			# if Zs was *always* ny,nx, we could do it, but abEELS often
	iy=iy[kys<=maxk]  ; kys=kys[kys<=maxk]				# has nLayers,nPts,ny,nx and so on. instead, use np.take
	fft=np.take(fft,ix,axis=-1)					# which accepts a list of indices and an axis
	fft=np.take(fft,iy,axis=-2)

	# WHY CAN WE USE AN FFT TO CALCULATE AN INVERSE FFT??
	# ℱ(ξ) = ∫ f(x) exp(-i2πξx) dx : https://en.wikipedia.org/wiki/Fourier_transform
	# ℱ⁻¹[g(x)] := ∫ exp(2πixξ) g(ξ) dξ : https://en.wikipedia.org/wiki/Fourier_inversion_theorem
	# it's the same expression, but with a negative sign! apply the -x either before or after
	if inverse:							
		kxs*=-1 ; kys*=-1					# these are now x and y (user passed kx,ky), so apply negative
		fft=fft[::-1,::-1] ; kxs=kxs[::-1] ; kys=kys[::-1]	# x,y now descending, so flip all axes
		# (if we want x to start at zero, we can roll. we assume f(x) is periodic anyway though, so it doesn't really matter)
		i=np.argmin(np.absolute(kxs)) ; j=np.argmin(np.absolute(kys))
		fft=np.roll(np.roll(fft,-i,axis=-1),-j,axis=-2)
		kxs=np.roll(kxs,-i) ; kys=np.roll(kys,-j)
		# and rolling alone isn't enough. values below 0 must be shifted by length (assumes periodic cell)
		lx=(kxs[1]-kxs[0])*len(kxs) ; ly=(kys[1]-kys[0])*len(kys)
		kxs[kxs<0]+=lx ; kys[kys<0]+=ly
	return fft,kxs,kys

# OLD VERSION, RESTORED FROM U Virginia/Research/Various Code/rivannaProcessing/W_Ansyscompare13/Various Code/niceplot/nicecontour.py
# numpy FFT assumes index 0 is t=0 or x=0. if that's not the case, your phase will be wrong! here, you pass Z,xs,ys (Z has y,x index ordering just like contour), we roll as appropriate, and then FFT. we return reciprocal space kx,ky too. 
#def fft2(zs,xs,ys,maxk=np.inf):
#	i=np.argmin(np.absolute(xs)) ; 	j=np.argmin(np.absolute(ys))
#	zs=np.roll(np.roll(zs,-j,axis=-2),-i,axis=-1)
#	kx=np.fft.fftfreq(len(xs),xs[1]-xs[0]) ; ky=np.fft.fftfreq(len(ys),ys[1]-ys[0])
#	f=np.fft.fft2(zs)
#	if maxk<np.amax(kx):
#		ix=np.arange(len(kx)) ; ix[kx<-maxk]=-1 ; ix[kx>maxk]=-1 ; ix=ix[ix>=0]
#		iy=np.arange(len(ky)) ; iy[ky<-maxk]=-1 ; iy[ky>maxk]=-1 ; iy=iy[iy>=0]
#		kx=kx[ix] ; ky=ky[iy] ; f=np.take(f,ix,axis=-1) ; f=np.take(f,iy,axis=-2)
#	return f,kx,ky







