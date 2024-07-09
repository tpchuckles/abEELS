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

ZXYlc=[]
# zvals should be an NxM matrix, ordered [j,i], xvals should be length M [j], yvals is length N [i], following the convention of numpy (row index is first) or PIL (y index of pixel is first, equivalent to numpy). you should pass Zs.T if your Zs matrix is ordered x,y instead of y,x 
def contour(zvals,xvals,yvals,filename='',heatOrContour="heat",useLast=False,extras=[],**kwargs):


	# 1D DATASET PASSED, USE TRIANGULATION INTERPOLATION
	if isinstance(zvals[0],int) or isinstance(zvals[0],float):
		# Create grid values first.
		xi = np.linspace(min(xvals), max(xvals), 1000)
		yi = np.linspace(min(yvals), max(yvals), 1000)
		# Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
		triang = tri.Triangulation(xvals, yvals)
		interpolator = tri.LinearTriInterpolator(triang, zvals)
		Xi, Yi = np.meshgrid(xi, yi)
		zi = interpolator(Xi, Yi)
		#plt.tricontourf(xvals,yvals,zvals) ; plt.show()
		xvals=Xi ; yvals=Yi ; zvals=zi
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
		return
	# if we're saving files, use the "Agg" backend. (and save off whatever the current backend is, and restore it after we save it, to prevent messing up the user's python environment). if we're showing, just default to whatever the user's default is
	#print(xvals,yvals,len(xvals),len(yvals),np.shape(zvals))
	backend=matplotlib.get_backend()
	if len(filename)!=0:# and filename!="PLOTOBJ":
		matplotlib.use("Agg") # https://stackoverflow.com/questions/31156578/matplotlib-doesnt-release-memory-after-savefig-and-close


	global fig,ax,CS,cbar
	if not useLast: # TODO this was commented out before. idk why??? (we need it now to allow surface plots to be overlaid)
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
	if "zlim" in kwargs.keys():
		zlim=kwargs["zlim"] ; LB={True:LB,False:zlim[0]}[zlim[0] is None] ; UB={True:UB,False:zlim[1]}[zlim[1] is None]
	nticks=kwargs.get("nticks",10)
	#print(LB,UB)
	ticks=kwargs.get("zticks",np.linspace(LB,UB,nticks))
	
	# for heatmaps, you can use tricontourf, but that won't work for contours. need to follow https://matplotlib.org/stable/gallery/images_contours_and_fields/irregulardatagrid.html


	if heatOrContour in ["heat","both"]:
		CS=plt.contourf(xvals,yvals,zvals,levels=np.linspace(LB,UB,500),cmap=kwargs.get("cmap",defaultcmap))
		#print(np.amin(zvals),np.amax(zvals))
		cbar=plt.colorbar(ticks=ticks)
		for c in CS.collections:
			c.set_edgecolor("face")
		#	c.set_rasterized(True)
		nDecimals=max(0,int(1-np.floor(np.log(UB-LB)/np.log(10)))) # 0.35-0 --> -0.4559319556497244 --> -1 --> could be represented at 3.5e-1. if it was 35, we'd want 0 decimals. if it was 3.5 we'd want 1 decimal. -1 we want 2. 350, we still want 0 decimals
		#print(nDecimals)
		#nDecimals=max(nDecimals,0) # https://stackoverflow.com/questions/19986662/rounding-a-number-in-python-but-keeping-ending-zeros
		ticks=cbar.get_ticks()
		ticks=[ format(v,'.'+str(nDecimals)+'f') for v in ticks]
		cbar.ax.set_yticklabels(ticks)
		cbar.ax.set_title(kwargs.get("zlabel","ZTITLE"))
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
		plt.imshow(zvals,extent=(min(xvals),max(xvals),min(yvals),max(yvals)),cmap=kwargs.get("cmap",defaultcmap),aspect=aspect)
	if useLast:
		print(CS.collections)

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

def getContObjs():
	#print(id(ax),id(fig),id(plt)) # from gui.py > TDTR_fitting.py, plt ends up being shared between niceplot.py and nicecontour.py
	return ax,fig
def getCS():
	return CS
def setContObjs(a,f):
	global ax,fig ; ax,fig=a,f