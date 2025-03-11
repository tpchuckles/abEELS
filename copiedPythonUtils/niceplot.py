import matplotlib,os
import matplotlib.pyplot as plt #https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.html
import numpy as np
from cycler import cycler

# THESE ARE ALL YOUR USER-CONFIGURABLE SETTINGS.
# WHAT DEFAULT COLORS AND SYMBOLS DO YOU WANT TO ROTATE THROUGH? WHAT COLORMAP DO YOU WANT INTEGER COLOR VALUES TO REFER TO?
defCols=['k', 'g', 'b', 'r']
defMark=['o','^','s','P','d','X','v','+','x']
defaultMarkerSize = 10
defaultLineWidth = 2
cycle=cycler(color=defCols*len(defMark))+cycler(marker=defMark*len(defCols)) # https://github.com/matplotlib/cycler/issues/41
#cmap=matplotlib.cm.inferno
#cmap=matplotlib.cm.rainbow
cmap=matplotlib.cm.plasma
#cmap=matplotlib.cm.hsv
# WHAT GLOBAL TICK, AXIS, FONT SETTINGS DO YOU WANT?
params={ 'xtick':{ 'direction':'in', 'top':True , 'bottom':True , 'major.pad':7 }, # (rule #4)
	'ytick':{ 'direction':'in', 'left':True , 'right':True , 'major.pad':7 },
	'font':{ 'family':'arial', 'weight':'regular' , 'size':16 }, # (rule #5)
	'axes':{ 'autolimit_mode':'round_numbers' , "prop_cycle":cycle , 'titlesize':16 }, # (rule #6 Titile size should be the same size as the fontsize)
	#'svg':{'fonttype':'none'}, # ensure fonts in exported svgs are text objects wen you open them with inkscape: https://stackoverflow.com/questions/34387893/output-matplotlib-figure-to-svg-with-text-as-text-not-curves
	'figure':{ 'figsize':(8,6),'dpi':192 },
	'legend':{ 'edgecolor':'black' , 'fancybox':False }			# fancybox=false removes corner curvature
	#'mathtext':{'default': 'regular' }
	}
def setPlotRC(key,para):
	matplotlib.rc(key, **para)
for key in params:
	setPlotRC(key,params[key]) # replaces "matplotlib.rc(key, **params[key])"

markerSize = defaultMarkerSize ; lineWidth = defaultLineWidth

# RULES FOR PLOTTING:
# 1. plots must be generated in origin or matlab
# 2. Axes must be labeled with both title and unit
# 3. Border must appear on all edges of plot area
# 4. ticks must be set to "in"
# 5. the same font style (Arial) and sizes must be used for plot labels, units, tick number labels
# 6. the values of the ticks at the beginning and end of axes must be clearly labeled. "reasonable and legible" tick number labels must be chosen for ticks.
# 7. if linear scale, the axis values must start at (or span across) zero
# 8. all data and labels must be legible when printed in black and white. symbols should be clearly differentiable, legends should be clear and discernible from data being presented, data should be easily interpretable. 
# Answers: 2 is encouraged by giving the user dummy labels which should be in obvious need of attention. 3,4,5,6 are handled via rcparams (tick location, direction, font, autolimit_mode). 7 is handled via a manual set_xlim set_ylim right before incorporating kwargs. 8 is addressed in part by sensible default color/marker/linestyle choices. 

# ASK MATPLOTLIB WHAT MARKERS/LINESTYLES/COLOR SHORTCUTS ARE AVAILABLE
standardOptions={ "markers" : list(matplotlib.markers.MarkerStyle.markers.keys()) ,
		"linestyles" : list(matplotlib.lines.lineStyles.keys()) ,
		"colors" : list(matplotlib._color_data.BASE_COLORS.keys())+list(matplotlib._color_data.TABLEAU_COLORS.keys())+list(matplotlib.colors.CSS4_COLORS.keys()) }

#print(standardOptions)

# plot() is a basic wrapper for the likes of: "plt.plot(...);plt.xlabel(...);...
#  ;plt.show() or plt.savefig() etc". It also takes advantage of as many 
#  rcparams as possible to adhere to the rules for plotting, and takes 
#  matplotlib rcparams as optional arguments to pass in.
# Arguments:
# xs - a 2D list of x values: list of datasets, each dataset it itself a list
# ys - a 2D list of y values
# ye - a list matching ys: error value(s) for each dataset may be list or float
# markers - a 1D list describing how each dataset should be plotted. 
#	piece 1 denotes color: may be a standard color code (https://matplotlib.
#	  org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html), or a 2 digit int 
#	  from 00-99 (see float2rgb(c) below).
#	piece 2 denotes form: may be a marker symbol (https://matplotlib.org/sta
#	  ble/api/markers_api.html), or a linestyle symbol https://matplotlib.or
#	  g/stable/gallery/lines_bars_and_markers/linestyles.html, OR, "fN" 
#	  where N denotes that we're to fill between this and another dataset. 
#	if 2-character colors or 2-character form is used, separate with comma
#	examples: "ko" is a black o. "b-" is a blue line. "00f0" denotes a red 
#	  band will be filled in between this dataset and another which also has
#	  "f0". tab:blue,o"
#	hidden feature for markers: you can also pass a list with dicts, and
#	  we'll just pass the dict into the plot functions (plot, errorbar,
#	  fill_between) as kwargs. you'll need to manually set things like 
#	  "color":"red", but you'll also be able to include bonus parameters
#	  like linewidth, markersize, alpha, etc.
# filename - if a filename is provided, we'll save the plot instead of showing
# multiplot - 3 methods: split (two plots side-by-side sharing a y axis), 
#  stacked (two plot one-above-another sharing x axis), shared (one plot with a
#  shared x axis and 2 y axes). pass a tuple: (method,[indicesList]) where 
#  indicesList is a list of which dataset (xs,ys) goes on which plot
# Notes on expectations: function can handle basic plotting of points (ax.plot,
#  marker args, etc), lines (linestyle args), errorbars (ax.errorbar, with both
#  marker or linestyle args), and fill_between (you can plot shaded error bands
#  by passing a pair of datasets, y+ye and y-ye, and filling between them). 
def plot( xs, ys, xe='', ye='', markers='', labels='', filename='', multiplot='', fontsize='',extras=[], **kwargs):
	# if we're saving files, use the "Agg" backend. (if you don't, saving off, say, a thousand or so plots will fill up your memory, as some backends suck at garbage collection: https://stackoverflow.com/questions/31156578/matplotlib-doesnt-release-memory-after-savefig-and-close ). So we'll save off whatever the current backend is, and restore it after we save it, to prevent messing up the user's python environment. if we're showing, just default to whatever the user's default is. (other useful links: https://stackoverflow.com/questions/56656777/userwarning-matplotlib-is-currently-using-agg-which-is-a-non-gui-backend-so , https://stackoverflow.com/questions/55811545/importerror-cannot-load-backend-tkagg-which-requires-the-tk-interactive-fra )

	if ".csv" in filename:
		saveCSV(xs,ys,ye,kwargs.get("xlabel","xlabel"),kwargs.get("ylabel","ylabel"),labels,filename)
		return


	backend=matplotlib.get_backend()
	if len(filename)!=0:
		matplotlib.use("Agg") # 


		#data=
		#numpy.savetxt(kwargs["filename"], , delimiter=",")

	if "cmap" in kwargs.keys():
		global cmap
		cmap=kwargs["cmap"]

	global axs,frames,fig,markerSize,lineWidth ; axs=[] ; frames=[]
	lineWidth = kwargs.get("lw", defaultLineWidth) ; markerSize = kwargs.get("ms", defaultMarkerSize);
	
	# CREATE THE PLOT OBJECTS
	if len(multiplot)>0:
		axs,fig=genMultiAx(**multiplot)
	else:
		fig,ax=plt.subplots() ; axs.append(ax)
		
	ye=list(ye)+[0]*(len(xs)-len(ye)) ; xe=list(xe)+[0]*(len(xs)-len(xe))
	# PUT THE DATA ON THE PLOT
	for i in range(len(xs)):
		x=xs[i] ; y=ys[i] ; ax=axs[0]
		if len(axs)>1:
			ax=axs[multiplot["indices"][i]]
		# PROCESS MARKERS
		if len(markers)>i:
			kw=handleMarkers(markers[i]) # turns marker strings into a dict to pass to each plot function
			#print(kw)
			#print("kw",markers[i],"-->",kw)
		else:
			kw={"linestyle":'',"markersize":markerSize}
		# LABELS (for legend) ALSO GO IN KWARGS
		kw["label"]="dataset "+str(i+1)
		if len(labels)>i:
			kw["label"]=processText(labels[i])

		# ADD TO PLOT
		# ERRORBARS: each y entry may have a corresponding ye entry. the ye entry may be:
		# a) single value - all datapoints in the set get the same sized symmetric error bars: "10" --> +/-10 for each datapoint
		# b) list of values - each datapoint receives its own symmetric error bar: [1,2,3,2,4] --> first point +/- 1, second point +/-2 and so on
		# c) pair of single-value lists: - all datapoints in the set get the same sized asymettric error bars --> [[10],[20]] --> +20,-10 
		# d) pair of lists - each datapoint receives its own asymmetric error bar
		# BEWARE: if there are 2 points in a dataset and we receive a pair of ye values: it is undefined whether that intends (b) or (c).
		#if len(ye)>i:
		#	print("YERROR:",ye[i])
			"""
			if islist(ye[i]) and len(ye[i])==2 and len(ys[i])==2 and not islist(ye[i][0]):
				print("WARNING FOR ERROR SET "+str(i)+": if a y error entry has length==2 AND the dataset itself has length==2: it is unclear if the user intends for: each datapoint receives a symmetric error bar, OR, the full dataset receives asymmetric error bars (same for each point). if you desire the latter (or wish to suppress this error, please fully-define a 2x2 matrix: [lowerboundslist,upperboundslist]. WE WILL ASSUME YOU MEAN POINT-SPECIFIC, NOT ASYMETRIC")
				ye[i]=np.asarray(ye[i])[None,:]*np.ones(2)[:,None]
			if islist(ye[i]) and len(ye[i])==2:
				if islist(ye[i][0]):
					ye[i]=np.asarray(ye[i])
				else:
					ye[i]=np.asarray(ye[i]).reshape((2,1))
			"""
		#	ax.errorbar(xs[i], ys[i], yerr=ye[i], capsize=2, **kw)


		hasYe=(islist(ye[i]) or ye[i]!=0) ; hasXe=(islist(xe[i]) or xe[i]!=0)
		#print(hasYe,hasXe,ye[i],xe[i])

		#print("niceplot",i,xs[i],ys[i],xe[i],ye[i])

		if hasYe and hasXe: #len(ye)>i or len(xe)>i:
			ax.errorbar(xs[i], ys[i], yerr=ye[i], xerr=xe[i], capsize=2, **kw)
		elif hasYe:
			ax.errorbar(xs[i], ys[i], yerr=ye[i], capsize=2, **kw)
		elif hasXe:
			ax.errorbar(xs[i], ys[i], xerr=xe[i], capsize=2, **kw)
		elif "fill" in kw.keys():					# FILL BETWEEN
			for j in range(i,len(xs)):				# check all other datasets (all after this one!)
				if (i==j) or (j>=len(markers)):
					continue
				kw2=handleMarkers(markers[j])
				if "fill" in kw2.keys() and kw2["fill"]==kw["fill"]:
					del kw["fill"]				# we just used this to understand if handleMarkers wanted
					ax.fill_between(xs[i], ys[i], ys[j], **kw)	# us to do filling. so delete it
					break
		else:								# OR BASIC PLOT WITH POINTS OR LINES
			ax.plot(xs[i], ys[i], **kw)
		#print(kw)
	for ax in axs:
		if len(labels)==0 or max( [ len(lb) for lb in labels ] ) > 0:
			l=ax.legend()
			frames.append( l.get_frame() )
	#plt.legend()

	for i,ax in enumerate(axs):
		for key,getter,setter in zip(["ylim","xlim"],[ax.get_ylim,ax.get_xlim],[ax.set_ylim,ax.set_xlim]):
			passedlims=kwargs.get(key,[None,None]) ; curlims=list(getter()) ; scale=kwargs.get(key[0]+"scale","linear")
			if passedlims[0] is not None and isinstance(passedlims[0],list): # is using multiplot, we should allow both lims for axes!
				passedlims=passedlims[i]
			# rule: linear plots should include zero (don't "cheat" by zooming into only slightly-varying data!). this is overridable by including "nonzero" in the lims passed, OR, by specifying actual values 
			if scale=="linear" and "nonzero" not in passedlims: # read in existing limits, check if zero is outside their bounds
				curlims.append(0) ; curlims=[min(curlims),max(curlims)] ; setter(curlims)
			passedlims=[ None if l=="nonzero" else l for l in passedlims ] # remove "nonzero" from passed
			if True in [ p is not None for p in passedlims ]:
	#		if "log" not in kwargs.get(key[0]+"scale",""):
				setter(passedlims) # and set passed (may have been defaulted to Nones, which does nothing)
			#print(setter,passedlims)

	#print("niceplot > plot: xscale",kwargs.get("xscale",""),"xlim",axs[0].get_xlim())

	# HANDLE OTHER ARGUMENTS. Each allowable is mapped to a function, and a default
	#if kwargs.get("xscale","linear")!="log" and "nonzero" not in kwargs.get("xlim",["",""]):
	#	xlim=list(axs[0].get_xlim()) ; xlim.append(0) ; xlim=[min(xlim),max(xlim)] ; axs[0].set_xlim(xlim)
	#if kwargs.get("yscale","linear")!="log" and "nonzero" not in kwargs.get("ylim",["",""]):
	#	ylim=list(axs[0].get_ylim()) ; ylim.append(0) ; ylim=[min(ylim),max(ylim)] ; axs[0].set_ylim(ylim)
		title=kwargs.get("title","TITLE")
		if isinstance(title,list):
			title=title[i]
		ax.set_title( processText( title ) ) # get "title" kw from kwargs, defaulting to "title". pass through
		xlb=kwargs.get("xlabel","XLABEL")
		if isinstance(xlb,list):
			xlb=xlb[i]
		ax.set_xlabel( processText( xlb ) ) # processText, then set as title. and so on for xlabel,ylabel
		ylb=kwargs.get("ylabel","YLABEL")
		if isinstance(ylb,list):
			ylb=ylb[i]
		ax.set_ylabel( processText(ylb) )	
		#ax.set_ylabel( processText( kwargs.get("ylabel","YLABEL") ) ) 
		#if "xlim" in kwargs.keys():
		#	axs[0].set_xlim( kwargs.get("xlim") )
		#if "ylim" in kwargs.keys():
		#	ylim=kwargs.get("ylim") ; ylim=[ None if l=="nonzero" else l for l in ylim ]
		#	print(ylim)
		#	axs[0].set_ylim( ylim )
		ax.set_xscale( kwargs.get("xscale","linear") )
		ax.set_yscale( kwargs.get("yscale","linear") )
	setFace( kwargs.get("facecolor","white") )
	if "minorticks" in kwargs.keys():
		if kwargs["minorticks"]:
			plt.minorticks_on()
	if "figsize" in kwargs.keys():				# most are okay receiving None, except for set_size_inches
		fig.set_size_inches( kwargs.get("figsize") )
	if "fontsize" in kwargs.keys(): # https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
		#matplotlib.rc('font', **{'size': kwargs.get("fontsize") }) # TODO if you do this via rcparams, it "permanently" sets it for all following plots! not what we want
		#axs[0].set_font_size( kwargs.get("fontsize") )
		#axs[0].title.set_fontsize( kwargs.get("fontsize") )
		#matplotlib.rc('axes', **{'titlesize':kwargs.get("fontsize") })
		for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
			ax.get_xticklabels() + ax.get_yticklabels() + axs[0].get_legend().get_texts()):
			#item.set_fontsize(20)
			#print(item)
			item.set_fontsize( kwargs.get("fontsize"))
		#for item in axs[0].get_legend().get_texts():
		#	print(item)
		#	item.set_fontsize( kwargs.get("fontsize"))

#'font':{'family':'arial', 'weight':'regular' , 'size':16}
	#		key	   set function   default val,  pre-function
#	funcLookup={ 	"title" : (axs[0].set_title , "TITLE" , processText) , 
#			"xlabel": (axs[0].set_xlabel, "XLABEL", processText) ,
#			"ylabel": (axs[0].set_ylabel, "YLABEL", processText) ,
#			"xlim"  : (axs[0].set_xlim  , None    , None       ) , 
#			"ylim"  : (axs[0].set_ylim  , None    , None       ) , 
#			"xscale": (axs[0].set_xscale, "linear", None       ) , 
#			"yscale": (axs[0].set_yscale, "linear", None       ) ,
#			"facecolor": (setFace       , "white" , None       ) ,
#			"figsize": (fig.set_size_inches, None , None       ) }
#	
#	for k in funcLookup.keys():		# for every allowed funct/arg
#		f,d,f2=funcLookup[k]
#		if k in kwargs.keys():		# if the user passed it, override the default with that
#			d=kwargs[k]		
#		if d is not None:		# if there's a value, set it
#			if f2 is not None:
#				d=f2(d)
#			f(d)			# replaces things like "ax.set_title(value)"

	for extra in extras:
		extra(axs,fig)
	#if filename=="PLOTOBJ":
	#	return axs[0],fig
	if len(filename)>0:
		if ".svg" in filename:
			matplotlib.rc("svg", **{'fonttype':'none'}) # ensures text in svg files is saved as text (for later editing)
		directory="/".join(filename.replace("\\","/").split("/")[:-1]) # path/to/file.png --> path/to
		if len(directory)>0: # possible image file was destined for THIS directory (""), so skip if that's the case
			os.makedirs(directory,exist_ok=True)
		plt.savefig(filename)
		plt.close() # without this, a later "show" will blast you by showing all previously-saved plots: https://mldoodles.com/matplotlib-save-plot-but-dont-show/
	else:
		plt.show()
	matplotlib.use(backend)

def islist(val): # WHY? if val is a numpy array, then isinstance(val,list) will return False!
	if isinstance(val,(float,int,str)):
		return False
	return True

# lots of valid options for markers: "k" (black, default in a symbol), "k-" (black line, mpl standard), "k,-" (comma-separated), "tab:blue" (blue, default in a symbol), "tab:blue,-" (blue line), "o" (default in the color, o symbol), "-" (defailt in the color, line symbol)
def handleMarkers(m):
	kw={}
	#m=markers[i]
	if isinstance(m,str):
		if len(m)==0:
			return {"linestyle":'',"markersize":markerSize}
		if "," in m:						# COMMA-SEPARATED, color,form
			c,f=m.split(",")
		elif len(m)==2:						# OR DEFAULT TO char1,char2
			c,f=m[0],m[1:]					# TODO bug: "99" for purple --> "9","9". or "-." should be dash-dot
		else:							# length1, or n, try as both color and form
			c,f=m,m

		if "f" in f:						# FILL BETWEEN, e.g. "k,fABC" black, fill this and other "fABC"
			kw["fill"]=f#.replace("f","")			# (why not pass indices? this allows safe re-ordering of dsets)
			kw["linewidth"]=0
			kw["alpha"]=.2
		elif f in standardOptions["markers"]:			# MARKER, NOT LINESTYLE, e.g. "k.", dot
			kw["marker"]=f ; kw["linestyle"]=''; kw["markersize"]=markerSize
		elif f in standardOptions["linestyles"]:		# LINESTYLE, NOT MARKER, e.g. "b:", dotted line
			kw["marker"]='' ; kw["linestyle"]=f; kw["linewidth"]=lineWidth
		else:							# NEITHER, let mpl default in markers, suppress line
			kw["linestyle"]='' ; kw["markersize"]=markerSize

		if c in standardOptions["colors"] or c[0]=="#":			# STANDARD COLOR CODE
			kw["color"]=c
		else:
			for char in c:
				if char not in "0123456789":		# INVALID COLOR, we'll let mpl default in color
					break
			else:						# 2 DIGIT COLOR INSTEAD OF SHORTCUT CODE
				kw["color"]=float2rgb(int(c)/100)
		#print("m",m,"kw",kw)
		return kw
	return m

# usage: "plot(Xs,Ys,markers=rainbow(len(Xs),":"))"
def rainbow(n,m='-'):
	return [ str(int(i/n*100))+","+m for i in range(n) ]

def genMultiAx(method,indices,scaling=''):
	axs=[] ; nPlots=len(set(indices))
	if len(scaling)==0:
		scaling=[1]*nPlots
	if method=="split": # break in x axis? https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/broken_axis.html
		fig,axs=plt.subplots(1,nPlots,sharey=True,gridspec_kw={"wspace":0,'width_ratios':scaling}) # 1 row, 2 columns, 0 spacing between the two, using ratios for sizing
		axs[0].spines['right'].set_visible(False) #; axs[1].spines['left'].set_visible(False) # get rid of extra spline and ticks
		axs[0].yaxis.tick_left() ; axs[1].yaxis.tick_right()
	if method=="stacked":
		fig,axs=plt.subplots(nPlots,sharex=True,gridspec_kw={"hspace":0,'height_ratios':scaling})
	if method=="shared":
		fig,ax1=plt.subplots() ; ax2=ax1.twinx() ; axs=[ax1,ax2]
	return axs,fig

def float2rgb(c): # float, 0 to 1
	#import matplotlib.cm as cm
	#return cm.inferno(1-c)[:3]		
	#return cm.rainbow(1-c)[:3]		
	return cmap(1-c)[:3]		

def processText(text): # automagically recognize things like "W m^-2 K^-1" and turn into the appropriate form: "W m$^{2}$ K$^{-1}$"
	terms=text.split()							# "TBC (W m^-2 K^-1)" --> ["TBC","(W","m^-2","K^-1)"]
	for i,t in enumerate(terms):
		if "^" in t:
			a,b=t.split("^")					# "K^-1)" --> ["K","-1)"]
			b1="".join( [c for c in b if c in "-0123456789.+" ] )	# only "-1" should be superscripted. ion doses may be "C^+" too, for example
			b2="".join( [c for c in b if c not in "-0123456789.+" ] ) # ")" should not be superscripted
			terms[i]=a+"$^{"+b1+"}$"+b2				# reassembled: "K$^{-1}$)"
			#terms[i]=a+"\\textsuperscript{"+b1+"}"+b2
	#print(" ".join(terms))
	return " ".join(terms)

def setFace(fc): # https://stackoverflow.com/questions/19863368/how-to-change-the-legend-edgecolor-and-facecolor-in-matplotlib
	for ax,fr in zip(axs,frames):
		ax.set_facecolor(fc)
		fr.set_facecolor(fc)

def setLegendLW(lw):
	for fr in frames:
		fr.set_linewidth(lw)

def getPlotObjs():
	#print(id(axs[0]),id(fig),id(plt))
	return axs[0],fig

def invertPlotColors(axs,fig):
	axs[0].set_facecolor("black")			# inside area of plot --> black
	fig.set_facecolor("black")			# outside area of plot --> black
	for s in ["bottom","top","left","right"]:	# border lines around plot --> white
		axs[0].spines[s].set_color("white")
	axs[0].xaxis.label.set_color('white')		# x axis label text --> white
	axs[0].tick_params(axis='x', colors='white')	# x axis tick marks --> white
	axs[0].yaxis.label.set_color('white')
	axs[0].tick_params(axis='y', colors='white')
	axs[0].title.set_color('white')
	leg=axs[0].legend()				# retreive legend from the axes
	for text in leg.get_texts():
		text.set_color("white")			# each line of text on the legend --> white
	frame=leg.get_frame()
	frame.set_facecolor('black')			# inside area of legend --> black
	frame.set_edgecolor('white')			# legend border lines --> white

# this function returns an extras function! use it to add color-gradient lines (or alpha intensity gradient lines, since matplotlib's alphas arg doesn't accept lists of values). note: if you pass an empty Xs and Ys lists, the plot extents will not be auto-set properly, so you might want to plot the datasets invisibly first (e.g., in white) too. e.g., try "plot([x],[y],markers=['w-'],extras=[addGradientLine(x,y,cs)])" where x,y,cs are single lists of values, with cs being the color. cs can either be a list of values 0-1 (passed to the colormap) or an nx3 or nx4 array (RGB(A) colors)
def addGradientLine(x,y,cs,nth=1):
	def extra(axs,fig):
		from matplotlib.collections import LineCollection
		points=np.asarray([x,y]).T.reshape(-1,1,2)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)
		lc = LineCollection(segments[::nth],linestyle="-",lw=2)
		if len(np.shape(cs))>1:
			lc.set(color=cs[::nth])
		else:
			lc.set(cmap=cmap)
			lc.set_array(cs[::nth])
		axs[0].add_collection(lc)
	return extra

def saveCSV(xs,ys,ye,xlabel,ylabel,labels,filename):
	#print("saveCSV","xs",xs,"ys",ys,"ye",ye,"xlabel",xlabel,"ylabel",ylabel,"labels",labels,"filename",filename)
	#xs=[ np.asarray(x) for x in xs ] ; ys=[ np.asarray(y) for y in ys ] # each element must be numpy array in order for astype to work
	ye=list(ye) ; ye=ye+['']*(len(xs)-len(ye)) #; ye
	maxlen=max( [ len(x) for x in xs ] )
	tosave=np.zeros(( maxlen , len(xs)*3 ) ).astype(str) ; tosave[:,:]='' # 3 columns per dataset (x,y,padding), fields will hold strings
	#print(ye)
	for i,(x,y,e) in enumerate(zip(xs,ys,ye)):
		if len(x)==0:
			continue
		x,y=np.asarray(x),np.asarray(y)
		tosave[:len(x),i*3]=x.astype(str) ; tosave[:len(x),i*3+1]=y.astype(str) # set columns from datasets
		#print(type(e))
		if isinstance(e,(int,float)): # error bars can be a constant number (same errorbar for each datapoint)
			tosave[:len(x),i*3+2]=str(e)
		elif len(e)==len(x): # or it can be a different value for each datapoint
			e=np.asarray(e)
			tosave[:len(e),i*3+2]=e.astype(str)

	header=",,,".join(labels)+"\n"+"".join( [xlabel+","+ylabel+",,"]*len(xs) ) # include datalabels and x,y labels
	np.savetxt(filename,tosave,header=header,delimiter=',',fmt="%s")

def readCSV(filename):
	lines=open(filename).readlines()
	labels=lines[0][2:].split(",")[::3]
	xlabel,ylabel=lines[1][2:].split(",")[:2]
	xs,ys,ye=[],[],[]
	for i in range(len(labels)):
		xs.append([]) ; ys.append([]) ; ye.append([])
		for l in lines[2:]:
			x,y,e=l.split(",")[i*3:i*3+3]
			if len(x)==0:
				break
			xs[-1].append(float(x)) ; ys[-1].append(float(y))
			if len(e.strip())>0: # .strip handles potential trailing '\n'
				ye[-1].append(float(e))
			else:
				ye[-1].append(0)
	return xs,ys,ye,xlabel,ylabel,labels

# WHY ARE WE CODING UP OUR OWN VERSION OF GAUSSIAN BLUR? (instead of using scipy.ndimage.gaussian_filter1d etc)?
# I like being able to use the same gaussian function (including gaussian definition of beam radius) used for our thermal experiments (see TDTR_fitting.py and FD3D.py)
# TODO scipy.ndimage.gaussian_filter can do different blurWidth for each axis, can do 1D 2D etc, and is way way way way way faster than doing our own convolution. we should figure out what the equivalence is and just use that. https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
def gaussianBlur(data,blurWidth):
	if len(data)==2:
		xs,ys=data ; ND=2 ; from scipy.signal import convolve
	elif len(data)==3:
		zs,xs,ys=data ; ND=3 ; from scipy.signal import convolve2d
	def gauss2D(xs,w):
		return 	np.exp(-2*(xs-np.mean(xs))**2/w**2) # UNITS: W/m²
	def gauss3D(xs,ys,w):
		radii=np.sqrt( (xs[None,:]-np.mean(xs))**2+(ys[:,None]-np.mean(ys))**2 )
		return np.exp(-2*radii**2/w**2)
	if ND==2:
		gauss=gauss2D(xs,blurWidth)
		blurredY=convolve(ys, gauss, mode="same") # BEWARE: padding can help with edge effects. gonna take longer though...
		#n=len(ys) ; padded=np.zeros(3*n) ; padded[:n]=ys[::-1] ; padded[n:2*n]=ys[:] ; padded[2*n:]=ys[::-1]
		#blurredY=convolve(padded, gauss, mode="same")
		#blurredY=blurredY[n:2*n]
		blurredY *= np.trapz(ys)/np.trapz(blurredY)
		return blurredY
	if ND==3:
		gauss=gauss3D(xs,ys,blurWidth)
		#blurredZ=convolve2d(zs, gauss, mode='same',boundary='wrap') # BEWARE, wrapping might be inappropriate (left/right edges may not match)
		ny,nx=np.shape(zs) ; padded=np.zeros((3*ny,3*nx))
		for i in range(3):
			for j in range(3):
				# "unfold" from center tile (edge tiles are reflected. use [::-1] notation to populate with reversed matrix)
				dx=(i%2)*2-1 ; dy=(j%2)*2-1
				padded[ j*ny:(j+1)*ny , i*nx:(i+1)*nx ] = zs[ ::dy , ::dx ]
		#from nicecontour import contour ; contour(padded,np.arange(0,3*nx),np.arange(0,3*ny))
		blurredZ=convolve2d(padded, gauss, mode='same')
		blurredZ=blurredZ[ny:2*ny,nx:2*nx]
		blurredZ *= np.sum(zs)/np.sum(blurredZ)
		return blurredZ






