
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: This is a procedural interface to the matplotlib object-oriented
3: plotting library.
4: 
5: The following plotting commands are provided; the majority have
6: MATLAB |reg| [*]_ analogs and similar arguments.
7: 
8: .. |reg| unicode:: 0xAE
9: 
10: _Plotting commands
11:   acorr     - plot the autocorrelation function
12:   annotate  - annotate something in the figure
13:   arrow     - add an arrow to the axes
14:   axes      - Create a new axes
15:   axhline   - draw a horizontal line across axes
16:   axvline   - draw a vertical line across axes
17:   axhspan   - draw a horizontal bar across axes
18:   axvspan   - draw a vertical bar across axes
19:   axis      - Set or return the current axis limits
20:   autoscale - turn axis autoscaling on or off, and apply it
21:   bar       - make a bar chart
22:   barh      - a horizontal bar chart
23:   broken_barh - a set of horizontal bars with gaps
24:   box       - set the axes frame on/off state
25:   boxplot   - make a box and whisker plot
26:   violinplot - make a violin plot
27:   cla       - clear current axes
28:   clabel    - label a contour plot
29:   clf       - clear a figure window
30:   clim      - adjust the color limits of the current image
31:   close     - close a figure window
32:   colorbar  - add a colorbar to the current figure
33:   cohere    - make a plot of coherence
34:   contour   - make a contour plot
35:   contourf  - make a filled contour plot
36:   csd       - make a plot of cross spectral density
37:   delaxes   - delete an axes from the current figure
38:   draw      - Force a redraw of the current figure
39:   errorbar  - make an errorbar graph
40:   figlegend - make legend on the figure rather than the axes
41:   figimage  - make a figure image
42:   figtext   - add text in figure coords
43:   figure   - create or change active figure
44:   fill     - make filled polygons
45:   findobj  - recursively find all objects matching some criteria
46:   gca      - return the current axes
47:   gcf      - return the current figure
48:   gci      - get the current image, or None
49:   getp      - get a graphics property
50:   grid     - set whether gridding is on
51:   hist     - make a histogram
52:   ioff     - turn interaction mode off
53:   ion      - turn interaction mode on
54:   isinteractive - return True if interaction mode is on
55:   imread   - load image file into array
56:   imsave   - save array as an image file
57:   imshow   - plot image data
58:   legend   - make an axes legend
59:   locator_params - adjust parameters used in locating axis ticks
60:   loglog   - a log log plot
61:   matshow  - display a matrix in a new figure preserving aspect
62:   margins  - set margins used in autoscaling
63:   pause    - pause for a specified interval
64:   pcolor   - make a pseudocolor plot
65:   pcolormesh - make a pseudocolor plot using a quadrilateral mesh
66:   pie      - make a pie chart
67:   plot     - make a line plot
68:   plot_date - plot dates
69:   plotfile  - plot column data from an ASCII tab/space/comma delimited file
70:   pie      - pie charts
71:   polar    - make a polar plot on a PolarAxes
72:   psd      - make a plot of power spectral density
73:   quiver   - make a direction field (arrows) plot
74:   rc       - control the default params
75:   rgrids   - customize the radial grids and labels for polar
76:   savefig  - save the current figure
77:   scatter  - make a scatter plot
78:   setp      - set a graphics property
79:   semilogx - log x axis
80:   semilogy - log y axis
81:   show     - show the figures
82:   specgram - a spectrogram plot
83:   spy      - plot sparsity pattern using markers or image
84:   stem     - make a stem plot
85:   subplot  - make one subplot (numrows, numcols, axesnum)
86:   subplots - make a figure with a set of (numrows, numcols) subplots
87:   subplots_adjust - change the params controlling the subplot positions of current figure
88:   subplot_tool - launch the subplot configuration tool
89:   suptitle   - add a figure title
90:   table    - add a table to the plot
91:   text     - add some text at location x,y to the current axes
92:   thetagrids - customize the radial theta grids and labels for polar
93:   tick_params - control the appearance of ticks and tick labels
94:   ticklabel_format - control the format of tick labels
95:   title    - add a title to the current axes
96:   tricontour - make a contour plot on a triangular grid
97:   tricontourf - make a filled contour plot on a triangular grid
98:   tripcolor - make a pseudocolor plot on a triangular grid
99:   triplot - plot a triangular grid
100:   xcorr   - plot the autocorrelation function of x and y
101:   xlim     - set/get the xlimits
102:   ylim     - set/get the ylimits
103:   xticks   - set/get the xticks
104:   yticks   - set/get the yticks
105:   xlabel   - add an xlabel to the current axes
106:   ylabel   - add a ylabel to the current axes
107: 
108:   autumn - set the default colormap to autumn
109:   bone   - set the default colormap to bone
110:   cool   - set the default colormap to cool
111:   copper - set the default colormap to copper
112:   flag   - set the default colormap to flag
113:   gray   - set the default colormap to gray
114:   hot    - set the default colormap to hot
115:   hsv    - set the default colormap to hsv
116:   jet    - set the default colormap to jet
117:   pink   - set the default colormap to pink
118:   prism  - set the default colormap to prism
119:   spring - set the default colormap to spring
120:   summer - set the default colormap to summer
121:   winter - set the default colormap to winter
122: 
123: _Event handling
124: 
125:   connect - register an event handler
126:   disconnect - remove a connected event handler
127: 
128: _Matrix commands
129: 
130:   cumprod   - the cumulative product along a dimension
131:   cumsum    - the cumulative sum along a dimension
132:   detrend   - remove the mean or besdt fit line from an array
133:   diag      - the k-th diagonal of matrix
134:   diff      - the n-th differnce of an array
135:   eig       - the eigenvalues and eigen vectors of v
136:   eye       - a matrix where the k-th diagonal is ones, else zero
137:   find      - return the indices where a condition is nonzero
138:   fliplr    - flip the rows of a matrix up/down
139:   flipud    - flip the columns of a matrix left/right
140:   linspace  - a linear spaced vector of N values from min to max inclusive
141:   logspace  - a log spaced vector of N values from min to max inclusive
142:   meshgrid  - repeat x and y to make regular matrices
143:   ones      - an array of ones
144:   rand      - an array from the uniform distribution [0,1]
145:   randn     - an array from the normal distribution
146:   rot90     - rotate matrix k*90 degress counterclockwise
147:   squeeze   - squeeze an array removing any dimensions of length 1
148:   tri       - a triangular matrix
149:   tril      - a lower triangular matrix
150:   triu      - an upper triangular matrix
151:   vander    - the Vandermonde matrix of vector x
152:   svd       - singular value decomposition
153:   zeros     - a matrix of zeros
154: 
155: _Probability
156: 
157:   normpdf   - The Gaussian probability density function
158:   rand      - random numbers from the uniform distribution
159:   randn     - random numbers from the normal distribution
160: 
161: _Statistics
162: 
163:   amax      - the maximum along dimension m
164:   amin      - the minimum along dimension m
165:   corrcoef  - correlation coefficient
166:   cov       - covariance matrix
167:   mean      - the mean along dimension m
168:   median    - the median along dimension m
169:   norm      - the norm of vector x
170:   prod      - the product along dimension m
171:   ptp       - the max-min along dimension m
172:   std       - the standard deviation along dimension m
173:   asum      - the sum along dimension m
174:   ksdensity - the kernel density estimate
175: 
176: _Time series analysis
177: 
178:   bartlett  - M-point Bartlett window
179:   blackman  - M-point Blackman window
180:   cohere    - the coherence using average periodiogram
181:   csd       - the cross spectral density using average periodiogram
182:   fft       - the fast Fourier transform of vector x
183:   hamming   - M-point Hamming window
184:   hanning   - M-point Hanning window
185:   hist      - compute the histogram of x
186:   kaiser    - M length Kaiser window
187:   psd       - the power spectral density using average periodiogram
188:   sinc      - the sinc function of array x
189: 
190: _Dates
191: 
192:   date2num  - convert python datetimes to numeric representation
193:   drange    - create an array of numbers for date plots
194:   num2date  - convert numeric type (float days since 0001) to datetime
195: 
196: _Other
197: 
198:   angle     - the angle of a complex array
199:   griddata  - interpolate irregularly distributed data to a regular grid
200:   load      - Deprecated--please use loadtxt.
201:   loadtxt   - load ASCII data into array.
202:   polyfit   - fit x, y to an n-th order polynomial
203:   polyval   - evaluate an n-th order polynomial
204:   roots     - the roots of the polynomial coefficients in p
205:   save      - Deprecated--please use savetxt.
206:   savetxt   - save an array to an ASCII file.
207:   trapz     - trapezoidal integration
208: 
209: __end
210: 
211: .. [*] MATLAB is a registered trademark of The MathWorks, Inc.
212: 
213: 
214: '''
215: from __future__ import (absolute_import, division, print_function,
216:                         unicode_literals)
217: 
218: import six
219: 
220: import sys, warnings
221: 
222: from matplotlib.cbook import (
223:     flatten, exception_to_str, silent_list, iterable, dedent)
224: 
225: import matplotlib as mpl
226: # make mpl.finance module available for backwards compatability, in case folks
227: # using pylab interface depended on not having to import it
228: with warnings.catch_warnings():
229:     warnings.simplefilter("ignore")  # deprecation: moved to a toolkit
230:     import matplotlib.finance
231: 
232: from matplotlib.dates import (
233:     date2num, num2date, datestr2num, strpdate2num, drange, epoch2num,
234:     num2epoch, mx2num, DateFormatter, IndexDateFormatter, DateLocator,
235:     RRuleLocator, YearLocator, MonthLocator, WeekdayLocator, DayLocator,
236:     HourLocator, MinuteLocator, SecondLocator, rrule, MO, TU, WE, TH, FR,
237:     SA, SU, YEARLY, MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY, SECONDLY,
238:     relativedelta)
239: 
240: # bring all the symbols in so folks can import them from
241: # pylab in one fell swoop
242: 
243: ## We are still importing too many things from mlab; more cleanup is needed.
244: 
245: from matplotlib.mlab import (
246:     amap, base_repr, binary_repr, bivariate_normal, center_matrix, csv2rec,
247:     demean, detrend, detrend_linear, detrend_mean, detrend_none, dist,
248:     dist_point_to_segment, distances_along_curve, entropy, exp_safe,
249:     fftsurr, find, frange, get_sparse_matrix, get_xyz_where, griddata,
250:     identity, inside_poly, is_closed_polygon, ispower2, isvector, l1norm,
251:     l2norm, log2, longest_contiguous_ones, longest_ones, movavg, norm_flat,
252:     normpdf, path_length, poly_below, poly_between, prctile, prctile_rank,
253:     rec2csv, rec_append_fields, rec_drop_fields, rec_join, rk4, rms_flat,
254:     segments_intersect, slopes, stineman_interp, vector_lengths,
255:     window_hanning, window_none)
256: 
257: from matplotlib import cbook, mlab, pyplot as plt
258: from matplotlib.pyplot import *
259: 
260: from numpy import *
261: from numpy.fft import *
262: from numpy.random import *
263: from numpy.linalg import *
264: 
265: import numpy as np
266: import numpy.ma as ma
267: 
268: # don't let numpy's datetime hide stdlib
269: import datetime
270: 
271: # This is needed, or bytes will be numpy.random.bytes from
272: # "from numpy.random import *" above
273: bytes = six.moves.builtins.bytes
274: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_114545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, (-1)), 'unicode', u'\nThis is a procedural interface to the matplotlib object-oriented\nplotting library.\n\nThe following plotting commands are provided; the majority have\nMATLAB |reg| [*]_ analogs and similar arguments.\n\n.. |reg| unicode:: 0xAE\n\n_Plotting commands\n  acorr     - plot the autocorrelation function\n  annotate  - annotate something in the figure\n  arrow     - add an arrow to the axes\n  axes      - Create a new axes\n  axhline   - draw a horizontal line across axes\n  axvline   - draw a vertical line across axes\n  axhspan   - draw a horizontal bar across axes\n  axvspan   - draw a vertical bar across axes\n  axis      - Set or return the current axis limits\n  autoscale - turn axis autoscaling on or off, and apply it\n  bar       - make a bar chart\n  barh      - a horizontal bar chart\n  broken_barh - a set of horizontal bars with gaps\n  box       - set the axes frame on/off state\n  boxplot   - make a box and whisker plot\n  violinplot - make a violin plot\n  cla       - clear current axes\n  clabel    - label a contour plot\n  clf       - clear a figure window\n  clim      - adjust the color limits of the current image\n  close     - close a figure window\n  colorbar  - add a colorbar to the current figure\n  cohere    - make a plot of coherence\n  contour   - make a contour plot\n  contourf  - make a filled contour plot\n  csd       - make a plot of cross spectral density\n  delaxes   - delete an axes from the current figure\n  draw      - Force a redraw of the current figure\n  errorbar  - make an errorbar graph\n  figlegend - make legend on the figure rather than the axes\n  figimage  - make a figure image\n  figtext   - add text in figure coords\n  figure   - create or change active figure\n  fill     - make filled polygons\n  findobj  - recursively find all objects matching some criteria\n  gca      - return the current axes\n  gcf      - return the current figure\n  gci      - get the current image, or None\n  getp      - get a graphics property\n  grid     - set whether gridding is on\n  hist     - make a histogram\n  ioff     - turn interaction mode off\n  ion      - turn interaction mode on\n  isinteractive - return True if interaction mode is on\n  imread   - load image file into array\n  imsave   - save array as an image file\n  imshow   - plot image data\n  legend   - make an axes legend\n  locator_params - adjust parameters used in locating axis ticks\n  loglog   - a log log plot\n  matshow  - display a matrix in a new figure preserving aspect\n  margins  - set margins used in autoscaling\n  pause    - pause for a specified interval\n  pcolor   - make a pseudocolor plot\n  pcolormesh - make a pseudocolor plot using a quadrilateral mesh\n  pie      - make a pie chart\n  plot     - make a line plot\n  plot_date - plot dates\n  plotfile  - plot column data from an ASCII tab/space/comma delimited file\n  pie      - pie charts\n  polar    - make a polar plot on a PolarAxes\n  psd      - make a plot of power spectral density\n  quiver   - make a direction field (arrows) plot\n  rc       - control the default params\n  rgrids   - customize the radial grids and labels for polar\n  savefig  - save the current figure\n  scatter  - make a scatter plot\n  setp      - set a graphics property\n  semilogx - log x axis\n  semilogy - log y axis\n  show     - show the figures\n  specgram - a spectrogram plot\n  spy      - plot sparsity pattern using markers or image\n  stem     - make a stem plot\n  subplot  - make one subplot (numrows, numcols, axesnum)\n  subplots - make a figure with a set of (numrows, numcols) subplots\n  subplots_adjust - change the params controlling the subplot positions of current figure\n  subplot_tool - launch the subplot configuration tool\n  suptitle   - add a figure title\n  table    - add a table to the plot\n  text     - add some text at location x,y to the current axes\n  thetagrids - customize the radial theta grids and labels for polar\n  tick_params - control the appearance of ticks and tick labels\n  ticklabel_format - control the format of tick labels\n  title    - add a title to the current axes\n  tricontour - make a contour plot on a triangular grid\n  tricontourf - make a filled contour plot on a triangular grid\n  tripcolor - make a pseudocolor plot on a triangular grid\n  triplot - plot a triangular grid\n  xcorr   - plot the autocorrelation function of x and y\n  xlim     - set/get the xlimits\n  ylim     - set/get the ylimits\n  xticks   - set/get the xticks\n  yticks   - set/get the yticks\n  xlabel   - add an xlabel to the current axes\n  ylabel   - add a ylabel to the current axes\n\n  autumn - set the default colormap to autumn\n  bone   - set the default colormap to bone\n  cool   - set the default colormap to cool\n  copper - set the default colormap to copper\n  flag   - set the default colormap to flag\n  gray   - set the default colormap to gray\n  hot    - set the default colormap to hot\n  hsv    - set the default colormap to hsv\n  jet    - set the default colormap to jet\n  pink   - set the default colormap to pink\n  prism  - set the default colormap to prism\n  spring - set the default colormap to spring\n  summer - set the default colormap to summer\n  winter - set the default colormap to winter\n\n_Event handling\n\n  connect - register an event handler\n  disconnect - remove a connected event handler\n\n_Matrix commands\n\n  cumprod   - the cumulative product along a dimension\n  cumsum    - the cumulative sum along a dimension\n  detrend   - remove the mean or besdt fit line from an array\n  diag      - the k-th diagonal of matrix\n  diff      - the n-th differnce of an array\n  eig       - the eigenvalues and eigen vectors of v\n  eye       - a matrix where the k-th diagonal is ones, else zero\n  find      - return the indices where a condition is nonzero\n  fliplr    - flip the rows of a matrix up/down\n  flipud    - flip the columns of a matrix left/right\n  linspace  - a linear spaced vector of N values from min to max inclusive\n  logspace  - a log spaced vector of N values from min to max inclusive\n  meshgrid  - repeat x and y to make regular matrices\n  ones      - an array of ones\n  rand      - an array from the uniform distribution [0,1]\n  randn     - an array from the normal distribution\n  rot90     - rotate matrix k*90 degress counterclockwise\n  squeeze   - squeeze an array removing any dimensions of length 1\n  tri       - a triangular matrix\n  tril      - a lower triangular matrix\n  triu      - an upper triangular matrix\n  vander    - the Vandermonde matrix of vector x\n  svd       - singular value decomposition\n  zeros     - a matrix of zeros\n\n_Probability\n\n  normpdf   - The Gaussian probability density function\n  rand      - random numbers from the uniform distribution\n  randn     - random numbers from the normal distribution\n\n_Statistics\n\n  amax      - the maximum along dimension m\n  amin      - the minimum along dimension m\n  corrcoef  - correlation coefficient\n  cov       - covariance matrix\n  mean      - the mean along dimension m\n  median    - the median along dimension m\n  norm      - the norm of vector x\n  prod      - the product along dimension m\n  ptp       - the max-min along dimension m\n  std       - the standard deviation along dimension m\n  asum      - the sum along dimension m\n  ksdensity - the kernel density estimate\n\n_Time series analysis\n\n  bartlett  - M-point Bartlett window\n  blackman  - M-point Blackman window\n  cohere    - the coherence using average periodiogram\n  csd       - the cross spectral density using average periodiogram\n  fft       - the fast Fourier transform of vector x\n  hamming   - M-point Hamming window\n  hanning   - M-point Hanning window\n  hist      - compute the histogram of x\n  kaiser    - M length Kaiser window\n  psd       - the power spectral density using average periodiogram\n  sinc      - the sinc function of array x\n\n_Dates\n\n  date2num  - convert python datetimes to numeric representation\n  drange    - create an array of numbers for date plots\n  num2date  - convert numeric type (float days since 0001) to datetime\n\n_Other\n\n  angle     - the angle of a complex array\n  griddata  - interpolate irregularly distributed data to a regular grid\n  load      - Deprecated--please use loadtxt.\n  loadtxt   - load ASCII data into array.\n  polyfit   - fit x, y to an n-th order polynomial\n  polyval   - evaluate an n-th order polynomial\n  roots     - the roots of the polynomial coefficients in p\n  save      - Deprecated--please use savetxt.\n  savetxt   - save an array to an ASCII file.\n  trapz     - trapezoidal integration\n\n__end\n\n.. [*] MATLAB is a registered trademark of The MathWorks, Inc.\n\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 218, 0))

# 'import six' statement (line 218)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_114546 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 218, 0), 'six')

if (type(import_114546) is not StypyTypeError):

    if (import_114546 != 'pyd_module'):
        __import__(import_114546)
        sys_modules_114547 = sys.modules[import_114546]
        import_module(stypy.reporting.localization.Localization(__file__, 218, 0), 'six', sys_modules_114547.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 218, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'six', import_114546)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 220, 0))

# Multiple import statement. import sys (1/2) (line 220)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 220, 0), 'sys', sys, module_type_store)
# Multiple import statement. import warnings (2/2) (line 220)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 220, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 222, 0))

# 'from matplotlib.cbook import flatten, exception_to_str, silent_list, iterable, dedent' statement (line 222)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_114548 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 222, 0), 'matplotlib.cbook')

if (type(import_114548) is not StypyTypeError):

    if (import_114548 != 'pyd_module'):
        __import__(import_114548)
        sys_modules_114549 = sys.modules[import_114548]
        import_from_module(stypy.reporting.localization.Localization(__file__, 222, 0), 'matplotlib.cbook', sys_modules_114549.module_type_store, module_type_store, ['flatten', 'exception_to_str', 'silent_list', 'iterable', 'dedent'])
        nest_module(stypy.reporting.localization.Localization(__file__, 222, 0), __file__, sys_modules_114549, sys_modules_114549.module_type_store, module_type_store)
    else:
        from matplotlib.cbook import flatten, exception_to_str, silent_list, iterable, dedent

        import_from_module(stypy.reporting.localization.Localization(__file__, 222, 0), 'matplotlib.cbook', None, module_type_store, ['flatten', 'exception_to_str', 'silent_list', 'iterable', 'dedent'], [flatten, exception_to_str, silent_list, iterable, dedent])

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 0), 'matplotlib.cbook', import_114548)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 225, 0))

# 'import matplotlib' statement (line 225)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_114550 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 225, 0), 'matplotlib')

if (type(import_114550) is not StypyTypeError):

    if (import_114550 != 'pyd_module'):
        __import__(import_114550)
        sys_modules_114551 = sys.modules[import_114550]
        import_module(stypy.reporting.localization.Localization(__file__, 225, 0), 'mpl', sys_modules_114551.module_type_store, module_type_store)
    else:
        import matplotlib as mpl

        import_module(stypy.reporting.localization.Localization(__file__, 225, 0), 'mpl', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'matplotlib', import_114550)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')


# Call to catch_warnings(...): (line 228)
# Processing the call keyword arguments (line 228)
kwargs_114554 = {}
# Getting the type of 'warnings' (line 228)
warnings_114552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 5), 'warnings', False)
# Obtaining the member 'catch_warnings' of a type (line 228)
catch_warnings_114553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 5), warnings_114552, 'catch_warnings')
# Calling catch_warnings(args, kwargs) (line 228)
catch_warnings_call_result_114555 = invoke(stypy.reporting.localization.Localization(__file__, 228, 5), catch_warnings_114553, *[], **kwargs_114554)

with_114556 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 228, 5), catch_warnings_call_result_114555, 'with parameter', '__enter__', '__exit__')

if with_114556:
    # Calling the __enter__ method to initiate a with section
    # Obtaining the member '__enter__' of a type (line 228)
    enter___114557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 5), catch_warnings_call_result_114555, '__enter__')
    with_enter_114558 = invoke(stypy.reporting.localization.Localization(__file__, 228, 5), enter___114557)
    
    # Call to simplefilter(...): (line 229)
    # Processing the call arguments (line 229)
    unicode_114561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 26), 'unicode', u'ignore')
    # Processing the call keyword arguments (line 229)
    kwargs_114562 = {}
    # Getting the type of 'warnings' (line 229)
    warnings_114559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'warnings', False)
    # Obtaining the member 'simplefilter' of a type (line 229)
    simplefilter_114560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 4), warnings_114559, 'simplefilter')
    # Calling simplefilter(args, kwargs) (line 229)
    simplefilter_call_result_114563 = invoke(stypy.reporting.localization.Localization(__file__, 229, 4), simplefilter_114560, *[unicode_114561], **kwargs_114562)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 230, 4))
    
    # 'import matplotlib.finance' statement (line 230)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
    import_114564 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 230, 4), 'matplotlib.finance')

    if (type(import_114564) is not StypyTypeError):

        if (import_114564 != 'pyd_module'):
            __import__(import_114564)
            sys_modules_114565 = sys.modules[import_114564]
            import_module(stypy.reporting.localization.Localization(__file__, 230, 4), 'matplotlib.finance', sys_modules_114565.module_type_store, module_type_store)
        else:
            import matplotlib.finance

            import_module(stypy.reporting.localization.Localization(__file__, 230, 4), 'matplotlib.finance', matplotlib.finance, module_type_store)

    else:
        # Assigning a type to the variable 'matplotlib.finance' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'matplotlib.finance', import_114564)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')
    
    # Calling the __exit__ method to finish a with section
    # Obtaining the member '__exit__' of a type (line 228)
    exit___114566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 5), catch_warnings_call_result_114555, '__exit__')
    with_exit_114567 = invoke(stypy.reporting.localization.Localization(__file__, 228, 5), exit___114566, None, None, None)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 232, 0))

# 'from matplotlib.dates import date2num, num2date, datestr2num, strpdate2num, drange, epoch2num, num2epoch, mx2num, DateFormatter, IndexDateFormatter, DateLocator, RRuleLocator, YearLocator, MonthLocator, WeekdayLocator, DayLocator, HourLocator, MinuteLocator, SecondLocator, rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY, MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY, SECONDLY, relativedelta' statement (line 232)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_114568 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 232, 0), 'matplotlib.dates')

if (type(import_114568) is not StypyTypeError):

    if (import_114568 != 'pyd_module'):
        __import__(import_114568)
        sys_modules_114569 = sys.modules[import_114568]
        import_from_module(stypy.reporting.localization.Localization(__file__, 232, 0), 'matplotlib.dates', sys_modules_114569.module_type_store, module_type_store, ['date2num', 'num2date', 'datestr2num', 'strpdate2num', 'drange', 'epoch2num', 'num2epoch', 'mx2num', 'DateFormatter', 'IndexDateFormatter', 'DateLocator', 'RRuleLocator', 'YearLocator', 'MonthLocator', 'WeekdayLocator', 'DayLocator', 'HourLocator', 'MinuteLocator', 'SecondLocator', 'rrule', 'MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU', 'YEARLY', 'MONTHLY', 'WEEKLY', 'DAILY', 'HOURLY', 'MINUTELY', 'SECONDLY', 'relativedelta'])
        nest_module(stypy.reporting.localization.Localization(__file__, 232, 0), __file__, sys_modules_114569, sys_modules_114569.module_type_store, module_type_store)
    else:
        from matplotlib.dates import date2num, num2date, datestr2num, strpdate2num, drange, epoch2num, num2epoch, mx2num, DateFormatter, IndexDateFormatter, DateLocator, RRuleLocator, YearLocator, MonthLocator, WeekdayLocator, DayLocator, HourLocator, MinuteLocator, SecondLocator, rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY, MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY, SECONDLY, relativedelta

        import_from_module(stypy.reporting.localization.Localization(__file__, 232, 0), 'matplotlib.dates', None, module_type_store, ['date2num', 'num2date', 'datestr2num', 'strpdate2num', 'drange', 'epoch2num', 'num2epoch', 'mx2num', 'DateFormatter', 'IndexDateFormatter', 'DateLocator', 'RRuleLocator', 'YearLocator', 'MonthLocator', 'WeekdayLocator', 'DayLocator', 'HourLocator', 'MinuteLocator', 'SecondLocator', 'rrule', 'MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU', 'YEARLY', 'MONTHLY', 'WEEKLY', 'DAILY', 'HOURLY', 'MINUTELY', 'SECONDLY', 'relativedelta'], [date2num, num2date, datestr2num, strpdate2num, drange, epoch2num, num2epoch, mx2num, DateFormatter, IndexDateFormatter, DateLocator, RRuleLocator, YearLocator, MonthLocator, WeekdayLocator, DayLocator, HourLocator, MinuteLocator, SecondLocator, rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY, MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY, SECONDLY, relativedelta])

else:
    # Assigning a type to the variable 'matplotlib.dates' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 0), 'matplotlib.dates', import_114568)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 245, 0))

# 'from matplotlib.mlab import amap, base_repr, binary_repr, bivariate_normal, center_matrix, csv2rec, demean, detrend, detrend_linear, detrend_mean, detrend_none, dist, dist_point_to_segment, distances_along_curve, entropy, exp_safe, fftsurr, find, frange, get_sparse_matrix, get_xyz_where, griddata, identity, inside_poly, is_closed_polygon, ispower2, isvector, l1norm, l2norm, log2, longest_contiguous_ones, longest_ones, movavg, norm_flat, normpdf, path_length, poly_below, poly_between, prctile, prctile_rank, rec2csv, rec_append_fields, rec_drop_fields, rec_join, rk4, rms_flat, segments_intersect, slopes, stineman_interp, vector_lengths, window_hanning, window_none' statement (line 245)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_114570 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 245, 0), 'matplotlib.mlab')

if (type(import_114570) is not StypyTypeError):

    if (import_114570 != 'pyd_module'):
        __import__(import_114570)
        sys_modules_114571 = sys.modules[import_114570]
        import_from_module(stypy.reporting.localization.Localization(__file__, 245, 0), 'matplotlib.mlab', sys_modules_114571.module_type_store, module_type_store, ['amap', 'base_repr', 'binary_repr', 'bivariate_normal', 'center_matrix', 'csv2rec', 'demean', 'detrend', 'detrend_linear', 'detrend_mean', 'detrend_none', 'dist', 'dist_point_to_segment', 'distances_along_curve', 'entropy', 'exp_safe', 'fftsurr', 'find', 'frange', 'get_sparse_matrix', 'get_xyz_where', 'griddata', 'identity', 'inside_poly', 'is_closed_polygon', 'ispower2', 'isvector', 'l1norm', 'l2norm', 'log2', 'longest_contiguous_ones', 'longest_ones', 'movavg', 'norm_flat', 'normpdf', 'path_length', 'poly_below', 'poly_between', 'prctile', 'prctile_rank', 'rec2csv', 'rec_append_fields', 'rec_drop_fields', 'rec_join', 'rk4', 'rms_flat', 'segments_intersect', 'slopes', 'stineman_interp', 'vector_lengths', 'window_hanning', 'window_none'])
        nest_module(stypy.reporting.localization.Localization(__file__, 245, 0), __file__, sys_modules_114571, sys_modules_114571.module_type_store, module_type_store)
    else:
        from matplotlib.mlab import amap, base_repr, binary_repr, bivariate_normal, center_matrix, csv2rec, demean, detrend, detrend_linear, detrend_mean, detrend_none, dist, dist_point_to_segment, distances_along_curve, entropy, exp_safe, fftsurr, find, frange, get_sparse_matrix, get_xyz_where, griddata, identity, inside_poly, is_closed_polygon, ispower2, isvector, l1norm, l2norm, log2, longest_contiguous_ones, longest_ones, movavg, norm_flat, normpdf, path_length, poly_below, poly_between, prctile, prctile_rank, rec2csv, rec_append_fields, rec_drop_fields, rec_join, rk4, rms_flat, segments_intersect, slopes, stineman_interp, vector_lengths, window_hanning, window_none

        import_from_module(stypy.reporting.localization.Localization(__file__, 245, 0), 'matplotlib.mlab', None, module_type_store, ['amap', 'base_repr', 'binary_repr', 'bivariate_normal', 'center_matrix', 'csv2rec', 'demean', 'detrend', 'detrend_linear', 'detrend_mean', 'detrend_none', 'dist', 'dist_point_to_segment', 'distances_along_curve', 'entropy', 'exp_safe', 'fftsurr', 'find', 'frange', 'get_sparse_matrix', 'get_xyz_where', 'griddata', 'identity', 'inside_poly', 'is_closed_polygon', 'ispower2', 'isvector', 'l1norm', 'l2norm', 'log2', 'longest_contiguous_ones', 'longest_ones', 'movavg', 'norm_flat', 'normpdf', 'path_length', 'poly_below', 'poly_between', 'prctile', 'prctile_rank', 'rec2csv', 'rec_append_fields', 'rec_drop_fields', 'rec_join', 'rk4', 'rms_flat', 'segments_intersect', 'slopes', 'stineman_interp', 'vector_lengths', 'window_hanning', 'window_none'], [amap, base_repr, binary_repr, bivariate_normal, center_matrix, csv2rec, demean, detrend, detrend_linear, detrend_mean, detrend_none, dist, dist_point_to_segment, distances_along_curve, entropy, exp_safe, fftsurr, find, frange, get_sparse_matrix, get_xyz_where, griddata, identity, inside_poly, is_closed_polygon, ispower2, isvector, l1norm, l2norm, log2, longest_contiguous_ones, longest_ones, movavg, norm_flat, normpdf, path_length, poly_below, poly_between, prctile, prctile_rank, rec2csv, rec_append_fields, rec_drop_fields, rec_join, rk4, rms_flat, segments_intersect, slopes, stineman_interp, vector_lengths, window_hanning, window_none])

else:
    # Assigning a type to the variable 'matplotlib.mlab' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 0), 'matplotlib.mlab', import_114570)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 257, 0))

# 'from matplotlib import cbook, mlab, plt' statement (line 257)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_114572 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 257, 0), 'matplotlib')

if (type(import_114572) is not StypyTypeError):

    if (import_114572 != 'pyd_module'):
        __import__(import_114572)
        sys_modules_114573 = sys.modules[import_114572]
        import_from_module(stypy.reporting.localization.Localization(__file__, 257, 0), 'matplotlib', sys_modules_114573.module_type_store, module_type_store, ['cbook', 'mlab', 'pyplot'])
        nest_module(stypy.reporting.localization.Localization(__file__, 257, 0), __file__, sys_modules_114573, sys_modules_114573.module_type_store, module_type_store)
    else:
        from matplotlib import cbook, mlab, pyplot as plt

        import_from_module(stypy.reporting.localization.Localization(__file__, 257, 0), 'matplotlib', None, module_type_store, ['cbook', 'mlab', 'pyplot'], [cbook, mlab, plt])

else:
    # Assigning a type to the variable 'matplotlib' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 0), 'matplotlib', import_114572)

# Adding an alias
module_type_store.add_alias('plt', 'pyplot')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 258, 0))

# 'from matplotlib.pyplot import ' statement (line 258)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_114574 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 258, 0), 'matplotlib.pyplot')

if (type(import_114574) is not StypyTypeError):

    if (import_114574 != 'pyd_module'):
        __import__(import_114574)
        sys_modules_114575 = sys.modules[import_114574]
        import_from_module(stypy.reporting.localization.Localization(__file__, 258, 0), 'matplotlib.pyplot', sys_modules_114575.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 258, 0), __file__, sys_modules_114575, sys_modules_114575.module_type_store, module_type_store)
    else:
        from matplotlib.pyplot import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 258, 0), 'matplotlib.pyplot', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'matplotlib.pyplot' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 0), 'matplotlib.pyplot', import_114574)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 260, 0))

# 'from numpy import ' statement (line 260)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_114576 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 260, 0), 'numpy')

if (type(import_114576) is not StypyTypeError):

    if (import_114576 != 'pyd_module'):
        __import__(import_114576)
        sys_modules_114577 = sys.modules[import_114576]
        import_from_module(stypy.reporting.localization.Localization(__file__, 260, 0), 'numpy', sys_modules_114577.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 260, 0), __file__, sys_modules_114577, sys_modules_114577.module_type_store, module_type_store)
    else:
        from numpy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 260, 0), 'numpy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 0), 'numpy', import_114576)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 261, 0))

# 'from numpy.fft import ' statement (line 261)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_114578 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 261, 0), 'numpy.fft')

if (type(import_114578) is not StypyTypeError):

    if (import_114578 != 'pyd_module'):
        __import__(import_114578)
        sys_modules_114579 = sys.modules[import_114578]
        import_from_module(stypy.reporting.localization.Localization(__file__, 261, 0), 'numpy.fft', sys_modules_114579.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 261, 0), __file__, sys_modules_114579, sys_modules_114579.module_type_store, module_type_store)
    else:
        from numpy.fft import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 261, 0), 'numpy.fft', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.fft' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 0), 'numpy.fft', import_114578)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 262, 0))

# 'from numpy.random import ' statement (line 262)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_114580 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 262, 0), 'numpy.random')

if (type(import_114580) is not StypyTypeError):

    if (import_114580 != 'pyd_module'):
        __import__(import_114580)
        sys_modules_114581 = sys.modules[import_114580]
        import_from_module(stypy.reporting.localization.Localization(__file__, 262, 0), 'numpy.random', sys_modules_114581.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 262, 0), __file__, sys_modules_114581, sys_modules_114581.module_type_store, module_type_store)
    else:
        from numpy.random import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 262, 0), 'numpy.random', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.random' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 0), 'numpy.random', import_114580)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 263, 0))

# 'from numpy.linalg import ' statement (line 263)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_114582 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 263, 0), 'numpy.linalg')

if (type(import_114582) is not StypyTypeError):

    if (import_114582 != 'pyd_module'):
        __import__(import_114582)
        sys_modules_114583 = sys.modules[import_114582]
        import_from_module(stypy.reporting.localization.Localization(__file__, 263, 0), 'numpy.linalg', sys_modules_114583.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 263, 0), __file__, sys_modules_114583, sys_modules_114583.module_type_store, module_type_store)
    else:
        from numpy.linalg import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 263, 0), 'numpy.linalg', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.linalg' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'numpy.linalg', import_114582)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 265, 0))

# 'import numpy' statement (line 265)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_114584 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 265, 0), 'numpy')

if (type(import_114584) is not StypyTypeError):

    if (import_114584 != 'pyd_module'):
        __import__(import_114584)
        sys_modules_114585 = sys.modules[import_114584]
        import_module(stypy.reporting.localization.Localization(__file__, 265, 0), 'np', sys_modules_114585.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 265, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 0), 'numpy', import_114584)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 266, 0))

# 'import numpy.ma' statement (line 266)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_114586 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 266, 0), 'numpy.ma')

if (type(import_114586) is not StypyTypeError):

    if (import_114586 != 'pyd_module'):
        __import__(import_114586)
        sys_modules_114587 = sys.modules[import_114586]
        import_module(stypy.reporting.localization.Localization(__file__, 266, 0), 'ma', sys_modules_114587.module_type_store, module_type_store)
    else:
        import numpy.ma as ma

        import_module(stypy.reporting.localization.Localization(__file__, 266, 0), 'ma', numpy.ma, module_type_store)

else:
    # Assigning a type to the variable 'numpy.ma' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 0), 'numpy.ma', import_114586)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 269, 0))

# 'import datetime' statement (line 269)
import datetime

import_module(stypy.reporting.localization.Localization(__file__, 269, 0), 'datetime', datetime, module_type_store)


# Assigning a Attribute to a Name (line 273):
# Getting the type of 'six' (line 273)
six_114588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'six')
# Obtaining the member 'moves' of a type (line 273)
moves_114589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), six_114588, 'moves')
# Obtaining the member 'builtins' of a type (line 273)
builtins_114590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), moves_114589, 'builtins')
# Obtaining the member 'bytes' of a type (line 273)
bytes_114591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), builtins_114590, 'bytes')
# Assigning a type to the variable 'bytes' (line 273)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 0), 'bytes', bytes_114591)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
