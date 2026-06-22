# pact
Python Antigenic Cartograpy Tools (pact) is an extension of Racmacs. Unlike Racmacs, it can work both on "square" titer tables (which has sera in columns
and antigens in rows) and "flat" titer tables which has columns 'antigen_id', 'serum_id', 'titer'. It handles multiple 
repeats correctly without needing to merge them. There is also the option to add table_bias when there is table_id column 
in the flat table. This is to be used for instance when you want to combine different tables from different sources which 
you think might have overall titer magnitude differences. You can also use it for other purposes such as trying to combine 
two results using different cell types / assay types where you suspect only difference is in magnitude and not fold-drop. 
You can also allow antigen and serum avidity terms (called row and column avidity in the package). These parameters are 
regularized, see gradient_MDS for details. Default is that there is no row,column avidity or table bias. See tests folder
for details.

The stress function used is equivalent to Racmacs if row/col avidity and table bias terms are not used. If
these are included, they get added to the estimated titer in the expected way and are regularized whose parameters
are determined by the prior input to the optimizers (see gradient_MDS for details).

# Installation
You should be to install this with this pip on linux, macos or windows provided you have a C compiler available.
There are two optional dependencies: plotly and PyRacmacs. Both are used for some plotting functionalities, which
if not available will raise and error if you call these functions. See inside plot_lib.py. PyRacmacs is also
optionally used in some of the tests to compare results to Racmacs. 

# Tests
There are three tests. test_benchmark compares two different optimizations routines in pact against Racmacs (requires PyRacmacs),
test_global_minima.py evaluates global optima finding success of pact and finally test_long.py tests pact on a larger version
of the 2004 map which has extra data and repeats are not averaged (warning: this is unpublished data).

# Visualization
If you have PyRacmacs then pact can connect to it to produce maps ala Racmacs viewer. Otherwise you can also
use dedicated view function to get a plotly plot where you can color antigens and sera by any observable that
you supply to the map object (the options will appear as a roll down menu in the map). See test_long.py in tests
for an example on how to use it. 

# AI Statement
The viewing functionalities were created with Claude. Some of the docstrings were created
with Claude. Addition of table_bias terms was done with Claude. AI was used only 
under human supervision (me).