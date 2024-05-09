reset
set autoscale xfixmin
set autoscale xfixmax
set autoscale yfixmin
set autoscale yfixmax
set pm3d map
set palette rgb 33,13,10
splot "solution.dat" bin array=62x62 format="%lf" rotate=90deg with image
