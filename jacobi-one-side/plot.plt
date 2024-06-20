reset
set autoscale xfixmin
set autoscale xfixmax
set autoscale yfixmin
set autoscale yfixmax
set pm3d map
set palette rgb 33,13,10

# Set terminal to PNG
set term png

set title "60x60 matrix size, 2000 iterations"

# Set output file
set output "solution.png"

# Plot and save
splot "solution.dat" bin array=62x62 format="%lf" rotate=90deg with image notitle

# Reset output
set output

