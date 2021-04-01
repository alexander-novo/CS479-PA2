if (!exists("outfile")) outfile='plot.png'
# if (!exists("outfile")) outfile='plot.pdf'

# Read quadratic parameters
paramNames = system('head -1 '.paramFile)
paramCount = words(paramNames)
paramLine = system('head -2 '.paramFile.' | tail -1')

do for [i = 1:paramCount] {
	eval(word(paramNames, i).' = '.word(paramLine, i))
}

# General Quadratic, a function of two variables - we're only interested in the contour curve at f(x,y) = 0, though.
f(x,y) = a*x*x + b*x*y + c*y*y + d*x + e*y + f

set contour
set view map
unset surface
set cntrparam levels discrete 0
set isosamples 4000,4000

set xrange [xmin:xmax]
set yrange [ymin:ymax]
set table $ContourTable
splot f(x, y)

unset table
unset contour
unset xrange
unset yrange
unset cntrparam
set surface

# set terminal pdf enhanced size 80in, 48in font "arial,80"
set terminal pngcairo enhanced size 12000, 7200 truecolor font "arial,90"
set encoding utf8
set output outfile

set autoscale fix

set border lw 3

set title plotTitle

BASE = -zmax
set isosamples 100
set pm3d at s explicit hidden3d
# set hidden3d lc rgb "black"
unset hidden3d
set palette model RGB define (0 "black", 1 "#7570b3", 2 "#1b9e77")
unset colorbox
set view 70
set xyplane at BASE
set xtics scale .5
set ytics scale .5
set ztics 0,(zmax/4)
set xlabel "x_1"
set ylabel "x_2"

splot pdfFile u 1:2:3:4 w pm3d lc rgb "black" lw 1 notitle,\
	  sample1 u 1:2:(BASE) w points pt 6 ps .75 lw 4.5 lc rgb "#a97570b3" notitle,\
      sample2 u 1:2:(BASE) w points pt 6 ps .75 lw 4.5 lc rgb "#a91b9e77" notitle,\
	  $ContourTable u 1:2:(BASE) w lines lw 6 lc rgb 'black' title "Decision Boundary",\
	  NaN w points pt 7 ps 16 lc rgb "#7570b3" title "ω_1",\
	  NaN w points pt 7 ps 16 lc rgb "#1b9e77" title "ω_2"