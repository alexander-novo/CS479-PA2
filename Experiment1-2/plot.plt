if (!exists("outfile")) outfile='plot.png'

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


# set terminal pdf enhanced size 8in, 4.8in
set terminal png enhanced size 4000, 2400 truecolor font "arial,90"
set encoding utf8
set output outfile

set autoscale fix

set border lw 3
set style fill transparent solid 0.075 noborder
set style circle radius 0.03

set title plotTitle

set xlabel "x_1"
set ylabel "x_2"

if(exists("raw")) {
	unset border
	unset xtics
	unset ytics
	unset title
	unset key
}

plot sample1 u 1:2 w circles lc rgb '#7570b3' notitle,\
     sample2 u 1:2 w circles lc rgb '#1b9e77' notitle,\
	 $ContourTable w lines lw 6 lc rgb 'black' title "Decision Boundary",\
	 NaN w circles fill solid 1.0 noborder lc rgb '#7570b3' title "ω_1",\
	 NaN w circles fill solid 1.0 noborder lc rgb '#1b9e77' title "ω_2"