if (!exists("outfile")) outfile='plot.pdf'

# Read quadratic parameters
paramNames = system('head -1 '.paramFile)
paramCount = words(paramNames)
paramLine = system('head -2 '.paramFile.' | tail -1')

do for [i = 1:paramCount] {
	eval(word(paramNames, i).' = '.word(paramLine, i))
}

set terminal pdf enhanced size 8in, 4.8in
set encoding utf8
set output outfile

set autoscale fix
set yrange [0:* < .3]
set xlabel "β"

set title plotTitle
set arrow from 0.,boundB to 0.5,boundB      nohead lc rgb "black" dashtype 2
set arrow from 0.,boundC to betaStar,boundC nohead lc rgb "red"   dashtype 4
set arrow from 0.5,0 to 0.5,boundB           nohead lc rgb "black" dashtype 2
set arrow from betaStar,0 to betaStar,boundC nohead lc rgb "red"   dashtype 4

set label "Bhattacharyya Bound" at .25,boundB tc rgb "black" center offset 0, character 1
set label "Chernoff Bound" at (betaStar / 2),boundC tc rgb "red" center offset 0, character -1
set label "β^*" at betaStar,0 center point offset 0, character -1

plot boundFile u 1:2 w lines title "f(β)",\
     boundFile u 1:3 w lines title "f'(β)"