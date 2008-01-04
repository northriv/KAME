set term x11
! ./nmrmemtest ~/testwave.dat 16384 0 > nmrmemtest.dat
plot \
"nmrmemtest.dat" u 1 w l, "" u 2 w l

