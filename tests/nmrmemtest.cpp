#include "nmrmem.h"
#include <fstream>

NMRMEM mem1;

int main(int argc, char**argv) {
	std::vector<fftw_complex> memin, memout(atoi(argv[2]));
	std::ifstream fs(argv[1]);
	char line[128];
	double tmin = 1e10, tmax = -1e6;
	while(fs.getline(line, 128)) {
		fftw_complex z;
		double t;
		if(sscanf(line, "%lf %lf %lf", &t, &z.re, &z.im) == 3)
			memin.push_back(z);
		tmin = std::min(tmin, t);
		tmax = std::max(tmax, t);
	}
	double dt = (tmax - tmin) / (memin.size() - 1);
	mem1.exec(memin, memout, lrint((atof(argv[3]) - tmin) / dt), 0.005);
	
	double f = 0.0;
	double df = 1.0 / (tmax - tmin);
	for(unsigned int i = 0; i < memout.size(); i++) {
		f += df;
		printf("%g %g %g\n", f, memout[i].re, memout[i].im);
	}
	return 0;
}
