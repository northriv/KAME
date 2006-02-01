/*
Finite Impulse Response Filter
*/


#ifndef FIR_H
#define FIR_H

#include <vector>

  class FIR
 {
 public:
 	FIR();
	~FIR();
	//! makes coeff. for BPF. Window func. method.
	//! \param taps odd num. a number of taps
	//! \param bandwidth 0 to 1.0. the unit is sampling freq.
	//! \param center 0.0 to 1.0. the unit is sampling freq.
	int setupBPF(int taps, double bandwidth, double center);
	//! \param src a pointer to src
	//! \param dst a pointer to dst
	//! \param len length
	int doFIR(double *src, double *dst, int len);
 private:
 	float oneFIR(float x, float *z, float *coeff, int taps);
	int m_Taps; ///< a number of taps divided by 2
	double m_BandWidth;
	double m_Center;
	std::vector<float> m_Coeff, m_Z; ///< coeff. buffer
};

#endif //FIR_H
