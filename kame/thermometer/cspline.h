#ifndef CSPLINE_H_
#define CSPLINE_H_

#include "support.h"

#include <map>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

//! Cubic (natural) spline approximation.
class CSplineApprox
{
public:
    CSplineApprox(const std::map<double, double> &pts);
    ~CSplineApprox();
    //! Do spline approx.
    double approx(double x) const;
private:
    gsl_interp_accel *m_accel;
    gsl_spline *m_spline;
};

#endif /*CSPLINE_H_*/
