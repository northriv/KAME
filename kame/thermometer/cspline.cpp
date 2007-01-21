#include "cspline.h"
#include <vector>

CSplineApprox::CSplineApprox(const std::map<double, double> &pts)
{
    m_accel = gsl_interp_accel_alloc();
    m_spline = gsl_spline_alloc(gsl_interp_cspline, pts.size());

    std::vector<double> x, y;
    for(std::map<double, double>::const_iterator it = pts.begin(); it != pts.end(); it++) {
        x.push_back(it->first);
        y.push_back(it->second);
    }
    gsl_spline_init(m_spline, &x[0], &y[0], x.size());
}

CSplineApprox::~CSplineApprox()
{
    gsl_spline_free(m_spline);
    gsl_interp_accel_free(m_accel);
}
double
CSplineApprox::approx(double x) const
{
    return gsl_spline_eval(m_spline, x, m_accel);
}
