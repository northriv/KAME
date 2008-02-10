/***************************************************************************
		Copyright (C) 2002-2008 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
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
