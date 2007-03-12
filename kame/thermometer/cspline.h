/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
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
