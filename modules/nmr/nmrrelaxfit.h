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
//---------------------------------------------------------------------------

#ifndef nmrrelaxfitH
#define nmrrelaxfitH

#include "xlistnode.h"

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

class XRelaxFunc : public XNode
{
	XNODE_OBJECT
protected:
	XRelaxFunc(const char *name, bool runtime) : XNode(name, runtime) {}
public:
	virtual ~XRelaxFunc() {}
	//! called during fitting
	//! \param f f(t, it1) will be passed
	//! \param dfdt df/d(it1) will be passed
	//! \param t a time P1 or 2tau
	//! \param data a relaxation function
	//! \param it1 1/T1 or 1/T2   
	virtual void relax(double *f, double *dfdt, double t, double it1) = 0;   
  
	static int relax_f (const gsl_vector * x, void *params,
						gsl_vector * f);  
	static int relax_df (const gsl_vector * x, void *params,
						 gsl_matrix * J);  
	static int relax_fdf (const gsl_vector * x, void *params,
						  gsl_vector * f, gsl_matrix * J);   
}; 

class XRelaxFuncList : public XAliasListNode<XRelaxFunc>
{
	XNODE_OBJECT
protected:
	XRelaxFuncList(const char *name, bool runtime);
public:
	virtual ~XRelaxFuncList() {}
};


//---------------------------------------------------------------------------
#endif
