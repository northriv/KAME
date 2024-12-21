/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
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

#ifndef xpythonmoduleH
#define xpythonmoduleH

#ifdef USE_PYBIND11

#define PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF //For mainthread call.

#include <pybind11/embed.h> //include before kame headers
#include <pybind11/functional.h> //needed to wrap std::function.
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include <pybind11/eigen.h>
#include "xnode.h"

#endif //USE_PYBIND11
//---------------------------------------------------------------------------
#endif //
