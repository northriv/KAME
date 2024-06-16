/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "rand.h"
#include "xthread.h"

#ifdef USE_STD_RANDOM
//C++11
    #include <random>
    struct Rand {
        Rand() : un01(0.0, 1.0) {}
        std::mt19937 rng_mt19937;
        std::uniform_real_distribution<> un01;
    };
    static XThreadLocal<Rand> stl_rand;
    double randMT19937() {
        return stl_rand->un01( stl_rand->rng_mt19937);
    }
#else
    static XMutex s_mutex;
    #include <boost/random/mersenne_twister.hpp>
    #include <boost/random/uniform_01.hpp>

    static boost::mt19937 s_rng_mt19937;
    static boost::uniform_01<boost::mt19937> s_rng_un01_mt19937(s_rng_mt19937);
    double randMT19937() {
        XScopedLock<XMutex> lock(s_mutex);
        return s_rng_un01_mt19937();
    }
#endif

