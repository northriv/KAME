/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef MONTECARLO_H_
#define MONTECARLO_H_

#include "config.h"
#include "thread.h"

#include "support.h"
#include "atomic.h"

#include <math.h>
#include <vector>
#include <deque>

#include <gsl/gsl_sf_erf.h>
//! erfc(x)=1-erf(x)=(2/sqrt(PI))int_x^inf exp(-t^2)dt
#define _erfc(x) gsl_sf_erfc(x)

#include <iostream>
#include <complex>

//! Bohr magnetron [J/T]
#define MU_B 9.27410e-24
//! Boltzman const. [J/K]
#define K_B 1.38062e-23
//! Abogadro's number [1/mol]
#define N_A 6.02217e23

//! Moment of A site. [mu_B]
#define A_MOMENT 10.0
//! lattice constant [m]
#define LATTICE_CONST 10.12e-10
#define LATTICE_VOLUME (LATTICE_CONST*LATTICE_CONST*LATTICE_CONST)

#define SEC_CACHE_ON_FACTOR 0.02
#define SEC_CACHE_OFF_FACTOR 0.01
#define THIRD_CACHE_ON_FACTOR 0.05
#define THIRD_CACHE_OFF_FACTOR 0.02

//! For SIMD vectorization.
//#define PACK_4FLOAT

//! Nearest neighbor exchange interaction [K]
#define J_NN -1.24


class MonteCarlo
{
public:
    template <typename T>
    struct Vector3 {
        Vector3() : x(0), y(0), z(0) {}
        Vector3(T nx, T ny) : x(nx), y(ny), z(0) {} 
        Vector3(T nx, T ny, T nz) : x(nx), y(ny), z(nz) {} 
        template <typename X>
        Vector3(const Vector3<X> &r) : x(r.x), y(r.y), z(r.z) {} 
        template <typename X>
        Vector3(const X n[3]) : x(n[0]), y(n[1]), z(n[2]) {} 
        T x; T y; T z;
        
        bool operator==(const Vector3 &s1) const
		{return ((x == s1.x) && (y == s1.y) && (z == s1.z));}
        template<typename X>
        Vector3 &operator+=(const Vector3<X> &s1) {
            x += s1.x; y += s1.y; z += s1.z;
            return *this;
        }
        template<typename X>
        Vector3 &operator-=(const Vector3<X> &s1) {
            x -= s1.x; y -= s1.y; z -= s1.z;
            return *this;
        }
        template<typename X>
        Vector3 &operator*=(const X &k) {
            x *= k; y *= k; z *= k;
            return *this;
        }
        //! square of distance between this and a point
        T distance2(const Vector3<T> &s1) const {
			T x1 = x - s1.x;
			T y1 = y - s1.y;
			T z1 = z - s1.z;
            return x1*x1 + y1*y1 + z1*z1;
        }
        //! square of distance between this and a line from s1 to s2
        T distance2(const Vector3<T> &s1, const Vector3<T> &s2) const  {
			T x1 = x - s1.x;
			T y1 = y - s1.y;
			T z1 = z - s1.z;
			T x2 = s2.x - s1.x;
			T y2 = s2.y - s1.y;
			T z2 = s2.z - s1.z;
			T zbab = x1*x2 + y1*y2 + z1*z2;
			T ab2 = x2*x2 + y2*y2 + z2*z2;
			T zb2 = x1*x1 + y1*y1 + z1*z1;
			return (zb2*ab2 - zbab*zbab) / ab2;
        }
        void normalize() {
            T ir = (T)1.0 / sqrt(x*x + y*y + z*z);
            x *= ir; y *= ir; z *= ir;
        }
        Vector3 &vectorProduct(const Vector3 &s1) {
			Vector3 s2;
            s2.x = y * s1.z - z * s1.y;
            s2.y = z * s1.x - x * s1.z;
            s2.z = x * s1.y - y * s1.x;
            *this = s2;
            return *this;
        }
        T innerProduct(const Vector3 &s1) const {
            return x * s1.x + y * s1.y + z * s1.z;
        }
        T abs() const {
            return sqrt(x*x+y*y+z*z);
        }
    }; 

    MonteCarlo(int num_threads);
    ~MonteCarlo();
    
    //! \return Delta U [J/A site].
    //! \arg flips # of flipping to be performed. # done is returned.
    //! \arg tests Min # of tests to be performed. # done is returned.
    double exec(double temp, Vector3<double> field,
				int *flips, long double *tests, double *DUav, Vector3<double> *Mav);
    //! randomize spins
    void randomize();
    //! [mu_B/A site]
    Vector3<double> magnetization();
    //! [mu_B/site]
    struct Quartet {
        Quartet() {
            sites[0] = 0.0;
            sites[1] = 0.0;
            sites[2] = 0.0;
            sites[3] = 0.0;
            twotwo = 0.0;
            onethree = 0.0;
        }
        double sites[4];
        //! icerule probability.
        double twotwo;
        //! 1-in,3-out or 1-out,3-in probability.
        double onethree;
    };
    Quartet siteMagnetization();
    //! internal energy U [J/A site]
    double internalEnergy();
    //! prepare interactions.
    //! \arg size # of lattices for one direction.
    //! \arg dfactor Bulk demagnetization factor D (0 < D < 1).
    //! \arg cutoff_real [L.U.].
    //! \arg cutoff_rec [2pi/L].
    //! \arg alpha Ewald convergence factor [1/L].
    //! \return # of interactions.
    static int setupField(int size, double dfactor,
						  double cutoff_real, double cutoff_rec, double alpha);
    //! read snapshot.
    void read(std::istream &);
    void read(const char *spins, double temp, Vector3<double> field);
    //! write snapshot.
    void write(std::ostream &);
    void write(char *spins, 
			   double *fields = 0, double *probabilities = 0);
    void write_bsite(Vector3<double> *fields);
    void write_8asite(Vector3<double> *fields);
    void write_48fsite(Vector3<double> *fields);
    
    static int length() {
        return s_L;
    }

    //! true if user interrupts.
    static volatile bool s_bAborting;

private:
    //! thread pool.
    volatile bool m_bTerminated;

    XCondition m_thread_pool_cond;
    XCondition m_hint_site2_last_cond;
    void execute();
    //! target spin for concurrent calc. of interactions.
    atomic<int> m_hint_site2_left, m_hint_site2_not_done;
    int m_hint_spin1_site;
    int m_hint_spin1_lattice_index;
    //! store calculated fields.
    double m_hint_fields[16];
    int m_hint_sec_cache_miss[16];

    //! Accelerate flipping.
    //! \return tests to flip.
    long double accelFlipping();
    int m_last_probability_buffer;
    int m_last_flipped_site;
    int m_last_flipped_lattice_index;
    bool m_play_back_buffer;
    std::vector<double> m_probability_buffers[2][16];
    
    //! Cache systems
    //! Primary cache: field at the target site.
    //! Secondary cache: field from a certain site at the target site.
    //! Third cache : field from a certain site with real space calc. at the target site.
    long m_hinteractions_called;
    bool m_sec_cache_enabled, m_third_cache_enabled;
    bool m_sec_cache_firsttime, m_third_cache_firsttime;
    //! total hittings to caches.
    long m_sec_cache_hit;
    atomic<long> m_third_cache_hit;
    //! cached interaction energy.
    std::vector<double> m_field_pri_cached[16];
    //! true if cached data is valid.
    std::vector<unsigned short> m_field_pri_cached_sane;
    //! cached interaction energy.
    std::vector<double> m_field_sec_cached[16][16];
    //! true if cached data is valid.
    std::vector<unsigned short> m_field_sec_cached_sane[16];
    //! cached interaction energy.
    std::vector<double> m_field_third_cached[16][16];
    //! true if cached data is valid.
    std::vector<unsigned short> m_field_third_cached_sane[16];

    //! many tests until fixed number of flipping is performed.
    //! \sa exec()
    void doTests(int *flips, long double tests);
    void flipSpin(int site, int lidx, double du, long double tests_after_check);
    inline double flippingProbability(int site, int lidx, double field, double *du);
    
    typedef Vector3<int> VectorInt;
    //! unit is 1/4 lattice const.
    static inline VectorInt distance(int site1, int site2, int di, int dj, int dk);
    //! max interaction distance for real space.
    static int s_cutoff_real;
    static double s_cutoff_real_radius;
    //! For reciprocal space. i.e. Ewald term.
    static int s_cutoff_rec;
    static double s_cutoff_rec_radius;
    //! Ewald convergence factor [1/m].
    static double s_alpha;
    //! Demagnetization D factor.
    static double s_dfactor;
    //! \return true if included.
    static int dipoleFieldReal(const Vector3<double> &dist_times_4, int site2, Vector3<double> *ret);
    static int dipoleFieldRec(const Vector3<double> &k, int site2, Vector3<double> *ret);

    typedef float Spin;

    static inline int lattice_index(int i, int j, int k);
    //! For real space calculations. Particle part.
    //! field along ising axis when spin is +1.
    //! index is given by distance_index()
    //! [T]
#ifdef PACK_4FLOAT
    struct PackedSpin {
        PackedSpin() {
            for(int i = 0; i < 4; i++) x[i] = 0.0;
        }
        PackedSpin(const PackedSpin &v) {
            for(int i = 0; i < 4; i++) x[i] = v.x[i];
        }
        Spin x[4];
        Spin sum() const {
            float h = 0.0;
            for(int i = 0; i < 4; i++) h += x[i];
            return h;
        }
        PackedSpin &operator+=(const PackedSpin &v) {
            for(int i = 0; i < 4; i++) x[i] += v.x[i];
            return *this;
        }
        PackedSpin &operator*=(const PackedSpin &v) {
            for(int i = 0; i < 4; i++) x[i] *= v.x[i];
            return *this;
        }
    } __attribute__((__aligned__(16)));
    struct FieldRealArray {std::vector<PackedSpin> align[4];};
#else
    typedef std::vector<Spin> FieldRealArray;
#endif
    static FieldRealArray s_fields_real[16][16];
    static FieldRealArray s_fields_real_B[16][16][3];
    static FieldRealArray s_fields_real_8a[8][16][3];
    static FieldRealArray s_fields_real_48f[48][16][3];
    static void addFieldsReal(MonteCarlo::Spin v, FieldRealArray& array, int di, int dj, int dk);
    //! For reciprocal space. i.e. Ewald term.
    static std::vector<Spin> s_fields_rec[16][16];
    static std::vector<Vector3<Spin> > s_fields_rec_generic[16];
    //! For self-energy caclulation.
    static double s_fields_rec_sum;
    static std::vector<std::complex<MonteCarlo::Spin> > s_exp_ph[16];
    static inline int reciprocal_index(int kx, int ky, int kz);
    //! spin orientations. in real space.
    //! images are repeated outside boundary along x.
#ifdef PACK_4FLOAT
    std::vector<PackedSpin> m_spins_real[16];
#else
    std::vector<Spin> m_spins_real[16];
#endif
    static inline int spins_real_index(int i, int j, int k);
    static inline int spins_real_index(int lidx);
    inline Spin readSpin(int site, int sidx);
    inline void writeSpin(Spin v, int site, int sidx);
    //! For reciprocal space. with q-cutoff.
    std::vector<std::complex<Spin> > m_spins_rec[16];
    void makeReciprocalImage();
    //! \arg diff e.g. -2, -1, 1, 2.
    inline void modifyReciprocalImage(Spin diff, int site, int i, int j, int k);

    //! internal field from surrounding spins along ising axis [T].
    inline double hinteraction(int site, int lidx);
    double hinteraction_miscache(int sec_cache_miss_cnt, int site, int lidx);
    inline double iterate_interactions(int site1, int lidx, int site2);
    Vector3<double> iterate_real_generic(const FieldRealArray[16][3], int i, int j, int k);
    inline double iterate_real_redirected(int cnt, const FieldRealArray &, int i, int j, int k, int site2);
    double iterate_real(int site1, int i, int j, int k, int site2);
    Vector3<double> iterate_rec_generic(Vector3<double> pos1, int i, int j, int k);
    Vector3<double> iterate_rec_generic(Vector3<double> pos1, int i, int j, int k, int site2);
    inline double iterate_rec_redirected(int cutoff, int site1, int i, int j, int k, int site2);
    double iterate_rec(int site1, int i, int j, int k, int site2);
         
    //! temperature. 1/k_B T [1/J].
    double m_beta;
    //! Along Ising direction [T]
    double m_ext_field[16];
    //! size of lattices
    static int s_L;
    //! # of spins
    static int s_num_spins;
    //! Delta U [J]
    long double m_DeltaU;
    void takeThermalAverage(long double tests_after_check);
    //! Sum Delta U [J]
    long double m_SumDeltaU;
    //! Sum Spin Polarization
    long double m_SumSpin[16];
    long double m_SumTests;
    //! 4*r^2 to nth neighbor.
    static std::vector<int> s_4r2_neighbor;
    
    std::deque<pthread_t> m_threads;
    static void * xthread_start_routine(void *);
    
    static Vector3<double> s_ASiteIsingVector[16];
};

//inline functions.
//! unit is 1/4 lattice const.
static const int cg_ASitePositions[16][3] = {
    {0,0,0}, {1,1,0}, {1,0,1}, {0,1,1},
    {2,2,0}, {3,3,0}, {3,2,1}, {2,3,1},
    {0,2,2}, {1,3,2}, {1,2,3}, {0,3,3},
    {2,0,2}, {3,1,2}, {3,0,3}, {2,1,3}
};
//! unit is 1/4 lattice const.
static const int cg_BSitePositions[16][3] = {
    {2,2,2}, {1,3,0}, {3,0,1}, {0,1,3},
    {0,0,2}, {3,1,0}, {1,2,1}, {2,3,3},
    {2,0,0}, {1,1,2}, {3,2,3}, {0,3,1},
    {0,2,0}, {3,3,2}, {1,0,3}, {2,1,1}
};
//! unit is 1/8 lattice const.
static const int cg_8aSitePositions[8][3] = {
    {1,1,1}, {5,5,1}, {1,5,5}, {5,1,5},
    {7,3,3}, {3,7,3}, {7,7,7}, {3,7,7}
};
#define OX (0.4201*8)
//! unit is 1/8 lattice const.
static const double cg_48fSitePositions[48][3] = {
    {OX,1,1}, {OX+4,5,1}, {OX,5,5}, {OX+4,1,5},
    {1,OX,1}, {5,OX+4,1}, {1,OX+4,5}, {5,OX,5},
    {1,1,OX}, {5,5,OX}, {1,5,OX+4}, {5,1,OX+4},
    {-OX+6,1,5}, {-OX+10,5,5}, {-OX+6,5,1}, {-OX+10,1,1},
    {5,-OX+6,1}, {1,-OX+10,1}, {5,-OX+10,5}, {1,-OX+6,5},
    {1,5,-OX+6}, {5,1,-OX+6}, {1,1,-OX+10}, {5,5,-OX+10},
    {7,OX+2,3}, {3,OX-2,3}, {7,OX-2,7}, {3,OX+2,7},
    {OX-2,3,3}, {OX+2,7,3}, {OX-2,7,7}, {OX+2,3,7},
    {3,3,OX-2}, {7,7,OX-2}, {3,7,OX+2}, {7,3,OX+2},
    {7,-OX+8,7}, {3,-OX+4,7}, {7,-OX+4,3}, {3,-OX+8,3},
    {-OX+4,7,3}, {-OX+8,3,3}, {-OX+4,3,7}, {-OX+8,7,7},
    {7,3,-OX+4}, {3,7,-OX+4}, {7,7,-OX+8}, {3,3,-OX+8}
};

//! Ising axes. to "in" direction.
static const int cg_ASiteIsingAxes[16][3] = {
    {1,1,1}, {-1,-1,1}, {-1,1,-1}, {1,-1,-1},
    {1,1,1}, {-1,-1,1}, {-1,1,-1}, {1,-1,-1},
    {1,1,1}, {-1,-1,1}, {-1,1,-1}, {1,-1,-1},
    {1,1,1}, {-1,-1,1}, {-1,1,-1}, {1,-1,-1}
};
inline int
MonteCarlo::reciprocal_index(int kx, int ky, int kz) {
    return (2*s_cutoff_rec + 1)*((2*s_cutoff_rec + 1)*kz + ky + s_cutoff_rec) + kx + s_cutoff_rec;
}
inline int
MonteCarlo::lattice_index(int i, int j, int k) {
    return s_L*(s_L*k + j) + i;
}
#ifdef PACK_4FLOAT
inline int
MonteCarlo::spins_real_index(int i, int j, int k) {
	int l = ((3 * s_L - 1) / 4 + 1) * 4;
	return 3*l*(s_L*k + j) + s_L + i;
}
inline int
MonteCarlo::spins_real_index(int lidx) {
	int l = ((3 * s_L - 1) / 4 + 1) * 4;
	return 3*l*(lidx / s_L) + s_L + (lidx % s_L);
}
inline MonteCarlo::Spin
MonteCarlo::readSpin(int site, int sidx) {
	int r = sidx / 4;
	int c = sidx % 4;
	return m_spins_real[site][r].x[c];
}
inline void
MonteCarlo::writeSpin(Spin v, int site, int sidx) {
	m_spins_real[site][sidx / 4].x[sidx % 4] = v;
	m_spins_real[site][(sidx - s_L) / 4].x[(sidx - s_L) % 4] = v;
	m_spins_real[site][(sidx + s_L) / 4].x[(sidx + s_L) % 4] = v;
}
#else
inline int
MonteCarlo::spins_real_index(int i, int j, int k) {
	return 3*s_L*(s_L*k + j) + i + s_L;
}
inline int
MonteCarlo::spins_real_index(int lidx) {
	return lidx * 3 -  (lidx % s_L) * 2 + s_L;
}
inline MonteCarlo::Spin
MonteCarlo::readSpin(int site, int sidx) {
	return m_spins_real[site][sidx];
}
inline void
MonteCarlo::writeSpin(Spin v, int site, int sidx) {
	m_spins_real[site][sidx - s_L] = v;
	m_spins_real[site][sidx] = v;
	m_spins_real[site][sidx + s_L] = v;
}
#endif
//! unit is 1/4 lattice const.
inline MonteCarlo::VectorInt
MonteCarlo::distance(int site1, int site2, int di, int dj, int dk)
{
    VectorInt v;
    v.x = 4*di + cg_ASitePositions[site2][0] - cg_ASitePositions[site1][0];
    v.y = 4*dj + cg_ASitePositions[site2][1] - cg_ASitePositions[site1][1];
    v.z = 4*dk + cg_ASitePositions[site2][2] - cg_ASitePositions[site1][2];
    return v;
}
inline double
MonteCarlo::iterate_interactions(int site1, int lidx, int site2)
{
    int n = lidx;
    int i = n % s_L;
    n /= s_L;
    int j = n % s_L;
    n /= s_L;
    int k = n;
    
    double h = iterate_rec(site1, i, j, k, site2);    
        
    if(m_third_cache_enabled &&
	   (m_field_third_cached_sane[site2][lidx] & (1u << site1))) {
        ++m_third_cache_hit;
        h += m_field_third_cached[site1][site2][lidx];
    }
    else {
        double hreal = iterate_real(site1, i, j, k, site2);
        if(m_third_cache_enabled) {
            m_field_third_cached[site1][site2][lidx] = hreal;
            m_field_third_cached_sane[site2][lidx] |= 1u << site1;
        }
        h += hreal;
    }
    
    if(m_sec_cache_enabled) {
        m_field_sec_cached[site1][site2][lidx] = h;
        m_field_sec_cached_sane[site2][lidx] |= 1u << site1;
    }
    
    return h;
}

inline double
MonteCarlo::hinteraction(int site1, int lidx)
{
    if(m_field_pri_cached_sane[lidx] & (1u << site1)) {
        return m_field_pri_cached[site1][lidx];
    }

    m_hinteractions_called++;

    double h = 0.0;
    int sec_cache_miss_cnt = 0;
    for(int site2 = 0; site2 < 16; site2++) {
        if(m_sec_cache_enabled &&
		   (m_field_sec_cached_sane[site2][lidx] & (1u << site1))) {
            m_sec_cache_hit++;
            h +=  m_field_sec_cached[site1][site2][lidx];
        }
        else {
            m_hint_sec_cache_miss[sec_cache_miss_cnt++] = site2;
        }
    }
    
    if(sec_cache_miss_cnt <= 8) {
        // Inefficient case for threading.
        for(int miss = 0; miss < sec_cache_miss_cnt; miss++) {  
            int site2 = m_hint_sec_cache_miss[miss];
            h += iterate_interactions(site1, lidx, site2);
        }
    }
    else
        h += hinteraction_miscache(sec_cache_miss_cnt, site1, lidx);

    // contribution from this spin must not be counted.
    // cache result.
    m_field_pri_cached[site1][lidx] = h;
    m_field_pri_cached_sane[lidx] |= 1u << site1;
    return h;
}

#endif /*MONTECARLO_H_*/
