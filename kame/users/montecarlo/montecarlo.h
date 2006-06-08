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
        Vector3(const X n[3]) : x(n[0]), y(n[1]), z(n[2]) {} 
        T x; T y; T z;
        
        bool operator==(const Vector3<T> &s1)  const {return ((x == s1.x) && (y == s1.y) && (z == s1.z));}
        Vector3<T> &operator+=(const Vector3<T> &s1) {
            x += s1.x; y += s1.y; z += s1.z;
            return *this;
        }
        Vector3<T> &operator-=(const Vector3<T> &s1) {
            x -= s1.x; y -= s1.y; z -= s1.z;
            return *this;
        }
        Vector3<T> &operator*=(T k) {
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
            T ir = (T)1.0 / sqrtf(x*x + y*y + z*z);
            x *= ir; y *= ir; z *= ir;
        }
        Vector3<T> &vectorProduct(const Vector3<T> &s1) {
        Vector3<T> s2;
            s2.x = y * s1.z - z * s1.y;
            s2.y = z * s1.x - x * s1.z;
            s2.z = x * s1.y - y * s1.x;
            *this = s2;
            return *this;
        }
        T innerProduct(const Vector3<T> &s1) const {
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
    double exec(double temp, Vector3<double> field, int *flips, long double *tests);
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
    //! \arg distance max interaction distance.
    //! \return # of interactions.
    static int setupField(int size, double dfactor, double distance);
    //! read snapshot.
    void read(std::istream &);
    void read(const char *spins, double temp, Vector3<double> field);
    //! write snapshot.
    void write(std::ostream &);
    void write(char *spins, double *fields = 0, double *probabilities = 0);
    
    static int length() {
        return s_L;
    }

    //! true if user interrupts.
    static volatile bool s_bAborting;
private:
    //! thread pool.
    inline void activateThreading();
    inline void deactivateThreading();
    volatile bool m_bTerminated;
    XCondition m_thread_pool_cond;
    volatile bool m_thread_pool_active;
    void execute();
    //! target spin for concurrent calc. of interactions.
    atomic<int> m_hint_site2_left;
    int m_hint_spin1_site;
    int m_hint_spin1_lattice_index;
    //! store calculated fields.
    double m_hint_fields[16];
    int m_hint_done[16];

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
    long m_sec_cache_hit, m_third_cache_hit;
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
    void doTests(int *flips, long double *tests);
    void flipSpin(int site, int lidx, double du);
    inline double flippingProbability(int site, int lidx, double field, double *du);
    
    typedef struct {int x,y,z;} VectorInt;
    //! unit is 1/4 lattice const.
    static inline VectorInt distance(int site1, int site2, int di, int dj, int dk);
    //! max interaction distance for real space.
    static int s_cutoff_real;
    //! For reciprocal space. i.e. Ewald term.
    static int s_cutoff_rec;

    static inline int lattice_index(int i, int j, int k);
    //! For real space calculations. Particle part.
    //! field along ising axis when spin is +1.
    //! index is given by distance_index()
    //! [T]
    static std::vector<double> s_fields_real[16][16];
    static inline int distance_index(int di, int dj, int dk);
    //! For reciprocal space. i.e. Ewald term.
    static std::vector<double> s_fields_rec[16][16];
    //! For self-energy caclulation.
    static double s_fields_rec_sum;
    static std::vector<std::complex<double> > s_exp_ph[16];
    static inline int reciprocal_index(int kx, int ky, int kz);
    //! flag. true if interaction is to be counted.
    static std::vector<int> s_is_inside_sphere_real[16][16];
    static std::vector<int> s_is_inside_sphere_rec;
    //! spin orientations. in real space.
    //! images are repeated outside boundary along x.
    std::vector<int> m_spins_real[16];
    static inline int spins_real_index(int i, int j, int k);
    static inline int spins_real_index(int lidx);
    //! For reciprocal space. with q-cutoff.
    std::vector<std::complex<double> > m_spins_rec[16];
    void makeReciprocalImage();
    //! \arg diff e.g. -2, -1, 1, 2.
    inline void modifyReciprocalImage(int diff, int site, int i, int j, int k);

    //! internal field from surrounding spins along ising axis [T].
    inline double hinteraction(int site, int lidx);
    double hinteraction_miscache(int site, int lidx);
    double hinteraction_miscache_threading(int site, int lidx);
    inline double iterate_interactions(int site1, int lidx, int site2);
    inline double iterate_real_redirected(int cnt, int site1, int i, int j, int k, int site2);
    inline double iterate_real(int site1, int i, int j, int k, int site2);
    inline double iterate_rec_redirected(int cutoff, int site1, int i, int j, int k, int site2);
    inline double iterate_rec(int site1, int i, int j, int k, int site2);
         
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
//! Ising axes. to "in" direction.
static const int cg_ASiteIsingAxes[16][3] = {
    {1,1,1}, {-1,-1,1}, {-1,1,-1}, {1,-1,-1},
    {1,1,1}, {-1,-1,1}, {-1,1,-1}, {1,-1,-1},
    {1,1,1}, {-1,-1,1}, {-1,1,-1}, {1,-1,-1},
    {1,1,1}, {-1,-1,1}, {-1,1,-1}, {1,-1,-1}
};

inline int
MonteCarlo::distance_index(int di, int dj, int dk) {
    int size = 2*s_cutoff_real + 1;
    return size*(size*(dk + s_cutoff_real) + (dj + s_cutoff_real)) + (di + s_cutoff_real);
}
inline int
MonteCarlo::reciprocal_index(int kx, int ky, int kz) {
    return (2*s_cutoff_rec + 1)*((2*s_cutoff_rec + 1)*kz + ky + s_cutoff_rec) + kx + s_cutoff_rec;
}
inline int
MonteCarlo::lattice_index(int i, int j, int k) {
    return s_L*(s_L*k + j) + i;
}
inline int
MonteCarlo::spins_real_index(int i, int j, int k) {
    return 3*s_L*(s_L*k + j) + i + s_L;
}
inline int
MonteCarlo::spins_real_index(int lidx) {
    return lidx * 3 -  (lidx % s_L) * 2 + s_L;
}
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
MonteCarlo::hinteraction(int site, int lidx)
{
    if(m_field_pri_cached_sane[lidx] & (1u << site)) {
        return m_field_pri_cached[site][lidx];
    }

    m_hinteractions_called++;
    double h = hinteraction_miscache(site, lidx);

    // contribution from this spin must not be counted.
    // cache result.
    m_field_pri_cached[site][lidx] = h;
    m_field_pri_cached_sane[lidx] |= 1u << site;
    return h;
}
inline void
MonteCarlo::activateThreading()
{
    if(!m_thread_pool_active) {
        if(m_threads.empty()) return;
        m_thread_pool_active = true;
        m_thread_pool_cond.broadcast();
    }
}
inline void
MonteCarlo::deactivateThreading()
{
    m_thread_pool_active = false;
}

#endif /*MONTECARLO_H_*/
