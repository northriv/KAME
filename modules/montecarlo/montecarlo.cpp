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
//#define _drand() ((double)random() / (0x7fffffffuL))
//#define _drand() (drand48())

//#include <boost/random/lagged_fibonacci.hpp>
//typedef boost::lagged_fibonacci23209 trandom_generator;
#include <boost/random/mersenne_twister.hpp>
typedef boost::mt19937 trandom_generator;
#include <boost/random/uniform_01.hpp>
extern boost::uniform_01<trandom_generator> g_float_random_dist;
extern trandom_generator g_float_random_generator;
#define _drand() (g_float_random_dist())

#include "montecarlo.h"

#include <pthread.h>

trandom_generator g_float_random_generator;
//trandom_generator g_float_random_generator(static_cast<unsigned int>(std::time(0)));
boost::uniform_01<trandom_generator> g_float_random_dist(g_float_random_generator);

MonteCarlo::Vector3<double> MonteCarlo::s_ASiteIsingVector[16];
volatile bool MonteCarlo::s_bAborting = false;

using namespace std;

int MonteCarlo::s_L;
int MonteCarlo::s_num_spins;
double MonteCarlo::s_alpha;
double MonteCarlo::s_dfactor;
int MonteCarlo::s_cutoff_real;
double MonteCarlo::s_cutoff_real_radius;
MonteCarlo::FieldRealArray MonteCarlo::s_fields_real[16][16];
MonteCarlo::FieldRealArray MonteCarlo::s_fields_real_B[16][16][3];
MonteCarlo::FieldRealArray MonteCarlo::s_fields_real_8a[8][16][3];
MonteCarlo::FieldRealArray MonteCarlo::s_fields_real_48f[48][16][3];
int MonteCarlo::s_cutoff_rec;
double MonteCarlo::s_cutoff_rec_radius;
std::vector<MonteCarlo::Spin> MonteCarlo::s_fields_rec[16][16];
std::vector<MonteCarlo::Vector3<MonteCarlo::Spin> > MonteCarlo::s_fields_rec_generic[16];
double MonteCarlo::s_fields_rec_sum;
std::vector<std::complex<MonteCarlo::Spin> > MonteCarlo::s_exp_ph[16];
std::vector<int> MonteCarlo::s_4r2_neighbor;

MonteCarlo::MonteCarlo(int num_threads)
	: 
	m_bTerminated(false),
	m_sec_cache_enabled(false),
	m_third_cache_enabled(false),
	m_sec_cache_firsttime(true),
	m_third_cache_firsttime(true)
{
    int lsize = s_num_spins/16;

    for(int site1 = 0; site1 < 16; site1++) {
	#ifdef PACK_4FLOAT
        m_spins_real[site1].resize(spins_real_index(0,0,s_L)/4);
	#else
        m_spins_real[site1].resize(3*lsize);
	#endif
        
        m_field_pri_cached[site1].resize(lsize, 0.0);
        m_field_pri_cached_sane.resize(lsize, 0);
        for(int site2 = 0; site2 < 16; site2++) {
            m_field_sec_cached[site2][site1].resize(lsize, 0.0);
            m_field_sec_cached_sane[site2].resize(lsize, 0);
            m_field_third_cached[site2][site1].resize(lsize, 0.0);
            m_field_third_cached_sane[site2].resize(lsize, 0);
        }
        m_probability_buffers[0][site1].resize(lsize, 0.0);
        m_probability_buffers[1][site1].resize(lsize, 0.0);
    }
        
    fprintf(stderr, "# of spins = %d\n", 16*lsize);
    randomize();

	for(int i = 0; i < num_threads - 1; i++) {
		pthread_t tid;
		int ret =
			pthread_create(&tid, NULL, xthread_start_routine, this);
		ASSERT(!ret);
		m_threads.push_back(tid);
    }
}
MonteCarlo::~MonteCarlo()
{
	{
		XScopedLock<XCondition> lock(m_thread_pool_cond);
        m_bTerminated = true;
        m_thread_pool_cond.broadcast();
	}
    for(deque<pthread_t>::iterator it = m_threads.begin(); it != m_threads.end(); it++)
	{
		void *retv;
		int ret = pthread_join(*it, &retv);
		ASSERT(!ret);
	}
}
void
MonteCarlo::read(istream &is)
{
    if(!is.good()) throw "input io error\n";
    string str;
    do {
        getline(is, str);
    } while(str[0] != '#');
    while(str[0] == '#') {
        getline(is, str);
    }
    int size;
    sscanf(str.c_str(), "size=%d", &size); 
    if(size != s_L) throw "size mismatch\n";
    is >> str;
    if(str != "[") throw "ill format\n";
    for(int site1 = 0; site1 < 16; site1++) {
        is >> str;
        if(str != "[") throw "ill format\n";
        for(int k1 = 0; k1 < s_L; k1++) {
            is >> str;
            if(str != "[") throw "ill format\n";
            for(int j1 = 0; j1 < s_L; j1++) {
                is >> str;
                if(str != "[") throw "ill format\n";
                for(int i1 = 0; i1 < s_L; i1++) {
                    is >> str;
                    if(str != "[") throw "ill format\n";
                    is >> str;
                    int x = atoi(str.c_str());
                    if(abs(x) != 1) throw "value be +-1\n";
                    int sidx = spins_real_index(i1,j1,k1);
                    writeSpin(x, site1, sidx);
                    is >> str;
                    if(str != "?") {
                        int lidx = lattice_index(i1,j1,k1);
                        m_field_pri_cached[site1][lidx] = atof(str.c_str());
                        m_field_pri_cached_sane[lidx] |= 1u << site1;
                    }
                    is >> str;
                    if(str != "]") throw "ill format\n";
                }
                is >> str;
                if(str != "]") throw "ill format\n";
            }
            is >> str;
            if(str != "]") throw "ill format\n";
        }
        is >> str;
        if(str != "]") throw "ill format\n";
    }
    is >> str;
    if(str != "]") throw "ill format\n";

    makeReciprocalImage();
}

void
MonteCarlo::write(ostream &os)
{
    if(!os.good()) throw "output io error\n";
    os << "# MonteCarlo calculation for pyrochlore. Kentaro Kitagawa." << endl;
    os << "# ver. 1.0." << endl;
    os << "# Spin configuration below." << endl;
    os << "size=" << s_L << endl;
    os << "[ ";
    for(int site1 = 0; site1 < 16; site1++) {
        os << "[ ";
        for(int k1 = 0; k1 < s_L; k1++) {
			os << "[ ";
			for(int j1 = 0; j1 < s_L; j1++) {
				os << "[ ";
                for(int i1 = 0; i1 < s_L; i1++) {
                    os << "[ ";
                    int lidx = lattice_index(i1,j1,k1);
                    os << readSpin(site1, spins_real_index(lidx));
                    if(m_field_pri_cached_sane[lidx] & (1u << site1)) {
                        char buf[31];
                        snprintf(buf, 30, "%.16g", m_field_pri_cached[site1][lidx]);
                        os << " " << buf;
					}
                    else
					{
						os << " ?";
					}
                    os << " ] " ;
                }
                os << "]" << endl;
            }
            os << "]" << endl;
        }
        os << "]" << endl;
    }
    os << "]" << endl;
}

void
MonteCarlo::randomize()
{
    fprintf(stderr, "Randomize spins\n");
    for(int site1 = 0; site1 < 16; site1++) {
        m_ext_field[site1] = 0.0;
    }
    for(int site1 = 0; site1 < 16; site1++) {
        for(int k1 = 0; k1 < s_L; k1++) {
			for(int j1 = 0; j1 < s_L; j1++) {
                for(int i1 = 0; i1 < s_L; i1++) {
                    int sidx = spins_real_index(i1,j1,k1);
                    int val = (_drand() < 0.5) ? 1 : -1;
                    writeSpin(val, site1, sidx);
                }
			}
        }
    }
    makeReciprocalImage();
}
MonteCarlo::Quartet
MonteCarlo::siteMagnetization()
{
    Quartet quartet;
    for(int k1 = 0; k1 < s_L; k1++) {
        for(int j1 = 0; j1 < s_L; j1++) {
            for(int i1 = 0; i1 < s_L; i1++) {
                int sidx = spins_real_index(i1,j1,k1);
                for(int trans = 0; trans < 4; trans++) {
                    int in = 0;
                    for(int site_wo_trans = 0; site_wo_trans < 4; site_wo_trans++) {
                        int site1 = site_wo_trans + 4*trans;
                        Spin spin = readSpin(site1, sidx);
                        quartet.sites[site1 % 4] += spin;
                        in += (spin == 1) ? 1 : 0;
                    }
                    quartet.twotwo += (in == 2) ? 1 : 0;
                    quartet.onethree += ((in == 1) || (in == 3)) ? 1 : 0;
                }
            }
        }
    }
    quartet.twotwo /= (s_num_spins/4);
    quartet.onethree /= (s_num_spins/4);
    for(int site1 = 0; site1 < 4; site1++) {
        quartet.sites[site1] /= (s_num_spins/4);
        quartet.sites[site1] *= A_MOMENT;
    }
    return quartet;
}
MonteCarlo::Vector3<double>
MonteCarlo::magnetization()
{
    Vector3<double> m;
    Quartet quartet = siteMagnetization();
    for(int site1 = 0; site1 < 4; site1++) {
        Vector3<double> v(s_ASiteIsingVector[site1]);
        v *= quartet.sites[site1];
        m += v;
    }
    m *= 1.0/4; // per one spin.
    return m;
}
void
MonteCarlo::takeThermalAverage(long double tests_after_check)
{
    m_SumDeltaU += m_DeltaU * tests_after_check;

    for(int site = 0; site < 16; site++) {
        m_SumSpin[site] += real(m_spins_rec[site][reciprocal_index(0,0,0)]) * tests_after_check;
    }

    m_SumTests += tests_after_check;
}

double
MonteCarlo::exec(double temp, Vector3<double> field, int *flips,
				 long double *tests, double *DUav, Vector3<double> *Mav)
{
    m_beta = 1.0/K_B/temp;

    for(int site1 = 0; site1 < 16; site1++) {
        m_ext_field[site1] = field.innerProduct(s_ASiteIsingVector[site1]);
    }
    
    m_DeltaU = 0.0;
    m_SumDeltaU = 0.0;
    for(int site = 0; site < 16; site++) {
        m_SumSpin[site] = 0.0;
    }
    m_SumTests = 0.0;

    doTests(flips, *tests);

    *DUav = m_SumDeltaU / s_num_spins / m_SumTests;
    Vector3<double> m;
    for(int site1 = 0; site1 < 16; site1++) {
        Vector3<double> v(s_ASiteIsingVector[site1]);
        v *= (double)(m_SumSpin[site1]);
        m += v;
    }
    m *= A_MOMENT / s_num_spins / m_SumTests;
    *Mav = m;
    *tests = m_SumTests;
    return m_DeltaU/s_num_spins;
}
inline double
MonteCarlo::flippingProbability(int site, int lidx, double h, double *pdu)
{
    int sidx = spins_real_index(lidx);
    double du = 2 * readSpin(site, sidx) * A_MOMENT * MU_B * (
        h + m_ext_field[site]);
    *pdu = du;

    if(du <= 0.0) return 1.0;

    // probability.
    return exp(-m_beta*du);
}
long double
MonteCarlo::accelFlipping()
{
    //calculate probabilities.
    m_sec_cache_enabled = true;

    double sum_p = 0.0;
    
    int current_buffer;
    if(m_last_flipped_lattice_index >= 0) {
        // swap buffers.
        current_buffer = 1 - m_last_probability_buffer;
    }
    else {
        current_buffer = 0;
        m_play_back_buffer = false;
    }
    if(m_play_back_buffer) {
        for(int site1 = 0; site1 < 16; site1++) {
            double *pprob = &m_probability_buffers[current_buffer][site1][0];
            for(int i = 0; i < s_num_spins/16; i++) {
				sum_p += *(pprob++);
            }
        }
    }
    else {
        // iterate target site.
        for(int site1 = 0; site1 < 16; site1++) {
            double *pprob = &m_probability_buffers[current_buffer][site1][0];
            int cnt = s_num_spins/16;
            for(int lidx = 0; lidx < cnt; lidx++) {
                double h = hinteraction(site1, lidx);
                double du = 0.0;
                double probability = flippingProbability(site1, lidx, h, &du);
                *pprob = probability;
                pprob++;
                sum_p += probability;
            }
        }
    }
    m_play_back_buffer = false;
    m_last_probability_buffer = current_buffer;
    
    double p_av = sum_p / s_num_spins;
    if(p_av <= 0.0) return -1;
    // counts to next flip.
    long double cnt_d = ceill(log2l(1.0-_drand()) / log2l((long double)1.0 - p_av));
    if(cnt_d <= 0.0) return -1;
//    if(cnt_d > 1e4 * m_spins.size()) return -1;
//    long long int cnt = llrintl(cnt_d);
//    if(cnt == 0) return 0;
    // choose target site
    double p = _drand()*sum_p;
    int lidx = s_num_spins/16 - 1;
    int site1 = 16;
    double pit = 0.0;
    for(int site = 0; site < 16; site++) {
        double *pprob = &m_probability_buffers[current_buffer][site][0];
        for(int i = 0; i < s_num_spins/16; i++) {
            pit += *(pprob++);
            if(pit >= p) {
                lidx = i;
                site1 = site;
                break;
            }
        }
        if(pit >= p) break;
    }
    ASSERT(pit < sum_p * 1.00001);
    ASSERT(site1 != 16);
    double du = 0.0;
    flippingProbability(site1, lidx, hinteraction(site1, lidx), &du);
    flipSpin(site1, lidx, du, cnt_d);
//    fprintf(stderr, "Accelerate Flipping done. Skipped tests = %lld\n", cnt);

    if((m_last_flipped_lattice_index == lidx) && (m_last_flipped_site == site1)) {
        m_play_back_buffer = true;
        fprintf(stderr, "0");
    }
    else {
        if(m_last_flipped_lattice_index >= 0) {
            int size = s_L;
            int n = lidx;
            int i1 = n % size;
            n /= size;
            int j1 = n % size;
            n /= size;
            int k1 = n;
            n = m_last_flipped_lattice_index;
            int i2 = n % size;
            n /= size;
            int j2 = n % size;
            n /= size;
            int k2 = n;
            VectorInt v = distance(site1, m_last_flipped_site,
								   (i1 - i2 + size) % size, (j1 - j2 + size) % size, (k1 - k2 + size) % size); 
            int d = v.x*v.x + v.y*v.y + v.z*v.z;
            ASSERT(d > 0);
            for(int i = 0; i < 10; i++) {
                if(i == 9) {
                    ASSERT(d >= s_4r2_neighbor[i]);
                    fprintf(stderr, ".");
                    break;
                }
                if(s_4r2_neighbor[i] == d) {
                    fprintf(stderr, "%d", i + 1);
                    break;
                }
            }
            if(sum_p > 3.0) {
				// leave accel flipping mode.
                m_last_probability_buffer = -1;
            }
        }
    }
    m_last_flipped_lattice_index = lidx;
    m_last_flipped_site = site1;
    return cnt_d;
}
double
MonteCarlo::internalEnergy() {
    bool abondon_cache = (_drand() < 0.05);
    if(abondon_cache) {
        fprintf(stderr, "Abondon cache.\n");
        fill(m_field_pri_cached_sane.begin(), m_field_pri_cached_sane.end(), 0);
    }
    //internal energy. [J/A-site]
    double U = 0.0;
    // iterate target site.
    for(int site1 = 0; site1 < 16; site1++) {
		for(int lidx = 0; lidx < s_num_spins/16; lidx++) {
			double h = 0.0;
			h = hinteraction(site1, lidx);
			// interacting field must be half.
			h *= 0.5;
			h += m_ext_field[site1];
			U += -readSpin(site1, spins_real_index(lidx)) * A_MOMENT * MU_B * h;
		}}
    
    U /= s_num_spins;
    return U;
}
void
MonteCarlo::doTests(int *flips, long double tests)
{
    int flipped = 0;
    int flipped_checked = 0;
    m_sec_cache_hit = 0;
    m_third_cache_hit = 0;
    m_hinteractions_called = 0;
    long tested = 0;
    long tests_after_check = 0;
    long double tests_accel_flip = 0;
    long double tests_accel_flip_started = 0;
    int flipped_accel_flip_started = 0;
    bool accel_flip = false;
    for(;;) {
        if((flipped >= *flips) && (m_SumTests + tests_after_check >= tests)) break;
        if(s_bAborting) {
            fprintf(stderr, "Signal caught! Aborting...\n");
            break;
        }

        if(accel_flip) {
            takeThermalAverage(tests_after_check);
            tests_after_check = 0;
            long double adv = accelFlipping();
            if(adv <= 0) {
                fprintf(stderr, "Spins are completely freezed!.\n");
                break;
            }
            tests_accel_flip += adv;
            flipped++;
            if(m_last_probability_buffer < 0) {
                fprintf(stderr, "\nSkipped tests = %Lg. Flipped = %d\n", 
						(long double)(tests_accel_flip - tests_accel_flip_started), flipped - flipped_accel_flip_started);
                accel_flip = false;
//                activateThreading();
            }
            continue;
        }
        
        // pick-up target spin.
        int idx = (int)floor(_drand() * s_num_spins);

        int site = idx % 16;
        int lidx = idx / 16;
        
        double h = hinteraction(site, lidx);

        double du = 0.0;
        double probability = flippingProbability(site, lidx, h, &du);
        
        tested++;
        tests_after_check++;
        
        if((probability >= 1) || (_drand() < probability)) {
            flipSpin(site, lidx, du, tests_after_check);
            tests_after_check = 0;
            flipped++;
        }
        
        if((tested % s_num_spins == 0) && (tested != 0)) {
            int flips = flipped - flipped_checked;
            
            if(flips <= 0) {
                fprintf(stderr, "Flipping Acceleration...");
                accel_flip = true;
                tests_accel_flip_started = tests_accel_flip;
                flipped_accel_flip_started = flipped;
                m_last_flipped_lattice_index = -1;
            }

            if(m_third_cache_enabled) {
                if(m_hinteractions_called > m_sec_cache_hit) {
                    double hit_prob = (double)m_third_cache_hit / (m_hinteractions_called - m_sec_cache_hit) / 16;
                    if(!m_third_cache_firsttime && (hit_prob < THIRD_CACHE_OFF_FACTOR)) {
                        m_third_cache_enabled = false;
                        for(int i = 0; i < 16; i++) {
                            fill(m_field_third_cached_sane[i].begin(), m_field_third_cached_sane[i].end(), 0);
                        }
                        fprintf(stderr, "Flip = %f %%\n", (double)100.0*flips / m_hinteractions_called);
                        fprintf(stderr, "Disable 3rd cache. hit = %f%%\n", 100.0*hit_prob);
                    }
                    m_third_cache_firsttime = false;
                }
            }
            else {
                double hit_prob_estimate = 
                    pow(1.0 - (double)flips / m_hinteractions_called /16 / (s_L*s_L*s_L)
						* (4.0*M_PI/3.0*s_cutoff_real/2*s_cutoff_real/2*s_cutoff_real/2), (double)s_num_spins);
                if(hit_prob_estimate > THIRD_CACHE_ON_FACTOR) {
                    m_third_cache_enabled = true;
                    m_third_cache_firsttime = true;
                    fprintf(stderr, "Flip = %f %%\n", (double)100.0*flips / m_hinteractions_called);
                    fprintf(stderr, "Enable 3rd cache. estimate = %f%%\n", 100.0*hit_prob_estimate);
                }
            }
            m_third_cache_hit = 0;

            if(m_sec_cache_enabled) {
                double hit_prob = (double)m_sec_cache_hit / (m_hinteractions_called) / 16;
                if(!m_sec_cache_firsttime && (hit_prob < SEC_CACHE_OFF_FACTOR)) {
                    m_sec_cache_enabled = false;
                    for(int i = 0; i < 16; i++) {
                        fill(m_field_sec_cached_sane[i].begin(), m_field_sec_cached_sane[i].end(), 0);
                    }
                    fprintf(stderr, "Flip = %f %%\n", (double)100.0*flips / m_hinteractions_called);
                    fprintf(stderr, "Disable secondary cache. hit = %f%%\n", 100.0*hit_prob);
                }
//                    fprintf(stderr, "Secondary cache hit = %f%%\n", 100.0*hit_prob);
                m_sec_cache_firsttime = false;
            }
            else {
                double hit_prob_estimate = 
                    pow(1.0 - (double)flips / m_hinteractions_called /16, (double)s_num_spins);
                if(hit_prob_estimate > SEC_CACHE_ON_FACTOR) {
                    m_sec_cache_enabled = true;
                    m_sec_cache_firsttime = true;
                    fprintf(stderr, "Flip = %f %%\n", (double)100.0*flips / m_hinteractions_called);
                    fprintf(stderr, "Enable secondary cache. estimate = %f%%\n", 100.0*hit_prob_estimate);
                }
            }
            m_sec_cache_hit = 0;
                        
            flipped_checked = flipped;
            m_hinteractions_called = 0;
        } 
    }
    if(accel_flip) {
        fprintf(stderr, "\nSkipped tests = %Lg. Flipped = %d\n", 
				(long double)(tests_accel_flip - tests_accel_flip_started), flipped - flipped_accel_flip_started);
    }
    *flips = flipped;
    takeThermalAverage(tests_after_check);
}

inline void
MonteCarlo::modifyReciprocalImage(Spin diff, int site1, int i, int j, int k)
{
    int cutoff = s_cutoff_rec;
    int cnt = 2*cutoff + 1;
    complex<Spin> *pspin = &m_spins_rec[site1][0];
    
    Vector3<double> pos1(cg_ASitePositions[site1]);
    pos1 *= LATTICE_CONST / 4.0;
    double phx = -2*M_PI / (LATTICE_CONST * s_L) * (i * LATTICE_CONST + pos1.x);
    double phy = -2*M_PI / (LATTICE_CONST * s_L) * (j * LATTICE_CONST + pos1.y);
    double phz = -2*M_PI / (LATTICE_CONST * s_L) * (k * LATTICE_CONST + pos1.z);
    complex<Spin> exp_i_rx = exp(complex<Spin>(0.0, phx));
    complex<Spin> exp_i_ry = exp(complex<Spin>(0.0, phy));
    complex<Spin> exp_i_rz = exp(complex<Spin>(0.0, phz));
        
    complex<Spin> exp_ikrz = ((Spin)diff)
		* exp(complex<Spin>(0.0, -cutoff * (phx + phy)));
    for(int kz = 0; kz <= cutoff; kz++) {
        complex<Spin> exp_ikryz = exp_ikrz;
        for(int ky = -cutoff; ky <= cutoff; ky++) {
            complex<Spin> exp_ikr = exp_ikryz;
            for(int n = 0; n < cnt; n++) {
                pspin[n] += exp_ikr;
                exp_ikr *= exp_i_rx;
            }
            pspin+=cnt;
            exp_ikryz *= exp_i_ry;
        }
        exp_ikrz *= exp_i_rz;
    }
    ASSERT(pspin == &*m_spins_rec[site1].end());
}
void
MonteCarlo::makeReciprocalImage()
{
    int cutoff = s_cutoff_rec;
    if(!cutoff) return;
    for(int site = 0; site < 16; site++) {
        m_spins_rec[site].clear();
        m_spins_rec[site].resize((2*cutoff+1)*(2*cutoff+1)*(cutoff+1), 0.0);
        for(int k = 0; k < s_L; k++) {
            for(int j = 0; j < s_L; j++) {
                for(int i = 0; i < s_L; i++) {
                    modifyReciprocalImage(readSpin(site, spins_real_index(i,j,k)), site, i, j, k);
                }
            }
        }
    }
}
void
MonteCarlo::flipSpin(int site1, int lidx, double du, long double tests_after_check)
{
    takeThermalAverage(tests_after_check);
    
    m_DeltaU += du;

    int sidx = spins_real_index(lidx);
    Spin oldv = readSpin(site1, sidx);
    int n = lidx;
    int i = n % s_L;
    n /= s_L;
    int j = n % s_L;
    n /= s_L;
    int k = n;
    modifyReciprocalImage(-2*oldv, site1, i, j, k);
    // flip spin. keep repeated image.
    writeSpin(-oldv, site1, sidx);
    ASSERT(spins_real_index(i,j,k) == sidx);
    
    //set dirty flags to caches.
    fill(m_field_pri_cached_sane.begin(), m_field_pri_cached_sane.end(), 0);
    if(m_sec_cache_enabled) {
        fill(m_field_sec_cached_sane[site1].begin(), m_field_sec_cached_sane[site1].end(), 0);
    }
    if(m_third_cache_enabled) {
        int size = s_L;
        int dist = s_cutoff_real;
        int r2bound = dist*dist;
        unsigned short *p0 = &m_field_third_cached_sane[site1][0];
        for(int dk = -dist; dk <= dist; dk++) {
            int dk2 = abs(dk) - 1;
            dk2 = dk2*dk2;
            unsigned short *p_k = p0 + lattice_index(0, 0, (k + dk + size) % size);
            for(int dj = -dist; dj <= dist; dj++) {
                int dj2 = abs(dj) - 1;
                dj2 = dk2 + dj2*dj2;
                unsigned short *p_j = p_k + lattice_index(0, (j + dj + size) % size, 0);
                for(int di = -dist; di <= dist; di++) {
                    int di2 = abs(di) - 1;
                    int r2 = dj2 + di2*di2;
                    if(r2 <= r2bound) {
                        p_j[lattice_index((i + di + size) % size, 0, 0)] = 0;
                    }
                }
            }
        }
    }
    
}
void
MonteCarlo::write(char *data, double *fields, double *probabilities)
{
    for(int site = 0; site < 16; site++) {
        for(int k1 = 0; k1 < s_L; k1++) {
            for(int j1 = 0; j1 < s_L; j1++) {
                for(int i1 = 0; i1 < s_L; i1++) {
                    *(data++) = lrint(readSpin(site, spins_real_index(i1,j1,k1)));
                    if(fields) {
                        int lidx = lattice_index(i1,j1,k1);
                        double h = hinteraction(site, lidx);
                        *(fields++) = h;
                        if(probabilities) {
                            double du;
                            double probability = flippingProbability(site, lidx, h, &du);
                            *(probabilities++) = probability;
                        }
                    }
                }
            }
        }
    }
}
void
MonteCarlo::write_bsite(Vector3<double> *fields)
{
    for(int site = 0; site < 16; site++) {
        for(int k1 = 0; k1 < s_L; k1++) {
            for(int j1 = 0; j1 < s_L; j1++) {
                for(int i1 = 0; i1 < s_L; i1++) {
                    Vector3<double> h;
                    h += iterate_real_generic(s_fields_real_B[site], i1, j1, k1);
                    Vector3<double> pos(cg_BSitePositions[site]);
                    pos *= 1.0/4.0;
                    h += iterate_rec_generic(pos, i1, j1, k1);
                    *(fields++) = h;
                }
            }
        }
    }
}
void
MonteCarlo::write_8asite(Vector3<double> *fields)
{
    for(int site = 0; site < 8; site++) {
        for(int k1 = 0; k1 < s_L; k1++) {
            for(int j1 = 0; j1 < s_L; j1++) {
                for(int i1 = 0; i1 < s_L; i1++) {
                    Vector3<double> h;
                    h += iterate_real_generic(s_fields_real_8a[site], i1, j1, k1);
                    Vector3<double> pos(cg_8aSitePositions[site]);
                    pos *= 1.0/8.0;
                    h += iterate_rec_generic(pos, i1, j1, k1);
                    *(fields++) = h;
                }
            }
        }
    }
}
void
MonteCarlo::write_48fsite(Vector3<double> *fields)
{
    for(int site = 0; site < 48; site++) {
        for(int k1 = 0; k1 < s_L; k1++) {
            for(int j1 = 0; j1 < s_L; j1++) {
                for(int i1 = 0; i1 < s_L; i1++) {
                    Vector3<double> h;
                    h += iterate_real_generic(s_fields_real_48f[site], i1, j1, k1);
                    Vector3<double> pos(cg_48fSitePositions[site]);
                    pos *= 1.0/8.0;
                    h += iterate_rec_generic(pos, i1, j1, k1);
                    *(fields++) = h;
                }
            }
        }
    }
}

void
MonteCarlo::read(const char *data, double temp, Vector3<double> field)
{
    m_beta = 1.0/K_B/temp;

    for(int site1 = 0; site1 < 16; site1++) {
        m_ext_field[site1] = field.innerProduct(s_ASiteIsingVector[site1]);
    }
    
    for(int site = 0; site < 16; site++) {
        for(int k1 = 0; k1 < s_L; k1++) {
            for(int j1 = 0; j1 < s_L; j1++) {
                for(int i1 = 0; i1 < s_L; i1++) {
                    writeSpin(*(data++), site, spins_real_index(i1,j1,k1));
                }
            }
        }
        if(m_sec_cache_enabled)
            fill(m_field_sec_cached_sane[site].begin(), m_field_sec_cached_sane[site].end(), 0);
        if(m_third_cache_enabled)
            fill(m_field_third_cached_sane[site].begin(), m_field_third_cached_sane[site].end(), 0);
    }
    fill(m_field_pri_cached_sane.begin(), m_field_pri_cached_sane.end(), 0);
    makeReciprocalImage();
}
