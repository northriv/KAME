#include "montecarlo.h"
using std::vector;
using std::complex;
#include <numeric>
using std::accumulate;

#include <pthread.h>
void *
MonteCarlo::xthread_start_routine(void *x)
{
    MonteCarlo *ptr = (MonteCarlo *)x;
    ptr->execute();
    return NULL;
}
int
MonteCarlo::setupField(int size, double dfactor, double radius)
{
    s_bAborting = false;
    s_L = size;
    s_num_spins = size*size*size*16;

    for(int site1 = 0; site1 < 16; site1++) {
        s_ASiteIsingVector[site1].x = cg_ASiteIsingAxes[site1][0] / sqrt(3.0);
        s_ASiteIsingVector[site1].y = cg_ASiteIsingAxes[site1][1] / sqrt(3.0);
        s_ASiteIsingVector[site1].z = cg_ASiteIsingAxes[site1][2] / sqrt(3.0);
    }
    
    int cutoff_real = (int)ceil(radius - 0.01);
    double alpha = 1.8/(radius * LATTICE_CONST);
    fprintf(stderr, "Ewalt convergence factor = (%g [LU])^{-1}\n", 1.0 / alpha / LATTICE_CONST);
    double mag_cutoff = _erfc(alpha*LATTICE_CONST*radius);
    fprintf(stderr, "Magnitude at the cutoff boundary = %g%%\n", 100.0*mag_cutoff);
    s_cutoff_real = cutoff_real;
    double radius_rec = sqrt(-log(mag_cutoff)) * 2.0 * alpha / (2.0*M_PI / (LATTICE_CONST * s_L));
    int cutoff_rec = (int)ceil(radius_rec);
    fprintf(stderr, "Cut-off of q-space calc. = %g [2pi/4L]\n", radius_rec);
    s_cutoff_rec = cutoff_rec;

    int cnt_n = 0;
    int cnt_nn = 0;
    double d_nn = 0.0;
    int cnt_nnn = 0;
    int cnt_3nn = 0;
    vector<int> cnt_n_r2(4*(cutoff_real+1)*4*(cutoff_real+1));
    for(int site1 = 0; site1 < 16; site1++) {
        Vector3<double> d1(s_ASiteIsingVector[site1]);
        for(int site2 = 0; site2 < 16; site2++) {
            s_fields_real[site1][site2].clear();
            s_fields_real[site1][site2].resize((cutoff_real*2+1)*(cutoff_real*2+1)*(cutoff_real*2+1));
            s_is_inside_sphere_real[site1][site2].clear();
            s_is_inside_sphere_real[site1][site2].resize((cutoff_real*2+1)*(cutoff_real*2+1)*(cutoff_real*2+1));
            for(int dk = -cutoff_real; dk <= cutoff_real; dk++) {
            for(int dj = -cutoff_real; dj <= cutoff_real; dj++) {
            for(int di = -cutoff_real; di <= cutoff_real; di++) {                
                VectorInt v = distance(site1,site2,di,dj,dk);
                int r2int = v.x*v.x + v.y*v.y + v.z*v.z;
                if(r2int == 0) {
                    continue;
                }
                // spherical boundary
                if(r2int - 0.01 > (4*radius)*(4*radius)) {
                    continue;
                }
                double r = LATTICE_CONST/4.0 * sqrt((double)r2int);
                double ir = 1.0/r;
                double alphar = alpha*r;
                double ir_eff = _erfc(alphar)*ir;
                double derfc = 2.0*alpha/sqrt(M_PI)*exp(-alphar*alphar);
                double ir3_eff = ir*ir*(ir_eff + derfc);
                double ir5_eff = ir*ir*(ir3_eff + 2.0/3.0*alpha*alpha*derfc);
                Vector3<double> rij((double)v.x, (double)v.y, (double)v.z);
                rij *= LATTICE_CONST/4.0;
                Vector3<double> m2(s_ASiteIsingVector[site2]);
                m2 *= 1e-7 * A_MOMENT * MU_B;
                double mr = m2.innerProduct(rij);
                Vector3<double> hdip(rij);
                hdip *= mr * ir5_eff * 3.0;
                m2 *= ir3_eff;
                hdip -= m2;

                double hdip_d1 = d1.innerProduct(hdip);
                ASSERT(fabs(hdip_d1) < 1.0);
    //            fprintf(stderr, "%g\n",h);

                //counts for real-space.
                cnt_n_r2[r2int]++;
                int didx = distance_index(di,dj,dk);
                s_is_inside_sphere_real[site1][site2][didx] = 1;
                s_fields_real[site1][site2][didx] += hdip_d1;
                
                cnt_n++;
                // Nearest neighbor.
                if(r2int <= 2*1*1) {
                    ASSERT(r2int == 2);
                    Vector3<double> d2(s_ASiteIsingVector[site2]);
    
                    d_nn += hdip_d1 / (K_B / (A_MOMENT * MU_B)) * 3.0 * (d1.innerProduct(d2));
                    s_fields_real[site1][site2][didx]
                         += J_NN * K_B / (A_MOMENT * MU_B) * 3.0 * (d1.innerProduct(d2));
    //                fprintf(stderr, "%g\n",g_fields[didx]);
                    cnt_nn++;
                    continue;
                }
                // Next nearest neighbor.
                if(r2int <= 6) {
                    cnt_nnn++;
                    continue;
                }
                // 3rd nearest neighbor.
                if(r2int <= 8) {
                    cnt_3nn++;
                    continue;
                }
            }
            }
            }
        }
    }
    s_4r2_neighbor.clear();
    for(int i = 0; i < (int)cnt_n_r2.size(); i++) {
        if(cnt_n_r2[i] == 0) continue;
//        fprintf(stderr, "neighbor r = %f, cnt = %f\n", sqrt((double)i)/4.0, cnt_n_r2[i]/16.0);
        s_4r2_neighbor.push_back(i);
    }
    ASSERT(cnt_n % 16 == 0);
    cnt_n /= 16;
    fprintf(stderr, "# of neighbors = %d\n", cnt_n);
    ASSERT(cnt_nn % 16 == 0);
    cnt_nn /= 16;
    ASSERT(cnt_nn % 6 == 0);
    fprintf(stderr, "# of nearest neighbors = %d\n", cnt_nn);
    ASSERT(cnt_nnn % 16 == 0);
    cnt_nnn /= 16;
    ASSERT(cnt_nnn % 6 == 0);
    fprintf(stderr, "# of next nearest neighbors = %d\n", cnt_nnn);
    ASSERT(cnt_3nn % 16 == 0);
    cnt_3nn /= 16;
    ASSERT(cnt_3nn % 6 == 0);
    fprintf(stderr, "# of 3rd nearest neighbors = %d\n", cnt_3nn);
    d_nn /= 16;
    d_nn /= cnt_nn;
    fprintf(stderr, "D_NN = %g [K]\n", d_nn);
        
    int rec_size = (2*cutoff_rec + 1)*(2*cutoff_rec + 1)*(cutoff_rec + 1);
    s_is_inside_sphere_rec.clear();
    s_is_inside_sphere_rec.resize(rec_size);
    for(int site1 = 0; site1 < 16; site1++) {
        Vector3<double> d1(s_ASiteIsingVector[site1]);

        for(int site2 = 0; site2 < 16; site2++) {
            Vector3<double> m2(s_ASiteIsingVector[site2]);
            m2 *= 4.0 * M_PI * 1e-7 * A_MOMENT * MU_B / LATTICE_VOLUME / (s_num_spins/16);

            s_fields_rec[site1][site2].clear();
            s_fields_rec[site1][site2].resize(rec_size);

            for(int kz = 0; kz <= cutoff_rec; kz++) {
            for(int ky = -cutoff_rec; ky <= cutoff_rec; ky++) {
            for(int kx = -cutoff_rec; kx <= cutoff_rec; kx++) {
                int k2int = kx*kx + ky*ky + kz*kz;
                int ridx = reciprocal_index(kx,ky,kz);
                if(k2int >= radius_rec*radius_rec) {
                    continue;
                }
                s_is_inside_sphere_rec[ridx] = 1;

                Vector3<double> k(kx,ky,kz);
                k *= 2.0*M_PI / (LATTICE_CONST * s_L);
                double k2 = k.innerProduct(k);
                double hdipq = 0.0;
                if(k2int == 0) {
                    // surface term.
                    hdipq = - dfactor * m2.innerProduct(d1);
                }
                else {
                    hdipq = - exp(-k2/(4.0*alpha*alpha)) / k2
                         * k.innerProduct(m2) * k.innerProduct(d1);
                }
                // summation of minus-k space.
                if(kz != 0)
                    hdipq *= 2.0;
                s_fields_rec[site1][site2][ridx] = hdipq;
            }
            }
            }
        }
    }
    // For self-energy correction.
    s_fields_rec_sum = accumulate(s_fields_rec[0][0].begin(), s_fields_rec[0][0].end(), 0.0);

    for(int site1 = 0; site1 < 16; site1++) {
    s_exp_ph[site1].clear();
    if(rec_size*s_num_spins*sizeof(double) < 500000000) {
        //create deep cache.
        s_exp_ph[site1].resize(rec_size * s_num_spins / 16);
    
        Vector3<double> pos1(cg_ASitePositions[site1]);
        pos1 *= LATTICE_CONST / 4.0;
        for(int k = 0; k < s_L; k++) {
        for(int j = 0; j < s_L; j++) {
        for(int i = 0; i < s_L; i++) {
            int lidx = lattice_index(i,j,k);
            double phx = 2*M_PI / (LATTICE_CONST * s_L) * (i * LATTICE_CONST + pos1.x);
            double phy = 2*M_PI / (LATTICE_CONST * s_L) * (j * LATTICE_CONST + pos1.y);
            double phz = 2*M_PI / (LATTICE_CONST * s_L) * (k * LATTICE_CONST + pos1.z);
            complex<double> exp_i_rx = exp(complex<double>(0.0, phx));
            complex<double> exp_i_ry = exp(complex<double>(0.0, phy));
            complex<double> exp_i_rz = exp(complex<double>(0.0, phz));
                
            complex<double> exp_ikrz = exp(complex<double>(0.0, -cutoff_rec * (phx + phy)));
            for(int kz = 0; kz <= cutoff_rec; kz++) {
            complex<double> exp_ikryz = exp_ikrz;
            for(int ky = -cutoff_rec; ky <= cutoff_rec; ky++) {
            complex<double> exp_ikr = exp_ikryz;
            for(int kx = -cutoff_rec; kx <= cutoff_rec; kx++) {
                int ridx = reciprocal_index(kx,ky,kz);
                s_exp_ph[site1][lidx * rec_size + ridx] = exp_ikr;
                exp_ikr *= exp_i_rx;
            }
            exp_ikryz *= exp_i_ry;
            }
            exp_ikrz *= exp_i_rz;
            }
            ASSERT(abs(s_exp_ph[site1][(lidx+1) * rec_size-1]
                /exp(complex<double>(0.0, (cutoff_rec) * (phx + phy + phz)))
                - complex<double>(1.0,0)) < 1e-8);
        }
        }
        }
    }
    }

    return cnt_n;
}

inline double
MonteCarlo::iterate_real_redirected(int cnt, int site1, int i, int j, int k, int site2)
{
    int cutoff = s_cutoff_real;
    ASSERT(cnt == cutoff*2 + 1);
    double h = 0.0;
    // note that spin images are repeated outside boundary along x.
    int *pspin_i = &m_spins_real[site2][spins_real_index(i - cutoff, 0, 0)];
    double *pfield = &s_fields_real[site1][site2][0];
    int *pinside = &s_is_inside_sphere_real[site1][site2][0];
    for(int dk = -cutoff; dk <= cutoff; dk++) {
        int *pspin_k = pspin_i + spins_real_index(-s_L, 0, (k+s_L+dk) % s_L);
        for(int dj = -cutoff; dj <= cutoff; dj++) {
            int *pspin = pspin_k + spins_real_index(-s_L, (j+s_L+dj) % s_L, 0);
//            ASSERT(pspin == &m_spins_real[site2][spins_real_index(i - cutoff,(j+s_L+dj) % s_L,(k+s_L+dk) % s_L)]);
            for(int n = 0; n < cnt; n++) {
                if(*pinside) {
                    double e = *pfield;
                    int spin = *pspin;
                    h += e * spin;
                }
                pspin++;
                pfield++;
                pinside++;
            }
//            ASSERT(pspin == &m_spins_real[site2][spins_real_index(i + cutoff + 1,(j+s_L+dj) % s_L,(k+s_L+dk) % s_L)]);
        }
    }
    ASSERT(pfield == &*s_fields_real[site1][site2].end());
    ASSERT(pinside == &*s_is_inside_sphere_real[site1][site2].end());
    return h;
}
inline double
MonteCarlo::iterate_real(int site1, int i, int j, int k, int site2)
{
    int cnt = 2 * s_cutoff_real + 1;
    // Trick to use constant count. use -funroll-loops. to enhance optimization.
    switch(cnt) {
    case 3:
        return iterate_real_redirected(3, site1, i, j, k, site2);
    case 5:
        return iterate_real_redirected(5, site1, i, j, k, site2);
    case 7:
        return iterate_real_redirected(7, site1, i, j, k, site2);
    case 9:
        return iterate_real_redirected(9, site1, i, j, k, site2);
    case 11:
        return iterate_real_redirected(11, site1, i, j, k, site2);
    default:
        return iterate_real_redirected(cnt, site1, i, j, k, site2);
    }
}
inline double
MonteCarlo::iterate_rec_redirected(int cutoff, int site1, int i, int j, int k, int site2)
{
    double h = 0.0;
    complex<double> *pspin = &m_spins_rec[site2][0];
    double *pfield = &s_fields_rec[site1][site2][0];
    int *pinside = &s_is_inside_sphere_rec[0];

    if(s_exp_ph[site1].size()) {
        complex<double> *pexp_ph =
            &s_exp_ph[site1][lattice_index(i,j,k) * s_is_inside_sphere_rec.size()];
        for(int m = 0; m < (cutoff+1)*(2*cutoff+1); m++) {
            for(int n = 0; n < 2*cutoff+1; n++) {
                if(*pinside) {
                    double e = *pfield;
                    complex<double> spin = *pspin;
                    h += e * real(spin * *pexp_ph);
                }
//                ASSERT(pfield == &s_fields_rec[site1][site2][reciprocal_index(kx,ky,kz)]);
                pspin++;
                pfield++;
                pinside++;
                pexp_ph++;
            }
        }
    }
    else {
        Vector3<double> pos1(cg_ASitePositions[site1]);
        pos1 *= LATTICE_CONST / 4.0;
        int lidx = lattice_index(i,j,k);
        double phx = 2*M_PI / (LATTICE_CONST * s_L) * (i * LATTICE_CONST + pos1.x);
        double phy = 2*M_PI / (LATTICE_CONST * s_L) * (j * LATTICE_CONST + pos1.y);
        double phz = 2*M_PI / (LATTICE_CONST * s_L) * (k * LATTICE_CONST + pos1.z);
        complex<double> exp_i_rx = exp(complex<double>(0.0, phx));
        complex<double> exp_i_ry = exp(complex<double>(0.0, phy));
        complex<double> exp_i_rz = exp(complex<double>(0.0, phz));
            
        complex<double> exp_ikrz = exp(complex<double>(0.0, -cutoff * (phx + phy)));
        for(int kz = 0; kz <= cutoff; kz++) {
            complex<double> exp_ikryz = exp_ikrz;
            for(int ky = -cutoff; ky <= cutoff; ky++) {
                complex<double> exp_ikr = exp_ikryz;
                for(int n = 0; n < 2*cutoff+1; n++) {
                    if(*pinside) {
                        double e = *pfield;
                        complex<double> spin = *pspin;
                        h += e * real(spin * exp_ikr);
                    }
                    pspin++;
                    pfield++;
                    pinside++;
                    exp_ikr *= exp_i_rx;
                }
                exp_ikryz *= exp_i_ry;
            }
            exp_ikrz *= exp_i_rz;
        }
    }
    // subtract self-energy.
    if(site1 == site2) {
        int spin_self = m_spins_real[site1][spins_real_index(i,j,k)];
        h -=  s_fields_rec_sum * spin_self;
    }
//    ASSERT(pspin == &*m_spins_rec[site2].end());
//    ASSERT(pfield == &*s_fields_rec[site1][site2].end());
    return h;
}
inline double
MonteCarlo::iterate_rec(int site1, int i, int j, int k, int site2)
{
    int cutoff = s_cutoff_rec;
    // Trick to use constant count. use -funroll-loops. to enhance optimization.
    switch(cutoff) {
    case 2:
        return iterate_rec_redirected(2, site1, i, j, k, site2);
    case 3:
        return iterate_rec_redirected(3, site1, i, j, k, site2);
    case 4:
        return iterate_rec_redirected(4, site1, i, j, k, site2);
    case 5:
        return iterate_rec_redirected(5, site1, i, j, k, site2);
    case 6:
        return iterate_rec_redirected(6, site1, i, j, k, site2);
    case 7:
        return iterate_rec_redirected(7, site1, i, j, k, site2);
    default:
        return iterate_rec_redirected(cutoff, site1, i, j, k, site2);
    }
}

inline double
MonteCarlo::iterate_interactions(int site1, int lidx, int site2)
{
    if(m_sec_cache_enabled) {
        if(m_field_sec_cached_sane[site2][lidx] & (1u << site1)) {
            m_sec_cache_hit++;
            return m_field_sec_cached[site1][site2][lidx];
        }
    }
    int n = lidx;
    int i = n % s_L;
    n /= s_L;
    int j = n % s_L;
    n /= s_L;
    int k = n;
    
    double h = iterate_rec(site1, i, j, k, site2);    
        
    if(m_third_cache_enabled &&
        (m_field_third_cached_sane[site2][lidx] & (1u << site1))) {
        m_third_cache_hit++;
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

void
MonteCarlo::execute()
{
  for(;;) {
    int site2 = m_hint_site2_left;
    if(site2 <= 1) {
       //spin lock
       if(m_hint_site2_left <= 1)
            pauseN(1); // for Hyper-Threading.
       if(!m_thread_pool_active) {
           m_thread_pool_cond.wait();
       }
       if(m_bTerminated)
            return;
       continue;
    }
    if(!m_hint_site2_left.compareAndSet(site2, site2 - 1))
        continue;
    site2--;
    ASSERT(site2 >= 1);

    readBarrier();
    double h = iterate_interactions(
        m_hint_spin1_site, m_hint_spin1_lattice_index, site2);
    m_hint_fields[site2] = h;
    memoryBarrier();
    m_hint_done[site2] = 2;
  }
}

//paralleled summation.
double
MonteCarlo::hinteraction_miscache_threading(int site1, int lidx)
{
    m_hint_spin1_site = site1;
    m_hint_spin1_lattice_index = lidx;
    for(int idx = 0; idx < 16; idx++) {
        m_hint_done[idx] = 0;
    }
    memoryBarrier();
    // this is trigger.
    m_hint_site2_left = 16;
    
    double h = 0.0;
    
    for(;;) {  
        int site2 = m_hint_site2_left;
        if(!m_hint_site2_left.compareAndSet(site2, site2 - 1))
            continue;
        site2--;
        ASSERT(site2 >= 0);
        
        h += iterate_interactions(site1, lidx, site2);
        m_hint_done[site2] = 1;
        
        if(site2 == 0) {
            break;
        }
      }
 
    readBarrier();
    for(int site2 = 15; site2 >= 0; site2--) {
        while(m_hint_done[site2] == 0) {
           //spin lock
            pauseN(1); // for Hyper-Threading.
            readBarrier();
        }
        if(m_hint_done[site2] > 1) {
            readBarrier();
            h += m_hint_fields[site2];
        }
    }
    return h;
}
double
MonteCarlo::hinteraction_miscache(int site1, int lidx)
{
    if(m_thread_pool_active)
        return hinteraction_miscache_threading(site1, lidx);
    double h = 0.0;
    for(int site2 = 0; site2 < 16; site2++) {  
        h += iterate_interactions(site1, lidx, site2);
    }
    return h;
}
