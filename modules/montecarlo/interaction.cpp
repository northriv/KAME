/***************************************************************************
		Copyright (C) 2002-2012 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "montecarlo.h"
using std::vector;
using std::complex;
#include <numeric>
using std::accumulate;

void MonteCarlo::addFieldsReal(MonteCarlo::Spin v, FieldRealArray &array, int di, int dj, int dk) {
    int size = 2*s_cutoff_real + 1;
#ifdef PACK_4FLOAT
    int sizex = (2*s_cutoff_real + 3)/4 + 1;
    for(int al= 0; al < 4; al++) {
        int r = sizex*(size*(dk + s_cutoff_real) + (dj + s_cutoff_real)) + (di + s_cutoff_real + al) / 4;
        int c = (di + s_cutoff_real + al) % 4;
        array.align[al][r].x[c] += v;
    }
#else
    int r =  size*(size*(dk + s_cutoff_real) + (dj + s_cutoff_real)) + (di + s_cutoff_real);
    array[r] += v;
#endif
}

#include <pthread.h>
void *
MonteCarlo::xthread_start_routine(void *x)
{
    MonteCarlo *ptr = (MonteCarlo *)x;
    ptr->execute();
    return NULL;
}
int
MonteCarlo::dipoleFieldReal(const Vector3<double> &v, int site2, Vector3<double> *ret)
{
    double r2int = v.x*v.x + v.y*v.y + v.z*v.z;
    // spherical boundary
    if(r2int - 0.01 > (4*s_cutoff_real_radius)*(4*s_cutoff_real_radius)) {
        return false;
    }
    double r = LATTICE_CONST/4.0 * sqrt((double)r2int);
    double ir = 1.0/r;
    double alphar = s_alpha*r;
    double ir_eff = erfc_(alphar)*ir;
    double derfc = 2.0*s_alpha/sqrt(M_PI)*exp(-alphar*alphar);
    double ir3_eff = ir*ir*(ir_eff + derfc);
    double ir5_eff = ir*ir*(ir3_eff + 2.0/3.0*s_alpha*s_alpha*derfc);
    Vector3<double> rij((double)v.x, (double)v.y, (double)v.z);
    rij *= LATTICE_CONST/4.0;
    Vector3<double> m2(s_ASiteIsingVector[site2]);
    m2 *= 1e-7 * A_MOMENT * MU_B;
    double mr = m2.innerProduct(rij);
    Vector3<double> hdip(rij);
    hdip *= mr * ir5_eff * 3.0;
    m2 *= ir3_eff;
    hdip -= m2;
    *ret = hdip;
    return true;
}
int
MonteCarlo::dipoleFieldRec(const Vector3<double> &v, int site2, Vector3<double> *ret)
{
    double k2int = v.x*v.x + v.y*v.y + v.z*v.z;
    if(k2int >= s_cutoff_rec_radius*s_cutoff_rec_radius) {
        return false;
    }

    Vector3<double> m2(s_ASiteIsingVector[site2]);
    m2 *= 4.0 * M_PI * 1e-7 * A_MOMENT * MU_B / LATTICE_VOLUME / (s_num_spins/16);
    Vector3<double> k(v.x, v.y, v.z);
    k *= 2.0*M_PI / (LATTICE_CONST * s_L);
    Vector3<double> hdip(k);
    if(k2int == 0) {
        // surface term.
        hdip = m2;
        hdip *= - s_dfactor;
    }
    else {
        double k2 = k.innerProduct(k);
        hdip *= - exp(-k2/(4.0*s_alpha*s_alpha)) / k2 * k.innerProduct(m2);
    }
    // summation of minus-k space.
    if(v.z != 0)
        hdip *= 2.0;
    *ret = hdip;
    return true;
}
int
MonteCarlo::setupField(int size, double dfactor,
					   double radius, double radius_rec, double alpha)
{
    s_bAborting = false;
    s_L = size;
    s_num_spins = size*size*size*16;
    s_dfactor = dfactor;

    for(int site1 = 0; site1 < 16; site1++) {
        s_ASiteIsingVector[site1].x = cg_ASiteIsingAxes[site1][0] / sqrt(3.0);
        s_ASiteIsingVector[site1].y = cg_ASiteIsingAxes[site1][1] / sqrt(3.0);
        s_ASiteIsingVector[site1].z = cg_ASiteIsingAxes[site1][2] / sqrt(3.0);
    }
    
    int cutoff_real = (int)ceil(radius - 0.01);
    alpha /= LATTICE_CONST;
    s_alpha = alpha;
    double mag_cutoff = erfc_(alpha*LATTICE_CONST*radius);
    fprintf(stderr, "Magnitude at the cutoff boundary for real = %g%%\n", 100.0*mag_cutoff);
    s_cutoff_real = cutoff_real;
    s_cutoff_real_radius = radius;
//    double radius_rec = sqrt(-log(mag_cutoff)) * 2.0 * alpha / (2.0*M_PI / (LATTICE_CONST * s_L));
    int cutoff_rec = (int)ceil(radius_rec);
    s_cutoff_rec = cutoff_rec;
    s_cutoff_rec_radius = radius_rec;
    double mag_cutoff_rec = exp(-pow(radius_rec / (2.0 * alpha) * (2.0 * M_PI / (LATTICE_CONST * s_L)), 2.0)); 
    fprintf(stderr, "Magnitude at the cutoff boundary for rec = %g%%\n", 100.0*mag_cutoff_rec);

    for(int site2 = 0; site2 < 16; site2++) {
	#ifdef PACK_4FLOAT
        int size = (cutoff_real*2+1)*(cutoff_real*2+1)*((cutoff_real*2 + 3)/4 + 1);
        for(int al = 0; al < 4; al++) {
            for(int site1 = 0; site1 < 16; site1++) {
                s_fields_real[site1][site2].align[al].clear();
                s_fields_real[site1][site2].align[al].resize(size);
            }
            for(int d = 0; d < 3; d++) {
                for(int site1 = 0; site1 < 16; site1++) {
                    s_fields_real_B[site1][site2][d].align[al].clear();
                    s_fields_real_B[site1][site2][d].align[al].resize(size);
                }
                for(int site1 = 0; site1 < 8; site1++) {
                    s_fields_real_8a[site1][site2][d].align[al].clear();
                    s_fields_real_8a[site1][site2][d].align[al].resize(size);
                }
                for(int site1 = 0; site1 < 48; site1++) {
                    s_fields_real_48f[site1][site2][d].align[al].clear();
                    s_fields_real_48f[site1][site2][d].align[al].resize(size);
                }
            }
        }
	#else
        int size = (cutoff_real*2+1)*(cutoff_real*2+1)*(cutoff_real*2+1);
        for(int site1 = 0; site1 < 16; site1++) {
            s_fields_real[site1][site2].clear();
            s_fields_real[site1][site2].resize(size, 0.0);
        }
        for(int d = 0; d < 3; d++) {
            for(int site1 = 0; site1 < 16; site1++) {
                s_fields_real_B[site1][site2][d].clear();
                s_fields_real_B[site1][site2][d].resize(size, 0.0);
            }
            for(int site1 = 0; site1 < 8; site1++) {
                s_fields_real_8a[site1][site2][d].clear();
                s_fields_real_8a[site1][site2][d].resize(size, 0.0);
            }
            for(int site1 = 0; site1 < 48; site1++) {
                s_fields_real_48f[site1][site2][d].clear();
                s_fields_real_48f[site1][site2][d].resize(size, 0.0);
            }
        }
	#endif
    }


    int cnt_n = 0;
    int cnt_nn = 0;
    double d_nn = 0.0;
    int cnt_nnn = 0;
    int cnt_3nn = 0;
    vector<int> cnt_n_r2(4*(cutoff_real+1)*4*(cutoff_real+1));
    for(int site1 = 0; site1 < 16; site1++) {
        Vector3<double> d1(s_ASiteIsingVector[site1]);
        for(int site2 = 0; site2 < 16; site2++) {
            for(int dk = -cutoff_real; dk <= cutoff_real; dk++) {
				for(int dj = -cutoff_real; dj <= cutoff_real; dj++) {
					for(int di = -cutoff_real; di <= cutoff_real; di++) {                
						VectorInt v = distance(site1,site2,di,dj,dk);
						int r2int = v.x*v.x + v.y*v.y + v.z*v.z;
						if(r2int == 0) {
							continue;
						}
						Vector3<double> hdip;
						if(!dipoleFieldReal(v, site2, &hdip))
							continue;
						double hdip_d1 = d1.innerProduct(hdip);
						assert(fabs(hdip_d1) < 1.0);
						//            fprintf(stderr, "%g\n",h);

						//counts for real-space.
						cnt_n_r2[r2int]++;
						addFieldsReal(hdip_d1, s_fields_real[site1][site2], di, dj, dk);
                
						cnt_n++;
						// Nearest neighbor.
						if(r2int <= 2*1*1) {
							assert(r2int == 2);
							Vector3<double> d2(s_ASiteIsingVector[site2]);
    
							d_nn += hdip_d1 / (K_B / (A_MOMENT * MU_B)) * 3.0 * (d1.innerProduct(d2));
							addFieldsReal(
								J_NN * K_B / (A_MOMENT * MU_B) * 3.0 * (d1.innerProduct(d2))
								, s_fields_real[site1][site2], di, dj, dk);
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
    
    for(int site1 = 0; site1 < 16; site1++) {
        for(int site2 = 0; site2 < 16; site2++) {
            for(int dk = -cutoff_real; dk <= cutoff_real; dk++) {
				for(int dj = -cutoff_real; dj <= cutoff_real; dj++) {
					for(int di = -cutoff_real; di <= cutoff_real; di++) {                
						Vector3<double> v;
						v.x = 4*di + cg_ASitePositions[site2][0] - cg_BSitePositions[site1][0];
						v.y = 4*dj + cg_ASitePositions[site2][1] - cg_BSitePositions[site1][1];
						v.z = 4*dk + cg_ASitePositions[site2][2] - cg_BSitePositions[site1][2];
						Vector3<double> hdip;
						if(!dipoleFieldReal(v, site2, &hdip))
							continue;
						addFieldsReal(hdip.x, s_fields_real_B[site1][site2][0], di, dj, dk);
						addFieldsReal(hdip.y, s_fields_real_B[site1][site2][1], di, dj, dk);
						addFieldsReal(hdip.z, s_fields_real_B[site1][site2][2], di, dj, dk);
					}
				}
            }
        }
    }
    for(int site1 = 0; site1 < 8; site1++) {
        for(int site2 = 0; site2 < 16; site2++) {
            for(int dk = -cutoff_real; dk <= cutoff_real; dk++) {
				for(int dj = -cutoff_real; dj <= cutoff_real; dj++) {
					for(int di = -cutoff_real; di <= cutoff_real; di++) {                
						Vector3<double> v;
						v.x = 4*di + cg_ASitePositions[site2][0] - cg_8aSitePositions[site1][0] * 0.5;
						v.y = 4*dj + cg_ASitePositions[site2][1] - cg_8aSitePositions[site1][1] * 0.5;
						v.z = 4*dk + cg_ASitePositions[site2][2] - cg_8aSitePositions[site1][2] * 0.5;
						Vector3<double> hdip;
						if(!dipoleFieldReal(v, site2, &hdip))
							continue;
						double r2int = v.x*v.x + v.y*v.y + v.z*v.z;
						//Nearest Neighbor.
						if(r2int < 0.7501) {
							hdip *= AHF_DY_O1 / AHF_DY_O1_DIPOLE;
						}
						addFieldsReal(hdip.x, s_fields_real_8a[site1][site2][0], di, dj, dk);
						addFieldsReal(hdip.y, s_fields_real_8a[site1][site2][1], di, dj, dk);
						addFieldsReal(hdip.z, s_fields_real_8a[site1][site2][2], di, dj, dk);
					}
				}
            }
        }
    }
    for(int site1 = 0; site1 < 48; site1++) {
        for(int site2 = 0; site2 < 16; site2++) {
            for(int dk = -cutoff_real; dk <= cutoff_real; dk++) {
				for(int dj = -cutoff_real; dj <= cutoff_real; dj++) {
					for(int di = -cutoff_real; di <= cutoff_real; di++) {                
						Vector3<double> v;
						v.x = 4*di + cg_ASitePositions[site2][0] - cg_48fSitePositions[site1][0] * 0.5;
						v.y = 4*dj + cg_ASitePositions[site2][1] - cg_48fSitePositions[site1][1] * 0.5;
						v.z = 4*dk + cg_ASitePositions[site2][2] - cg_48fSitePositions[site1][2] * 0.5;
						Vector3<double> hdip;
						if(!dipoleFieldReal(v, site2, &hdip))
							continue;
						addFieldsReal(hdip.x, s_fields_real_48f[site1][site2][0], di, dj, dk);
						addFieldsReal(hdip.y, s_fields_real_48f[site1][site2][1], di, dj, dk);
						addFieldsReal(hdip.z, s_fields_real_48f[site1][site2][2], di, dj, dk);
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
    assert(cnt_n % 16 == 0);
    cnt_n /= 16;
    fprintf(stderr, "# of neighbors = %d\n", cnt_n);
    assert(cnt_nn % 16 == 0);
    cnt_nn /= 16;
    assert(cnt_nn % 6 == 0);
    fprintf(stderr, "# of nearest neighbors = %d\n", cnt_nn);
    assert(cnt_nnn % 16 == 0);
    cnt_nnn /= 16;
    assert(cnt_nnn % 6 == 0);
    fprintf(stderr, "# of next nearest neighbors = %d\n", cnt_nnn);
    assert(cnt_3nn % 16 == 0);
    cnt_3nn /= 16;
    assert(cnt_3nn % 6 == 0);
    fprintf(stderr, "# of 3rd nearest neighbors = %d\n", cnt_3nn);
    d_nn /= 16;
    d_nn /= cnt_nn;
    fprintf(stderr, "D_NN = %g [K]\n", d_nn);
        
    int rec_size = (2*cutoff_rec + 1)*(2*cutoff_rec + 1)*(cutoff_rec + 1);
    for(int site1 = 0; site1 < 16; site1++) {
        Vector3<double> d1(s_ASiteIsingVector[site1]);

        for(int site2 = 0; site2 < 16; site2++) {
            s_fields_rec[site1][site2].clear();
            s_fields_rec[site1][site2].resize(rec_size);

            for(int kz = 0; kz <= cutoff_rec; kz++) {
				for(int ky = -cutoff_rec; ky <= cutoff_rec; ky++) {
					for(int kx = -cutoff_rec; kx <= cutoff_rec; kx++) {
						VectorInt v(kx, ky, kz);
						Vector3<double> hdip;
						if(!dipoleFieldRec(v, site2, &hdip))
							continue;             
						int ridx = reciprocal_index(kx,ky,kz);
						s_fields_rec[site1][site2][ridx] = hdip.innerProduct(d1);
					}
				}
            }
        }
    }
    for(int site2 = 0; site2 < 16; site2++) {
        s_fields_rec_generic[site2].clear();
        s_fields_rec_generic[site2].resize(rec_size);

        for(int kz = 0; kz <= cutoff_rec; kz++) {
			for(int ky = -cutoff_rec; ky <= cutoff_rec; ky++) {
				for(int kx = -cutoff_rec; kx <= cutoff_rec; kx++) {
					VectorInt v(kx, ky, kz);
					Vector3<double> hdip;
					if(!dipoleFieldRec(v, site2, &hdip))
						continue;             
					int ridx = reciprocal_index(kx,ky,kz);
					s_fields_rec_generic[site2][ridx] = hdip;
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
						assert(abs(complex<double>(s_exp_ph[site1][(lidx+1) * rec_size-1])
								   /exp(complex<double>(0.0, (cutoff_rec) * (phx + phy + phz)))
								   - complex<double>(1.0,0)) < 1e-6);
					}
				}
			}
		}
    }

    return cnt_n;
}

inline double
MonteCarlo::iterate_real_redirected(int cnt, const FieldRealArray &array, int i, int j, int k, int site2)
{
    int cutoff = s_cutoff_real;
    assert(cnt == cutoff*2 + 1);
#ifdef PACK_4FLOAT
    int l = (s_L - 1) / 4 + 1;
    int al = spins_real_index(i - cutoff,0,0) % 4;
    PackedSpin h;
    PackedSpin *pspin_i = &m_spins_real[site2][spins_real_index(i - cutoff,0,0) / 4];
    asm ("movaps %0, %%xmm3"
         :
         : "m" (h)
         : "%xmm3");
    const PackedSpin *pfield = &array.align[al][0];
    for(int dk = -cutoff; dk <= cutoff; dk++) {
        PackedSpin *pspin_k = pspin_i + spins_real_index(-s_L, 0, (k+s_L+dk) % s_L) / 4;
        for(int dj = -cutoff; dj <= cutoff; dj++) {
            PackedSpin *pspin = pspin_k + spins_real_index(-s_L, (j+s_L+dj) % s_L, 0) / 4;
//            assert(pspin == &m_spins_real[site2][spins_real_index(i - cutoff,(j+s_L+dj) % s_L,(k+s_L+dk) % s_L)]);
            for(int n = 0; n < (cnt + 2) / 4 + 1; n++) {
                asm ("movaps %0, %%xmm1;"
					 "movaps %1, %%xmm2;"
					 "mulps %%xmm1, %%xmm2;"
					 "addps %%xmm2, %%xmm3"
                     :
                     : "m" (pspin[n]), "m" (pfield[n])
                     : "%xmm1", "%xmm2", "%xmm3");
//                PackedSpin x(pspin[n]);
//                x *= pfield[n];
//                h += x;
            }
            pfield+=(cnt + 2) / 4 + 1;
//            assert(pspin == &m_spins_real[site2][spins_real_index(i + cutoff + 1,(j+s_L+dj) % s_L,(k+s_L+dk) % s_L)]);
        }
    }
    asm ("movaps %%xmm3, %0"
         : "=m" (h)
         :
         : "%xmm3");
    assert(pfield == &*array.align[al].end());
    return h.sum();
#else
    Spin h = 0.0;
    // note that spin images are repeated outside boundary along x.
    Spin *pspin_i = &m_spins_real[site2][spins_real_index(i - cutoff, 0, 0)];
    const Spin *pfield = &array[0];
    for(int dk = -cutoff; dk <= cutoff; dk++) {
        Spin *pspin_k = pspin_i + spins_real_index(-s_L, 0, (k+s_L+dk) % s_L);
        for(int dj = -cutoff; dj <= cutoff; dj++) {
            Spin *pspin = pspin_k + spins_real_index(-s_L, (j+s_L+dj) % s_L, 0);
//            assert(pspin == &m_spins_real[site2][spins_real_index(i - cutoff,(j+s_L+dj) % s_L,(k+s_L+dk) % s_L)]);
            for(int n = 0; n < cnt; n++) {
                h += pfield[n] * pspin[n];
            }
            pfield+=cnt;
//            assert(pspin == &m_spins_real[site2][spins_real_index(i + cutoff + 1,(j+s_L+dj) % s_L,(k+s_L+dk) % s_L)]);
        }
    }
    assert(pfield == &*array.end());
    return h;
#endif
}

MonteCarlo::Vector3<double>
MonteCarlo::iterate_real_generic(const FieldRealArray array[16][3], int i, int j, int k)
{
    int cutoff = s_cutoff_real;
    int cnt = 2*cutoff + 1;
    Vector3<double> hall;
    for(int site2 = 0; site2 < 16; site2++) {
        hall.x += iterate_real_redirected(cnt, array[site2][0], i, j, k, site2);
        hall.y += iterate_real_redirected(cnt, array[site2][1], i, j, k, site2);
        hall.z += iterate_real_redirected(cnt, array[site2][2], i, j, k, site2);
    }
    return hall;
}

double
MonteCarlo::iterate_real(int site1, int i, int j, int k, int site2)
{
    int cnt = 2 * s_cutoff_real + 1;
    // Trick to use constant count. use -funroll-loops. to enhance optimization.
    switch(cnt) {
    case 3:
        return iterate_real_redirected(3, s_fields_real[site1][site2], i, j, k, site2);
    case 5:
        return iterate_real_redirected(5, s_fields_real[site1][site2], i, j, k, site2);
    case 7:
        return iterate_real_redirected(7, s_fields_real[site1][site2], i, j, k, site2);
    case 9:
        return iterate_real_redirected(9, s_fields_real[site1][site2], i, j, k, site2);
    case 11:
        return iterate_real_redirected(11, s_fields_real[site1][site2], i, j, k, site2);
    default:
        return iterate_real_redirected(cnt, s_fields_real[site1][site2], i, j, k, site2);
    }
}
MonteCarlo::Vector3<double>
MonteCarlo::iterate_rec_generic(Vector3<double> pos1, int i, int j, int k)
{
    Vector3<double> h;
    for(int site2 = 0; site2 < 16; site2++) {
        h += iterate_rec_generic(pos1, i, j, k, site2);
    }
    return h;
}
MonteCarlo::Vector3<double>
MonteCarlo::iterate_rec_generic(Vector3<double> pos1, int i, int j, int k, int site2)
{
    int cutoff = s_cutoff_rec;
    int cnt = 2*cutoff+1;
    pos1 *= LATTICE_CONST;
    complex<Spin> *pspin = &m_spins_rec[site2][0];
    const Vector3<Spin> *pfield = &s_fields_rec_generic[site2][0];
    
    double phx = 2*M_PI / (LATTICE_CONST * s_L) * (i * LATTICE_CONST + pos1.x);
    double phy = 2*M_PI / (LATTICE_CONST * s_L) * (j * LATTICE_CONST + pos1.y);
    double phz = 2*M_PI / (LATTICE_CONST * s_L) * (k * LATTICE_CONST + pos1.z);
    complex<Spin> exp_i_rx = exp(complex<Spin>(0.0, phx));
    complex<Spin> exp_i_ry = exp(complex<Spin>(0.0, phy));
    complex<Spin> exp_i_rz = exp(complex<Spin>(0.0, phz));
    Vector3<Spin> h;
    complex<Spin> exp_ikrz = exp(complex<Spin>(0.0, -cutoff * (phx + phy)));
    for(int kz = 0; kz <= cutoff; kz++) {
        complex<Spin> exp_ikryz = exp_ikrz;
        for(int ky = -cutoff; ky <= cutoff; ky++) {
            complex<Spin> exp_ikr = exp_ikryz;
            for(int n = 0; n < cnt; n++) {
                Vector3<Spin> e(pfield[n]);
                e *= real(pspin[n] * exp_ikr);
                h += e;
                exp_ikr *= exp_i_rx;
            }
            pspin+=cnt;
            pfield+=cnt;
            exp_ikryz *= exp_i_ry;
        }
        exp_ikrz *= exp_i_rz;
    }
    return h;
}
inline double
MonteCarlo::iterate_rec_redirected(int cutoff, int site1, int i, int j, int k, int site2)
{
    int cnt = 2*cutoff+1;
    Spin h = 0.0;
    complex<Spin> *pspin = &m_spins_rec[site2][0];
    Spin *pfield = &s_fields_rec[site1][site2][0];

    if(s_exp_ph[site1].size()) {
        complex<Spin> *pexp_ph =
            &s_exp_ph[site1][lattice_index(i,j,k) * m_spins_rec[site2].size()];
        for(int m = 0; m < (cutoff+1)*(2*cutoff+1); m++) {
            for(int n = 0; n < cnt; n++) {
                h += pfield[n] * real(pspin[n] * pexp_ph[n]);
//                assert(pfield == &s_fields_rec[site1][site2][reciprocal_index(kx,ky,kz)]);
            }
            pspin+=cnt;
            pfield+=cnt;
            pexp_ph+=cnt;
        }
    }
    else {
        Vector3<double> pos1(cg_ASitePositions[site1]);
        pos1 *= 1.0/4.0;
        h = iterate_rec_generic(pos1, i, j, k, site2)
            .innerProduct(s_ASiteIsingVector[site1]);
    }
    // subtract self-energy.
    if(site1 == site2) {
        Spin spin_self = readSpin(site1, spins_real_index(i,j,k));
        h -=  s_fields_rec_sum * spin_self;
    }
//    assert(pspin == &*m_spins_rec[site2].end());
//    assert(pfield == &*s_fields_rec[site1][site2].end());
    return h;
}
double
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


// Thread pool.
void
MonteCarlo::execute()
{
	for(;;) {
		int left = m_hint_site2_left;
		if(left <= 1) {
			XScopedLock<XCondition> lock(m_thread_pool_cond);
			if(m_bTerminated)
				break;
			m_thread_pool_cond.wait();
			continue;
		}
		if(!m_hint_site2_left.compareAndSet(left, left - 1))
			continue;
    
		int site2 = m_hint_sec_cache_miss[left - 1];
		assert(site2 < 16);

		readBarrier();
		m_hint_fields[site2] = iterate_interactions(
			m_hint_spin1_site, m_hint_spin1_lattice_index, site2);
		memoryBarrier();
		if(m_hint_site2_not_done.decAndTest()) {
			XScopedLock<XCondition> lock(m_hint_site2_last_cond);
			m_hint_site2_last_cond.signal();
		}
	}
}

double
MonteCarlo::hinteraction_miscache(int sec_cache_miss_cnt, int site1, int lidx)
{
    double h = 0.0;
    
// threaded summation.

    m_hint_spin1_site = site1;
    m_hint_spin1_lattice_index = lidx;
//    memoryBarrier();
    m_hint_site2_not_done = sec_cache_miss_cnt;
    // this is trigger.
    m_hint_site2_left = sec_cache_miss_cnt;
	{
//    XScopedLock<XCondition> lock(m_thread_pool_cond);
		m_thread_pool_cond.broadcast();
	}

	for(;;) {
		int left = m_hint_site2_left;
		if(left == 0) {
			XScopedLock<XCondition> lock(m_hint_site2_last_cond);
			while(m_hint_site2_not_done > 0) {
				m_hint_site2_last_cond.wait();
			}
			break;
		}
		if(!m_hint_site2_left.compareAndSet(left, left - 1))
			continue;
        
		int site2 = m_hint_sec_cache_miss[left - 1];
		assert(site2 < 16);
        
		m_hint_fields[site2] = iterate_interactions(site1, lidx, site2);
		if(m_hint_site2_not_done.decAndTest()) {            
			readBarrier();
			break;
		}
	}
 
	for(int miss = 0; miss < sec_cache_miss_cnt; miss++) {  
		int site2 = m_hint_sec_cache_miss[miss];
		h += m_hint_fields[site2];
	}
	return h;
}

