/***************************************************************************
        Copyright (C) 2002-2017 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/

#ifndef NLLSFIT_H
#define NLLSFIT_H

#include <valarray>
#include <array>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit_nlin.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_ieee_utils.h>
#include <gsl/gsl_version.h>

class NonLinearLeastSquare {
public:
    template <class Func>
    NonLinearLeastSquare(Func f,
        std::valarray<double> init_params, size_t n,
        unsigned int max_iterations = 30);
    NonLinearLeastSquare() = default;
    NonLinearLeastSquare(NonLinearLeastSquare &&) = default;
    NonLinearLeastSquare(const NonLinearLeastSquare &) = delete;
    NonLinearLeastSquare& operator=(NonLinearLeastSquare &&) = default;

    std::valarray<double> params() const {return m_params;}
    std::valarray<double> errors() const {return m_errors;}
    double chiSquare() const {return m_chisq;}
    std::string status() const {return gsl_strerror(m_status);}
    bool isSuccessful() const {return m_status == GSL_SUCCESS;}
private:
    std::valarray<double> m_params, m_errors;
    double m_chisq;
    int m_status;
};


template <class Func>
NonLinearLeastSquare::NonLinearLeastSquare(Func func,
    std::valarray<double> init_params, size_t n,
    unsigned int max_iterations) {
    const gsl_multifit_fdfsolver_type *T;
    T = gsl_multifit_fdfsolver_lmsder;
    gsl_multifit_fdfsolver *s;
    int iter = 0;
    int status;
    gsl_multifit_function_fdf f;

    gsl_ieee_env_setup ();

    struct USER {
        std::function<bool(const double*, size_t n, size_t p,
            double *f, std::vector<double *> &df)> func;
        size_t n, p;
        std::vector<double *> df;
    } user;
    user.func = func;
    user.n = n;
    user.p = init_params.size();
    std::valarray<double> df(init_params.size() * n);
    user.df.resize(init_params.size());
    for(size_t p = 0; p < init_params.size(); ++p)
        user.df[p] = &df[n * p];

    auto cb_f = [](const gsl_vector * x, void *params, gsl_vector * f) -> int {
        auto user = reinterpret_cast<USER*>(params);
        bool ret = user->func(gsl_vector_const_ptr(x, 0), user->n, user->p, gsl_vector_ptr(f, 0), user->df);
        return ret ? GSL_SUCCESS : GSL_ERUNAWAY;
    };
    auto cb_df = [](const gsl_vector * x, void *params, gsl_matrix * J) -> int {
        auto user = reinterpret_cast<USER*>(params);
        bool ret = user->func(gsl_vector_const_ptr(x, 0), user->n, user->p, nullptr, user->df);
        if( !ret) {
            J = nullptr;
            return GSL_SUCCESS;
        }
        for(size_t i = 0; i < user->n; ++i) {
            for(size_t p = 0; p < user->p; ++p)
                gsl_matrix_set(J, i, p, user->df[p][i]);
        }
        return GSL_SUCCESS;
    };

    f.f = cb_f;
    f.df = cb_df;
    f.fdf = nullptr;
    f.n = n;
    f.p = init_params.size();
    f.params = &user;
    s = gsl_multifit_fdfsolver_alloc (T, n, init_params.size());
    gsl_vector_view x = gsl_vector_view_array( &init_params[0], init_params.size());
    gsl_multifit_fdfsolver_set (s, &f, &x.vector);


    do {
        iter++;
        status = gsl_multifit_fdfsolver_iterate (s);

        if (status)
            break;

        status = gsl_multifit_test_delta (s->dx, s->x,
                                          1e-4, 1e-4);
    } while (status == GSL_CONTINUE && iter < max_iterations);

    m_chisq = pow(gsl_blas_dnrm2(s->f), 2.0);

    m_params = init_params;
    for(int i = 0; i < init_params.size(); i++)
        m_params[i] = gsl_vector_get(s->x, i);

//Computes covariance of best fit parameters
    gsl_matrix *covar = gsl_matrix_alloc (init_params.size(), init_params.size());
#if (GSL_MAJOR_VERSION >= 2)
    gsl_matrix *J = gsl_matrix_alloc(n, init_params.size());
    gsl_multifit_fdfsolver_jac(s, J);
    gsl_multifit_covar(J, 0.0, covar);
#else
    #error GSL < 2 is obsolete, because of poor fit.
    gsl_multifit_covar (s->J, 0.0, covar);
#endif
    m_errors.resize(init_params.size());
    for(int i = 0; i < init_params.size(); i++) {
        double c = gsl_matrix_get(covar,i,i);

        m_errors[i] = (c > 0) ? sqrt(c * m_chisq / n) : -1.0;
    }
    gsl_matrix_free(covar);
#if (GSL_MAJOR_VERSION >= 2)
    gsl_matrix_free(J);
#endif
    gsl_multifit_fdfsolver_free(s);

    m_status = status;
}


#endif // NLLSFIT_H
