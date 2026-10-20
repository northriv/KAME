/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/
//! The searches of the LC auto-tuner that see through backlash, kept free of
//! KAME so that a simulator can run the very same code.
//!
//! Both read the fitted capacitances as the capacitors' own positions, never
//! read a gain off a move without allowing for the gear's slack, and learn
//! the slack from what did not arrive.  They were chosen and checked on a
//! simulator of the circuit, two STMs with play, cross-coupling, a curved
//! capacitance(angle) and fit noise: over 384 scenarios each (Q 50 to 1000,
//! play 0 to 60 deg.) neither failed, against 13 failures of the first
//! Alternate search and 61 of Newton's on a grid with other mechanics.
//!  - Alternate: one STM at a time, as by hand.  Per STM the gain, the play
//!    and the side the slack started on are identified jointly by replaying
//!    a play operator over every move of that STM; a step acts on the
//!    posterior medians.  6.0 moves on average.
//!  - Servo: both STMs at once.  Columns from probes without back legs, one
//!    clean verification leg, then Newton steps to the closed-form match,
//!    gains learned only from moves with the gear engaged, a play only from
//!    reversals.  5.1 moves on average.
//!
//! A search is written as the plain loop it is -- measure, decide, move --
//! against a rig.  The tuner, which gets one fit per sweep, keeps only the
//! history (Log) and next() replays it: the search runs from the start on
//! the recorded fits and moves until it asks for something not yet recorded,
//! which is the step to take.  A search is a pure function of the fits it
//! has seen, so the replay retraces it exactly, and a transaction that is
//! retried computes the same step again.
#ifndef LCTUNERSEARCH_H_
#define LCTUNERSEARCH_H_

#include <complex>
#include <cmath>
#include <cfloat>
#include <array>
#include <vector>
#include <algorithm>

namespace lctuner {

//! LCRFit's circuit: a series L1-R1-C1 branch in parallel with C2, in series
//! with R2, at a 50 Ohm port.  The formulas LCRFit fits with.
struct Circuit {
    double r1 = 0, r2 = 0, l1 = 0, c1 = 0, c2 = 0;
    double c1err = 0, c2err = 0; //!< 1-sigma of the fitted c1, c2
    std::complex<double> zlcr1(double omega) const {
        return std::complex<double>(r1, omega * l1 - 1.0 / (omega * c1));
    }
    //! Reflection, with the LCR branch's admittance given.
    std::complex<double> rl(double omega, std::complex<double> zlcr1_inv) const {
        auto zL = 1.0 / (std::complex<double>(0.0, omega * c2) + zlcr1_inv) + r2;
        return (zL - 50.0) / (zL + 50.0);
    }
    std::complex<double> rl(double omega) const {
        return rl(omega, 1.0 / zlcr1(omega));
    }
    //! Resonance freq.
    double f0() const {
        double f = 1.0 / 2 / M_PI / sqrt(l1 * c1);
        for(int it = 0; it < 5; ++it)
            f = 1.0 / 2 / M_PI / sqrt(l1 * c1 -
                c1 * c2 / (1.0/pow(50.0 + r2, 2.0) + pow(2 * M_PI * f * c2, 2.0)));
        return f;
    }
    //! |RL| at the resonance with a sign: positive while the coupling is
    //! loose (the resonance circle leaves the origin outside), negative once
    //! it is tight, zero exactly at critical coupling -- so the match is a
    //! root to find, like the frequency, not a minimum.  RL projected on the
    //! reflection with the LCR branch open, i.e. far off resonance.  Not the
    //! sign of Im RL(f0): that changes elsewhere than at the match (by 2.4%
    //! of C2 in a Q=50 model, where the match itself is at |RL| 0.001).
    double signedMatch() const {
        double omega = 2.0 * M_PI * f0();
        auto g = rl(omega), goff = rl(omega, 0.0);
        return std::real(g * std::conj(goff)) / std::abs(goff);
    }
    //! d(f0, signedMatch)/d(C1, C2), by finite differences:
    //! [0][j] for the resonant frequency [Hz/F], [1][j] for the match [1/F].
    std::array<std::array<double, 2>, 2> sensitivities() const {
        std::array<std::array<double, 2>, 2> s;
        double f = f0(), m = signedMatch();
        for(int j: {0, 1}) {
            Circuit t( *this);
            double &c = j ? t.c2 : t.c1;
            double dc = c * 1e-5;
            c += dc;
            s[0][j] = (t.f0() - f) / dc;
            s[1][j] = (t.signedMatch() - m) / dc;
        }
        return s;
    }
    //! RL = 0 at \a f, in closed form.  zL = 50 means
    //!   j w C2 + 1/(R1 + j X) = G := 1/(50 - R2), real, X = w L1 - 1/(w C1);
    //! the real part fixes X, the imaginary part then C2:
    //!   X = sqrt(R1/G - R1^2),  w C2 = X G / R1,  C1 = 1/(w (w L1 - X)).
    //! The other root, X < 0, would need C2 < 0, so the match is unique and
    //! tight/loose coupling has no say in it.  False when there is none: R1
    //! not below 50 - R2, or w L1 not above X.
    bool matchingCaps(double f, double &o1, double &o2) const {
        double omega = 2.0 * M_PI * f;
        double g = 1.0 / (50.0 - r2);
        double xsq = r1 / g - r1 * r1;
        if( !(g > 0.0) || !(xsq > 0.0))
            return false;
        double x = sqrt(xsq);
        if( !(omega * l1 > x))
            return false;
        o1 = 1.0 / (omega * (omega * l1 - x));
        o2 = x * g / (r1 * omega);
        return true;
    }
};

//! What a search did and saw, in order: a fit after every move, and as many
//! more as it asked for at one position.
struct Event {
    bool isFit = true;
    Circuit fit;
    double d[2] = {0.0, 0.0}; //!< [deg] a move of both STMs; one of them may be 0
};
using Log = std::vector<Event>;

enum class Search {Alternate, Servo};

//! The step to take.
struct Step {
    enum class Kind {Move, Measure, GiveUp};
    Kind kind = Kind::Measure;
    double d[2] = {0.0, 0.0}; //!< [deg], Kind::Move
};

//! Ends a search that has no way left to try.
struct GiveUp {};

namespace detail {

//! Most moves a search may take before it counts as failed.
constexpr int MaxMoves = 60;

//! A rig that plays back a Log, and stops the search at the first thing the
//! log does not hold yet.
class ReplayRig {
public:
    struct Frontier {bool wantsFit; double d[2];};
    struct Diverged {}; //!< the search did not retrace its own log
    ReplayRig(double ft, const Log &log) : m_ft(ft), m_log(log) {}
    double targetFreq() const {return m_ft;}
    Circuit measure() {
        if(m_pos == m_log.size())
            throw Frontier{true, {0.0, 0.0}};
        const Event &e = m_log[m_pos++];
        if( !e.isFit)
            throw Diverged{};
        return e.fit;
    }
    void move(int axis, double dtheta) {
        double d[2] = {0.0, 0.0};
        d[axis ? 1 : 0] = dtheta;
        moveBoth(d[0], d[1]);
    }
    void moveBoth(double d0, double d1) {
        if( !std::isfinite(d0) || !std::isfinite(d1))
            throw GiveUp{};
        if((d0 == 0.0) && (d1 == 0.0))
            return;
        if(m_pos == m_log.size()) {
            if(m_moves >= MaxMoves)
                throw GiveUp{};
            throw Frontier{false, {d0, d1}};
        }
        const Event &e = m_log[m_pos++];
        if(e.isFit || (e.d[0] != d0) || (e.d[1] != d1))
            throw Diverged{};
        m_th[0] += d0; m_th[1] += d1;
        ++m_moves;
    }
    //! The commanded angle, from where the search started [deg].
    double position(int axis) const {return m_th[axis ? 1 : 0];}
private:
    double m_ft;
    const Log &m_log;
    size_t m_pos = 0;
    int m_moves = 0;
    double m_th[2] = {0.0, 0.0};
};

//===========================================================================
// Alternate: baseline Alternate's control law, with what it learns replaced.
// Per STM the gain k, the play b and the initial slack side are identified
// jointly by replaying the play operator over every move of that STM (probe
// legs, same-direction moves, reversals, moves fully eaten by the slack
// alike); the step acts on the posterior medians of the gain and of the
// slack left, over the set S_j of hypotheses within chi2_min + 4.  The
// survey has no back legs.
namespace alternate {

//! One single-axis move of an STM and what the fits saw across it.
struct Move {
    double dth;             //!< commanded [deg]
    double dx[2];           //!< change of x = c/Cref - 1
    double sb[2], sa[2];    //!< 1-sigma of x before / after
    bool seen;              //!< a 5-sigma change in either capacitance
    bool rev;               //!< opposite to the STM's previous move
    double a = 0, sig = 1;  //!< arrival along u and its 1-sigma
};

//! A hypothesis on the play of one STM, and the gain fitted under it.
struct Hyp {
    double b = 0, f = 0;          //!< play [deg]; initial offset y0 = f * b / 2
    double k = 0, k2 = 0;         //!< x per deg (along u); quadratic term
    double chi2 = DBL_MAX;
    double yEnd = 0, phiEnd = 0;  //!< phi - theta, and phi - theta0, after the last move
    bool kdef = false;            //!< some move engaged the gear under it
    //! the slack left before a move in direction s engages
    double rem(double s) const {return std::max(0.0, b / 2 + s * yEnd);}
};

template <class Rig>
class Tuner {
public:
    explicit Tuner(Rig &r) : rig(r) {}

    void run() {
        Circuit fit = rig.measure();
        Cref[0] = fit.c1; Cref[1] = fit.c2;
        xcur[0] = 0; xcur[1] = 0; scur[0] = fit.c1err / Cref[0]; scur[1] = fit.c2err / Cref[1];
        //1. the survey, without back legs
        for(int j = 0; j < 2; ++j) {
            double leg = 10, cum = 0;
            for(;;) {
                fit = doMove(j, leg);
                cum += fabs(leg);
                const Move &m = ax[j].log.back();
                if(m.seen) {
                    double n = hypot(m.dx[0], m.dx[1]), sg = leg > 0 ? 1 : -1;
                    ax[j].u[0] = sg * m.dx[0] / n; ax[j].u[1] = sg * m.dx[1] / n;
                    ax[j].hasU = true;
                    afterMove(j);
                    break;
                }
                leg *= 2;
                if(cum + fabs(leg) > 720)
                    throw GiveUp{};
            }
        }
        auto jacOf = [&](const Circuit &f, int e, int i) {
            auto s = f.sensitivities(); return s[e][0] * D(0, i) + s[e][1] * D(1, i);
        };
        //The pairing that makes alternate steps converge.  A slack that
        //shortened an STM's test scales both entries of its column alike, so
        //it cancels out of this test.
        int fa = (fabs(jacOf(fit, 0, 0) * jacOf(fit, 1, 1)) >= fabs(jacOf(fit, 0, 1) * jacOf(fit, 1, 0))) ? 0 : 1;
        double ft = rig.targetFreq();
        int last = -1, pinned = 0;
        for(;;) {
            if(last >= 0)
                ax[last].errAfter = (last == fa) ? fit.f0() - ft : fit.signedMatch();
            int axis = (last < 0) ? fa : 1 - last;
            bool freq = (axis == fa);
            Ax &A = ax[axis];
            double err = freq ? fit.f0() - ft : fit.signedMatch();
            double gain = jacOf(fit, freq ? 0 : 1, axis);
            if( !std::isfinite(gain) || gain == 0)
                throw GiveUp{};
            double mv = -err / gain;
            double lim = std::min(720.0, 4 * A.maxSeen);
            //the safety net: halve an STM that keeps reversing without gaining
            size_t n = A.log.size();
            double errNow = A.errAfter;
            if(n >= 2 && A.log[n - 1].rev && A.log[n - 2].rev && std::isfinite(A.errAtLast) && std::isfinite(errNow)
                && fabs(errNow) > 0.5 * fabs(A.errAtLast))
                A.g = std::max(A.g / 2, 1.0 / 8);
            mv *= A.g;
            if(fabs(mv) > lim) mv = mv > 0 ? lim : -lim;
            //never past where a capacitor last answered the right way
            double th = rig.position(axis);
            if(th + mv > A.thHi) mv = A.thHi - th;
            if(th + mv < A.thLo) mv = A.thLo - th;
            if(fabs(mv) < 1e-9) {
                //pinned at the extremum it found: leave it to the other STM
                if(++pinned >= 2)
                    throw GiveUp{}; //both pinned: nothing left to try
                last = axis;
                continue;
            }
            pinned = 0;
            //the play's inverse: the slack left the way of this move
            double s = mv >= 0 ? 1 : -1;
            int si = s > 0 ? 0 : 1;
            double comp = A.remLo[si];
            //moves just made this way that did not show: the gear may still be
            //inside the play; double, as the survey does, as far as S_j allows
            double eaten = 0;
            for(size_t i = n; i-- > 0;) {
                const Move &m = A.log[i];
                if(m.seen || m.dth * s <= 0) break;
                eaten += fabs(m.dth);
            }
            comp = std::max(comp, std::min(A.remHi[si], eaten));
            mv += s * comp;
            if(fabs(mv) > 720) mv = mv > 0 ? 720 : -720;
            A.errAtLast = err;
            double thBefore = rig.position(axis);
            fit = doMove(axis, mv);
            afterMove(axis);
            const Move &m = A.log.back();
            //a capacitor that answered an engaged move backwards is past its extremum
            if(m.seen && !m.rev && m.a * m.dth < 0 && fabs(m.a) > 5 * m.sig) {
                if(m.dth > 0) A.thHi = std::min(A.thHi, thBefore); else A.thLo = std::max(A.thLo, thBefore);
            }
            if(m.seen && !m.rev) A.g = std::min(1.0, 1.5 * A.g);
            last = axis;
        }
    }

private:
    struct Ax {
        double u[2] = {0, 0}; bool hasU = false;
        std::vector<Move> log;
        double maxSeen = 0, g = 1, errAtLast = NAN, errAfter = NAN;
        double thLo = -DBL_MAX, thHi = DBL_MAX; //!< where the capacitor still answers the right way
        int lastDir = 0;
        Hyp best, lo; double kHi = 0, kStep = 0, bLo = 0; bool useK2 = false;
        double remLo[2] = {0, 0}, remHi[2] = {0, 0}; //!< slack left for a + / - move, over S_j
    };
    Rig &rig;
    double Cref[2], xcur[2], scur[2];
    Ax ax[2];

    //! dC_i/dtheta_j [F/deg] from the learned gain
    double D(int i, int j) const {return ax[j].kStep * ax[j].u[i] * Cref[i];}

    Circuit doMove(int j, double dth) {
        double xb[2] = {xcur[0], xcur[1]}, sb[2] = {scur[0], scur[1]};
        Ax &A = ax[j];
        bool rev = A.lastDir != 0 && dth * A.lastDir < 0;
        A.lastDir = dth > 0 ? 1 : -1;
        rig.move(j, dth);
        Circuit f = rig.measure();
        xcur[0] = f.c1 / Cref[0] - 1; xcur[1] = f.c2 / Cref[1] - 1;
        scur[0] = f.c1err / Cref[0]; scur[1] = f.c2err / Cref[1];
        Move m; m.dth = dth; m.rev = rev;
        for(int i = 0; i < 2; ++i) {m.dx[i] = xcur[i] - xb[i]; m.sb[i] = sb[i]; m.sa[i] = scur[i];}
        m.seen = fabs(m.dx[0]) > 5 * hypot(m.sb[0], m.sa[0]) || fabs(m.dx[1]) > 5 * hypot(m.sb[1], m.sa[1]);
        A.log.push_back(m);
        return f;
    }

    void arrivals(Ax &A) {
        for(auto &m: A.log) {
            m.a = m.dx[0] * A.u[0] + m.dx[1] * A.u[1];
            double v = 0;
            for(int i = 0; i < 2; ++i) v += A.u[i] * A.u[i] * (m.sb[i] * m.sb[i] + m.sa[i] * m.sa[i]);
            m.sig = sqrt(v);
        }
    }

    void afterMove(int j) {
        Ax &A = ax[j];
        if( !A.hasU) return;
        //refines u: the arrival-weighted mean direction of the seen moves
        arrivals(A);
        double v[2] = {0, 0};
        for(auto &m: A.log) if(m.seen) {v[0] += m.a * m.dx[0]; v[1] += m.a * m.dx[1];}
        double n = hypot(v[0], v[1]);
        if(n > 0) {A.u[0] = v[0] / n; A.u[1] = v[1] / n;}
        arrivals(A);
        A.maxSeen = 0;
        for(auto &m: A.log) if(m.seen) A.maxSeen = std::max(A.maxSeen, fabs(m.dth));
        fitAxis(A);
    }

    //! replays the play operator under (b, f) and fits the gain
    Hyp eval(const Ax &A, double b, double f) const {
        Hyp h; h.b = b; h.f = f;
        size_t n = A.log.size();
        double th = 0, phi = f * b / 2; //relative to where the search started
        double Sdd = 0, Sdq = 0, Sqq = 0, Sad = 0, Saq = 0, Saa = 0, pmin = phi, pmax = phi;
        for(size_t i = 0; i < n; ++i) {
            const Move &m = A.log[i];
            th += m.dth;
            double p = std::min(std::max(phi, th - b / 2), th + b / 2);
            double d = p - phi, q = p * p - phi * phi, w = 1 / (m.sig * m.sig);
            Sdd += w * d * d; Sdq += w * d * q; Sqq += w * q * q; Sad += w * m.a * d; Saq += w * m.a * q; Saa += w * m.a * m.a;
            phi = p; pmin = std::min(pmin, p); pmax = std::max(pmax, p);
        }
        h.yEnd = phi - th; h.phiEnd = phi;
        double k = 0, k2 = 0;
        if(Sdd > 0) {
            h.kdef = true;
            k = Sad / Sdd;
            if(A.useK2 && pmax > pmin) {
                //ridge towards a linear gear
                double sk2 = std::max(fabs(k), 1e-300) / (2 * (pmax - pmin));
                double a11 = Sdd, a12 = Sdq, a22 = Sqq + 1 / (sk2 * sk2), det = a11 * a22 - a12 * a12;
                if(det > 0) {k = (a22 * Sad - a12 * Saq) / det; k2 = (a11 * Saq - a12 * Sad) / det;}
                if(k < 0) {k = 0; k2 = Saq / (Sqq + 1 / (sk2 * sk2));}
                h.chi2 = Saa - 2 * k * Sad - 2 * k2 * Saq + k * k * Sdd + 2 * k * k2 * Sdq + k2 * k2 * Sqq + k2 * k2 / (sk2 * sk2);
            }
            else {
                if(k < 0) k = 0;
                h.chi2 = Saa - 2 * k * Sad + k * k * Sdd;
            }
        }
        else h.chi2 = Saa;
        h.k = k; h.k2 = k2;
        return h;
    }

    void fitAxis(Ax &A) {
        //k2 only once 3+ clean seen moves sample the gain at different places
        {
            std::vector<double> mid, len; double th = 0;
            for(size_t i = 0; i < A.log.size(); ++i) {
                const Move &m = A.log[i];
                bool clean = i > 0 && !m.rev && m.seen && A.log[i - 1].seen;
                if(clean) {mid.push_back(th + m.dth / 2); len.push_back(fabs(m.dth));}
                th += m.dth;
            }
            A.useK2 = false;
            if(mid.size() >= 3) {
                double lo = *std::min_element(mid.begin(), mid.end()), hi = *std::max_element(mid.begin(), mid.end());
                double sh = *std::min_element(len.begin(), len.end());
                A.useK2 = (hi - lo) >= 2 * sh;
            }
        }
        double B = 0; for(auto &m: A.log) B += fabs(m.dth);
        const int NG = 41;
        //the slack starts at one end of the play: every motor move leaves it there
        const double fs[2] = {-1, 1};
        const int NF = 2;
        std::vector<Hyp> all; all.reserve(NF * NG + 400);
        std::vector<std::vector<Hyp>> pts(NF); //per initial side: grid points plus refined minima
        Hyp best; bool have = false;
        auto tiny = [](double c) {return 1e-9 * (1 + fabs(c));};
        auto better = [&](const Hyp &h, const Hyp &b) {
            return h.chi2 < b.chi2 - tiny(b.chi2) || (fabs(h.chi2 - b.chi2) <= tiny(b.chi2) && h.b < b.b);
        };
        const double gr = (sqrt(5.0) - 1) / 2;
        for(int fi = 0; fi < NF; ++fi) {
            std::vector<Hyp> g(NG);
            for(int i = 0; i < NG; ++i) g[i] = eval(A, B * i / (NG - 1), fs[fi]);
            pts[fi] = g;
            //golden-section refinement of b around every local minimum of the
            //grid: with a good SNR a valley is narrower than the grid step
            if(B > 0) for(int i = 0; i < NG; ++i) {
                bool dip = (i == 0 || g[i].chi2 < g[i - 1].chi2 - tiny(g[i - 1].chi2)) && (i == NG - 1 || g[i].chi2 <= g[i + 1].chi2);
                if( !dip) continue;
                double lo = g[std::max(0, i - 1)].b, hi = g[std::min(NG - 1, i + 1)].b;
                double c = hi - gr * (hi - lo), d = lo + gr * (hi - lo);
                Hyp hc = eval(A, c, fs[fi]), hd = eval(A, d, fs[fi]);
                for(int it = 0; it < 40; ++it) {
                    if(hc.chi2 <= hd.chi2) {hi = d; d = c; hd = hc; c = hi - gr * (hi - lo); hc = eval(A, c, fs[fi]);}
                    else {lo = c; c = d; hc = hd; d = lo + gr * (hi - lo); hd = eval(A, d, fs[fi]);}
                }
                pts[fi].push_back(hc.chi2 <= hd.chi2 ? hc : hd);
            }
            std::sort(pts[fi].begin(), pts[fi].end(), [](const Hyp &a, const Hyp &b) {return a.b < b.b;});
            for(const Hyp &h: pts[fi]) {all.push_back(h); if( !have || better(h, best)) {best = h; have = true;}}
        }
        double T = best.chi2 + 4;
        //S_j's boundaries in b, refined by bisection, per initial side
        bool openEnd = false; //S_j reaches the largest play the moves could hide: no upper bound
        for(int fi = 0; fi < NF; ++fi) {
            auto &P = pts[fi];
            if(P.back().chi2 <= T) openEnd = true;
            for(size_t i = 1; i < P.size(); ++i) {
                bool in = P[i].chi2 <= T, inPrev = P[i - 1].chi2 <= T;
                if(in == inPrev) continue;
                double lo = P[i - 1].b, hi = P[i].b;
                for(int it = 0; it < 40; ++it) {
                    double m = 0.5 * (lo + hi); Hyp h = eval(A, m, fs[fi]);
                    if(h.chi2 <= T) all.push_back(h);
                    if((h.chi2 <= T) == inPrev) lo = m; else hi = m;
                }
            }
        }
        Hyp lo; bool haveLo = false; double kHi = 0;
        double remLo[2] = {DBL_MAX, DBL_MAX}, remHi[2] = {0, 0};
        for(const Hyp &h: all) {
            if(h.chi2 > T) continue;
            if( !haveLo || h.b < lo.b - 1e-12 || (fabs(h.b - lo.b) <= 1e-12 && h.chi2 < lo.chi2)) {lo = h; haveLo = true;}
            if(h.kdef) kHi = std::max(kHi, h.k);
            for(int si = 0; si < 2; ++si) {
                double r = h.rem(si ? -1 : 1);
                remLo[si] = std::min(remLo[si], r); remHi[si] = std::max(remHi[si], r);
            }
        }
        if(openEnd) remHi[0] = remHi[1] = DBL_MAX;
        //posterior medians (the gain, the slack left each way): each initial
        //side gets the same prior mass, flat in b over its part of S_j, times
        //the likelihood exp(-chi2/2)
        double kMed = best.k, remMed[2] = {remLo[0], remLo[1]};
        {
            struct W {double k, r0, r1, w;};
            std::vector<W> sm;
            for(int fi = 0; fi < NF; ++fi) {
                double bmin = DBL_MAX, bmax = -DBL_MAX;
                for(const Hyp &h: all) if(h.f == fs[fi] && h.chi2 <= T) {bmin = std::min(bmin, h.b); bmax = std::max(bmax, h.b);}
                if(bmin > bmax) continue;
                const int NS = 41;
                double db = (bmax - bmin) / (NS - 1);
                for(int i = 0; i < NS; ++i) {
                    Hyp h = eval(A, bmin + db * i, fs[fi]);
                    if(h.chi2 > T) continue;
                    double w = exp(-0.5 * (h.chi2 - best.chi2)) * (1.0 / NS);
                    sm.push_back({h.kdef ? h.k : -1, h.rem(1), h.rem(-1), w});
                    if(db <= 0) break;
                }
            }
            auto wmedian = [&](auto key, double dflt) {
                std::vector<std::pair<double, double>> v;
                for(auto &x: sm) {double kv = key(x); if(kv >= 0) v.push_back({kv, x.w});}
                if(v.empty()) return dflt;
                std::sort(v.begin(), v.end());
                double tot = 0; for(auto &p: v) tot += p.second;
                double acc = 0;
                for(auto &p: v) {acc += p.second; if(acc >= 0.5 * tot) return p.first;}
                return v.back().first;
            };
            kMed = wmedian([](const W &x) {return x.k;}, best.k);
            remMed[0] = wmedian([](const W &x) {return x.r0;}, remLo[0]);
            remMed[1] = wmedian([](const W &x) {return x.r1;}, remLo[1]);
        }
        //the median needs a play the data bound: with S_j reaching the largest
        //play the moves could hide, the flat prior is improper -- stay with the
        //least slack (and let the probe double through a dead band)
        if( !openEnd) for(int si = 0; si < 2; ++si) remLo[si] = remMed[si];
        A.best = best; A.lo = lo; A.bLo = lo.b; A.kHi = kHi;
        for(int si = 0; si < 2; ++si) {A.remLo[si] = remLo[si]; A.remHi[si] = remHi[si];}
        //the gain for the step (local, with k2), scaled from the best
        //hypothesis's to the posterior median over S_j: k_best while the gear
        //may have been loaded the leg's way from the start, the centre of the
        //dead band's plausible interval when it cannot
        double kb = best.k + (A.useK2 ? 2 * best.k2 * best.phiEnd : 0);
        if( !(kb > 0)) kb = best.k;
        double ratio = 1.0;
        if(best.k > 0) ratio = kMed / best.k;
        A.kStep = kb * ratio;
    }
};

} //namespace alternate

//===========================================================================
// Servo: the fitted C1/C2 act as a load-side encoder; the gear's play is
// handled one-sidedly (an uncertain play leaves an STM short, never past).
//  1. Probes, STM1 then STM2, no back legs: the column u_j of each STM (its
//     direction in relative (C1, C2)) and a lower-bound gain k_lo.
//  2. One clean verification leg (both STMs continue the way their gear is
//     loaded): the true gain, and the play when a probe started on the far
//     side.
//  3. Newton steps, both STMs per move, to the closed-form match of the
//     pooled model, decomposed on the columns.
//  4. Gated learning: gains only from engaged (clean) moves, seen at 5 sigma
//     of fit noise plus the columns' own uncertainty; the play only from
//     reversals, judged over the whole reversal.  A quadratic dial (from the
//     last two clean secants) follows a curved C(theta) and finds where it
//     turns back.
namespace servo {

struct V2 {
    double v[2] = {0, 0};
    double &operator[](int i) {return v[i];}
    double operator[](int i) const {return v[i];}
};
inline V2 mk(double a, double b) {V2 r; r[0] = a; r[1] = b; return r;}
inline V2 sub(const V2 &a, const V2 &b) {return mk(a[0] - b[0], a[1] - b[1]);}
inline V2 hyp(const V2 &a, const V2 &b) {return mk(hypot(a[0], b[0]), hypot(a[1], b[1]));}
inline double dot(const V2 &a, const V2 &b) {return a[0] * b[0] + a[1] * b[1];}
inline double nrm(const V2 &a) {return hypot(a[0], a[1]);}
inline int sgn(double x) {return x > 0 ? 1 : (x < 0 ? -1 : 0);}

template <class Rig>
struct Tuner {
    static constexpr double testFirst = 10;  //!< the first test rotation [deg]
    static constexpr double maxMove = 720;   //!< no single move beyond this [deg]
    static constexpr double seenSigma = 5;   //!< 'seen' = 5 sigma
    static constexpr int maxFits = 3;        //!< fits per stop, at most
    static constexpr int fitsProbe = 2, fitsNear = 2, fitsFar = 1; //!< per probe stop / step stop (|rl| < 0.3 or not)
    static constexpr int maxTurns = 2;       //!< crossings of a dial's turning point before giving up

    Rig &rig;
    double ft, w;
    double Cref[2] = {0, 0};
    //pooled probe constants (r1, r2, l1 do not depend on the motors)
    double sr1 = 0, sr2 = 0, sl1 = 0; int npool = 0;
    Circuit latest{};
    //a stop: the averaged fits at the current position (x relative to the first fit)
    struct Stop {V2 x, s; int n = 0; double sx[2] = {0, 0}, ss[2] = {0, 0};};
    Stop cur;
    //columns: unit direction u_j (+theta along +u_j) and its 1-sigma angle
    V2 u[2]; double sang[2] = {0, 0};
    double Bi[2][2] = {{1, 0}, {0, 1}};
    //per-axis gain
    double k[2] = {0, 0}, klo[2] = {0, 0};   //gain [x-units/deg] at the current position
    bool ver[2] = {false, false};
    double eps[2] = {-1, -1};          //|rho - 1| of the last clean move
    double r[2] = {0, 0};              //trust radius (until the gain is verified)
    //the play
    int eng[2] = {1, 1};               //the side the gear was last known loaded on
    double bEst[2] = {0, 0}, bVar[2] = {-1, -1};  //measured play (bVar < 0: none)
    double blo[2] = {0, 0};            //lower bound of the play
    //a reversal in progress: a move against the loaded side, until it is seen
    struct Rev {bool on = false, proper = false, clear = false; int dir = 0; double T = 0, A = 0, V = 0, needLo = 0;};
    Rev rv[2];
    struct RevRec {double travel, arr, sarr; bool proper;};
    std::vector<RevRec> pend[2];       //reversals seen before the gain was verified
    //the probe
    double L[2] = {0, 0}, dun[2] = {0, 0};
    V2 arrival[2], arrSig[2];
    double probeTravel = 0;
    //the dial: secant gains of the current engaged run, at their midpoints
    struct Secant {double s, mid, sig;};
    std::vector<Secant> sec[2];
    double kappaC[2] = {0, 0};         //curvature d(gain)/d(theta), 0 unless significant
    //oscillation guard
    int lastSign[2] = {0, 0}, lastSign2[2] = {0, 0};
    double lastP[2] = {0, 0}, lastP2[2] = {0, 0};
    double osc[2] = {1, 1};
    int negRho[2] = {0, 0}, turns[2] = {0, 0};

    explicit Tuner(Rig &r_) : rig(r_) {ft = rig.targetFreq(); w = 2 * M_PI * ft;}

    //---- measurement ---------------------------------------------------------
    void pool(const Circuit &c) {sr1 += c.r1; sr2 += c.r2; sl1 += c.l1; ++npool; latest = c;}
    Circuit pooled() const {Circuit c = latest; c.r1 = sr1 / npool; c.r2 = sr2 / npool; c.l1 = sl1 / npool; return c;}
    void addFit() {
        Circuit c = rig.measure();
        if(Cref[0] == 0) {Cref[0] = c.c1; Cref[1] = c.c2;}
        pool(c);
        cur.sx[0] += c.c1 / Cref[0] - 1; cur.sx[1] += c.c2 / Cref[1] - 1;
        cur.ss[0] += c.c1err / Cref[0]; cur.ss[1] += c.c2err / Cref[1];
        ++cur.n;
        for(int i = 0; i < 2; ++i) {cur.x[i] = cur.sx[i] / cur.n; cur.s[i] = cur.ss[i] / cur.n / sqrt((double)cur.n);}
    }
    void newStop(int nfits) {cur = Stop{}; for(int i = 0; i < nfits; ++i) addFit();}

    //---- model -----------------------------------------------------------------
    //! the match of the pooled model (the latest fit when that fails)
    bool target(V2 &xs) const {
        double c1, c2;
        if( !pooled().matchingCaps(ft, c1, c2) && !latest.matchingCaps(ft, c1, c2)) return false;
        xs = mk(c1 / Cref[0] - 1, c2 / Cref[1] - 1); return true;
    }
    double modelRL(const V2 &x) const {
        Circuit c = pooled(); c.c1 = Cref[0] * (1 + x[0]); c.c2 = Cref[1] * (1 + x[1]);
        if(c.c1 <= 0 || c.c2 <= 0) return 1.0;
        return std::abs(c.rl(w));
    }
    //! half-width along u_j of the model window |rl| < 0.1 through xs
    double halfWidth(const V2 &xs, int j) const {
        double res[2];
        for(int side = 0; side < 2; ++side) {
            double s = side ? -1 : 1, lo = 0, hi = 1e-7;
            auto at = [&](double t) {return modelRL(mk(xs[0] + s * t * u[j][0], xs[1] + s * t * u[j][1]));};
            while(at(hi) < 0.1 && hi < 4) {lo = hi; hi *= 2;}
            for(int it = 0; it < 40; ++it) {double m = 0.5 * (lo + hi); if(at(m) < 0.1) lo = m; else hi = m;}
            res[side] = 0.5 * (lo + hi);
        }
        return std::min(res[0], res[1]);
    }
    //! the t minimising the model's |rl| along x0 + t u_i
    double bestAlong(const V2 &x0, int i) const {
        auto at = [&](double t) {return modelRL(mk(x0[0] + t * u[i][0], x0[1] + t * u[i][1]));};
        double best = 0, bv = at(0);
        for(double span = 1.0; span > 1e-7; span *= 0.5)
            for(int sg: {-1, 1}) {double t = best + sg * span; double v = at(t); if(v < bv) {bv = v; best = t;}}
        return best;
    }

    //---- columns ---------------------------------------------------------------
    V2 perp(int j) const {return mk(-u[j][1], u[j][0]);}
    void setBasis() {
        double a = u[0][0], b = u[1][0], c = u[0][1], d = u[1][1], det = a * d - b * c;
        Bi[0][0] = d / det; Bi[0][1] = -b / det; Bi[1][0] = -c / det; Bi[1][1] = a / det;
    }
    V2 decomp(const V2 &dx) const {V2 a; for(int j = 0; j < 2; ++j) a[j] = Bi[j][0] * dx[0] + Bi[j][1] * dx[1]; return a;}
    //! 1-sigma of decomp(dx): the fit noise s of dx, plus what the column
    //! directions' uncertainty makes of the motion a along them
    V2 sigA(const V2 &s, const V2 &a) const {
        V2 res;
        for(int i = 0; i < 2; ++i) {
            double var = pow(Bi[i][0] * s[0], 2) + pow(Bi[i][1] * s[1], 2);
            for(int j = 0; j < 2; ++j) {V2 pj = perp(j); double bp = Bi[i][0] * pj[0] + Bi[i][1] * pj[1]; var += pow(a[j] * sang[j] * bp, 2);}
            res[i] = sqrt(var);
        }
        return res;
    }
    double perpNoise(int j, const V2 &sd) const {V2 pj = perp(j); return hypot(pj[0] * sd[0], pj[1] * sd[1]);}
    //! a move of axis j alone, arrival dx (noise sd), in the direction sg:
    //! fuses its direction into u_j (directions do not depend on the play)
    void fuseDir(int j, const V2 &dx, const V2 &sd, int sg) {
        double n = nrm(dx);
        V2 um = mk(sg * dx[0] / n, sg * dx[1] / n);
        if(dot(um, u[j]) < 0.5) return; //not this column: leave it
        double sm = perpNoise(j, sd) / n;
        double dpsi = atan2(u[j][0] * um[1] - u[j][1] * um[0], dot(u[j], um));
        double wgt = sang[j] * sang[j] / (sang[j] * sang[j] + sm * sm);
        double ps = atan2(u[j][1], u[j][0]) + wgt * dpsi;
        u[j] = mk(cos(ps), sin(ps));
        sang[j] = 1.0 / sqrt(1.0 / (sang[j] * sang[j]) + 1.0 / (sm * sm));
        setBasis();
    }

    //---- per-axis learning -----------------------------------------------------
    //! relative uncertainty of the gain over a move: |rho - 1| of the last
    //! clean move, or a default while there is none
    double gainEps(int j) const {return ver[j] ? (eps[j] < 0 ? 0.15 : eps[j]) : 0.3;}
    bool playKnown(int j) const {return bVar[j] >= 0;}
    //! a reversal of `travel` degrees arrived |a| = arr (1-sigma sarr).  When it
    //! started loaded on the other side (proper), the lost motion is the play;
    //! otherwise (a return from an unfinished reversal) only a part of it.
    //! The transmitted part, arr/k, carries the gain's uncertainty.
    void learnPlay(int j, double travel, double arr, double sarr, bool proper) {
        double tr = arr / k[j];
        double sl = hypot(sarr / k[j], tr * gainEps(j));
        double lost = travel - tr;
        if(proper) {
            //fuses with an earlier measurement that agrees; otherwise the latest wins
            if(bVar[j] < 0 || fabs(lost - bEst[j]) > 3 * sqrt(bVar[j] + sl * sl)) {bEst[j] = lost; bVar[j] = sl * sl;}
            else {double w1 = 1 / bVar[j], w2 = 1 / (sl * sl); bEst[j] = (bEst[j] * w1 + lost * w2) / (w1 + w2); bVar[j] = 1 / (w1 + w2);}
            blo[j] = std::max(0.0, bEst[j] - 2 * sqrt(bVar[j]));
        }
        else blo[j] = std::max(blo[j], lost - 2 * sl);
    }
    //! a clean (engaged) move of axis j by dth, ending at endPos, arrived a (1-sigma sa)
    void addSecant(int j, double dth, double a, double sa, double endPos) {
        double s = a / dth, mid = endPos - 0.5 * dth;
        sec[j].push_back({s, mid, sa / fabs(dth)});
        kappaC[j] = 0;
        size_t n = sec[j].size();
        if(n >= 2) {
            const Secant &p = sec[j][n - 2], &q = sec[j][n - 1];
            if(fabs(q.s - p.s) > 3 * hypot(q.sig, p.sig) && fabs(q.mid - p.mid) > 0) kappaC[j] = (q.s - p.s) / (q.mid - p.mid);
        }
        //the gain at the current position, never extrapolated through zero
        k[j] = s + kappaC[j] * (rig.position(j) - mid);
        if(sgn(k[j]) != sgn(s)) k[j] = s;
    }
    void resetRun(int j) {sec[j].clear(); kappaC[j] = 0;}
    //! gear travel to move by p along u_j on the dial model; *reach = false
    //! when the dial turns back before p (then the travel to the turning point)
    double gearNeed(int j, double p, bool *reach) const {
        *reach = true;
        double g = k[j], c = kappaC[j];
        if(c == 0) return p / g;
        double disc = g * g + 2 * c * p;
        if(disc < 0) {*reach = false; return -g / c;}
        return 2 * p / (g + sgn(g) * sqrt(disc));
    }
    double predictA(int j, double dth) const {return k[j] * dth + 0.5 * kappaC[j] * dth * dth;}
    //! begins a move against the loaded side.  A 'proper' reversal starts
    //! loaded (its lost motion is the play); a return from an unfinished
    //! reversal has at least min(T, play) to retrace.
    void startReversal(int j, int dir) {
        Rev &v = rv[j];
        double needLo = v.on ? std::min(v.T, blo[j]) : blo[j];
        bool proper = !v.on;
        v = Rev{}; v.on = true; v.proper = proper; v.dir = dir; v.needLo = needLo;
    }

    void doMove(double d0, double d1) {
        if(d0 != 0 && d1 != 0) rig.moveBoth(d0, d1);
        else if(d0 != 0) rig.move(0, d0);
        else if(d1 != 0) rig.move(1, d1);
    }

    //---- 1. probe of one axis, no back legs ------------------------------------
    void probe(int j) {
        Stop start = cur, prev = cur;
        double delta = testFirst * eng[j];
        dun[j] = 0;
        for(;;) {
            if(probeTravel + fabs(delta) > maxMove) throw GiveUp{};
            doMove(j == 0 ? delta : 0, j == 1 ? delta : 0);
            probeTravel += fabs(delta);
            newStop(fitsProbe);
            V2 dx = sub(cur.x, prev.x), sd = hyp(cur.s, prev.s);
            bool seen = std::max(fabs(dx[0]) / sd[0], fabs(dx[1]) / sd[1]) > seenSigma;
            if(seen) {
                double n = nrm(dx), sg = delta > 0 ? 1 : -1;
                u[j] = mk(sg * dx[0] / n, sg * dx[1] / n);
                sang[j] = perpNoise(j, sd) / n;
                L[j] = delta; klo[j] = n / fabs(delta); k[j] = klo[j];
                arrival[j] = sub(cur.x, start.x); arrSig[j] = hyp(cur.s, start.s);
                eng[j] = (int)sg;
                return;
            }
            dun[j] += fabs(delta);
            prev = cur;
            delta *= 2;
        }
    }

    //---- 2. verification leg ---------------------------------------------------
    void verify(const V2 &xs) {
        V2 p = decomp(sub(xs, cur.x)), sa = sigA(cur.s, mk(0, 0));
        double c[2];
        for(int j = 0; j < 2; ++j) {
            double n = p[j] / klo[j];   //an upper bound of the gear travel needed
            if(sgn(n) == eng[j]) c[j] = eng[j] * std::min(0.5 * fabs(n), fabs(L[j]));   //productive
            else c[j] = eng[j] * 0.5 * fabs(L[j]);                                     //informative
            if(klo[j] * fabs(c[j]) < seenSigma * sa[j]) c[j] = 0;
            r[j] = 4 * std::max(fabs(L[j]), fabs(c[j]));
        }
        Stop before = cur;
        doMove(c[0], c[1]);
        newStop(fitsProbe);
        V2 dx = sub(cur.x, before.x), sd = hyp(cur.s, before.s);
        for(int j = 0; j < 2; ++j) if(c[j] != 0 && c[1 - j] == 0 && nrm(dx) > 0) fuseDir(j, dx, sd, sgn(c[j]));
        V2 da = decomp(dx), sda = sigA(sd, da);
        for(int j = 0; j < 2; ++j) {
            if(c[j] == 0) continue;
            if(fabs(da[j]) > seenSigma * sda[j] && da[j] / c[j] > 0) {
                addSecant(j, c[j], da[j], sda[j], rig.position(j)); ver[j] = true;
                //lost motion of the probe: the play, when the probe started on the far side
                double sk = sda[j] / fabs(c[j]);
                V2 aa = decomp(arrival[j]), sarr = sigA(arrSig[j], aa);
                double arr = fabs(aa[j]);
                double e = (dun[j] + fabs(L[j])) - arr / k[j];
                double seN = hypot(sarr[j] / k[j], arr / k[j] * sk / k[j]);   //fit noise
                double se = hypot(seN, arr / k[j] * gainEps(j));               //and a gain that may differ there
                //unseen legs prove the far side; a first leg seen shows it only
                //beyond what a gain differing between the two legs could make of it
                if(dun[j] > 0 && e > 3 * seN) se = seN;
                if(e > 3 * se) {bEst[j] = e; bVar[j] = se * se; blo[j] = std::max(0.0, e - 2 * se);}
            }
        }
    }

    //---- 3./4. step loop with gated learning -------------------------------------
    void run() {
        newStop(fitsProbe);
        probe(0);
        probe(1);
        setBasis();
        V2 xs;
        if( !target(xs)) throw GiveUp{};
        verify(xs);
        int dud = 0;
        for(;;) {
            if( !target(xs)) throw GiveUp{};
            //(a) the need, along the columns, and in gear degrees on the dial
            V2 p = decomp(sub(xs, cur.x));
            double n[2], dth[2] = {0, 0}, tau[2], comp[2] = {0, 0};
            bool reach[2];
            for(int j = 0; j < 2; ++j) {n[j] = gearNeed(j, p[j], &reach[j]); tau[j] = 0.5 * halfWidth(xs, j);}
            //a dial that turns back short of the match: that STM to its turning
            //point, the other to the best match along that boundary
            for(int j = 0; j < 2; ++j) {
                if(reach[j]) continue;
                int i = 1 - j;
                double aj = predictA(j, n[j]);
                V2 xe = mk(cur.x[0] + aj * u[j][0], cur.x[1] + aj * u[j][1]);
                double t = bestAlong(xe, i);
                if(reach[i]) {p[i] = t; n[i] = gearNeed(i, t, &reach[i]);}
                p[j] = aj;
            }
            //(b) leave an axis alone when the other one alone would do
            bool leave[2];
            for(int j = 0; j < 2; ++j) {
                int i = 1 - j;
                leave[j] = modelRL(mk(cur.x[0] + p[i] * u[i][0], cur.x[1] + p[i] * u[i][1])) < 0.05;
            }
            if(leave[0] && leave[1]) {
                if(cur.n < maxFits) {addFit(); continue;}
                leave[0] = leave[1] = false;
            }
            for(int j = 0; j < 2; ++j) if(leave[j]) n[j] = 0;
            //the oscillation guard
            for(int j = 0; j < 2; ++j) {
                int sg = sgn(n[j]);
                if(sg != 0 && lastSign[j] != 0 && lastSign2[j] != 0 && sg != lastSign[j] && lastSign[j] != lastSign2[j] && fabs(p[j]) > 0.5 * lastP2[j])
                    osc[j] = std::max(0.125, osc[j] * 0.5);
                if(fabs(p[j]) < tau[j]) osc[j] = 1;
                if(sg != 0) {lastSign2[j] = lastSign[j]; lastP2[j] = lastP[j]; lastSign[j] = sg; lastP[j] = fabs(p[j]);}
            }
            //(c)-(f) the move of each axis
            bool revMove[2] = {false, false};
            auto plan = [&](int j, double need) {
                int sn = sgn(need);
                double e = gainEps(j);
                double kap = (e * fabs(p[j]) < tau[j]) ? 1.0 : 1.0 - std::min(0.3, 2 * e);
                if(playKnown(j) && ver[j]) kap = 1.0;
                if( !reach[j]) kap = 1.0;  //the turning point is an extremum: no overshoot to fear
                //a first reversal of an unknown play can only fall short by it
                if(sn != eng[j] && !rv[j].on && !playKnown(j)) kap = 1.0;
                kap *= osc[j];
                Rev &v = rv[j];
                if( !v.on && sn == eng[j]) comp[j] = 0;   //(d) clean
                else {                                    //(e) reversal, or its continuation
                    if( !v.on || sn != v.dir) startReversal(j, sn);
                    revMove[j] = true;
                    if(v.T == 0) {
                        comp[j] = v.needLo;     //the play's lower bound: short, never past ...
                        //... by more than the tolerance absorbs, when the play was measured
                        if(v.proper && playKnown(j))
                            comp[j] = std::max(comp[j], bEst[j] - std::max(0.0, 2 * sqrt(bVar[j]) - tau[j] / fabs(k[j])));
                    }
                    else if(v.proper && playKnown(j)) {
                        //a measured play, now seen to be at least needLo: bisects up to
                        //its upper bound (searches if the measurement proves wrong)
                        double hi = bEst[j] + 2 * sqrt(bVar[j]);
                        comp[j] = (hi > v.needLo) ? std::max(0.0, 0.5 * (v.needLo + hi) - v.T) : v.T;
                    }
                    //an unknown play known to have taken all of T: searched geometrically
                    else comp[j] = v.clear ? std::max(v.needLo - v.T, v.T) : std::max(0.0, v.needLo - v.T);
                }
                dth[j] = kap * need + sn * comp[j];
                double lim = std::min(maxMove, (ver[j] ? maxMove : r[j]) + comp[j]);
                if(fabs(dth[j]) > lim) dth[j] = sn * lim;
            };
            for(int j = 0; j < 2; ++j) if(n[j] != 0) plan(j, n[j]);
            if(dth[0] == 0 && dth[1] == 0) {
                if(cur.n < maxFits) {addFit(); continue;}
                int j = fabs(p[0]) / tau[0] > fabs(p[1]) / tau[1] ? 0 : 1;
                plan(j, p[j] / k[j]);
            }
            //(g)
            Stop before = cur;
            doMove(dth[0], dth[1]);
            newStop(modelRL(cur.x) < 0.3 ? fitsNear : fitsFar);
            V2 dx = sub(cur.x, before.x), sd = hyp(cur.s, before.s);
            //a single-axis move shows its column's direction
            for(int j = 0; j < 2; ++j)
                if(dth[j] != 0 && dth[1 - j] == 0 && std::max(fabs(dx[0]) / sd[0], fabs(dx[1]) / sd[1]) > seenSigma) fuseDir(j, dx, sd, sgn(dth[j]));
            V2 da = decomp(dx), sda = sigA(sd, da);
            //4. gated learning
            bool expectedUnseen = false, anySeen = false;
            for(int j = 0; j < 2; ++j) {
                if(dth[j] == 0) continue;
                bool seen = fabs(da[j]) > seenSigma * sda[j];
                int sd_ = sgn(dth[j]);
                if( !revMove[j]) {
                    //CLEAN: the gain (latest wins: local, it follows a curved dial)
                    anySeen |= seen;
                    if(seen) {
                        double rho = da[j] / predictA(j, dth[j]);
                        if(rho > 0) {
                            negRho[j] = 0;
                            addSecant(j, dth[j], da[j], sda[j], rig.position(j)); ver[j] = true; eps[j] = fabs(rho - 1);
                            if(rho >= 0.8 && rho <= 1.25) r[j] = std::min(maxMove, std::max(r[j], 2 * fabs(dth[j])));
                            if(rho < 0.5 || rho > 2) r[j] = std::max(0.5 * fabs(dth[j]), 5.0);
                            for(auto &rr: pend[j]) learnPlay(j, rr.travel, rr.arr, rr.sarr, rr.proper);
                            pend[j].clear();
                        }
                        else {
                            //the dial went backwards: it has turned (the dial model finds where)
                            addSecant(j, dth[j], da[j], sda[j], rig.position(j));
                            if(++negRho[j] >= 2) throw GiveUp{};
                            //crossing the turning point again and again: the match lies
                            //beyond the dial's range
                            if(++turns[j] > maxTurns) throw GiveUp{};
                        }
                    }
                    else if(fabs(predictA(j, dth[j])) > 2 * seenSigma * sda[j]) expectedUnseen = true;
                } else {
                    //REVERSAL: judged over its whole travel; never a gain from it
                    Rev &v = rv[j];
                    v.T += fabs(dth[j]); v.A += da[j] * sd_; v.V += sda[j] * sda[j];
                    if(seen) {
                        anySeen = true;
                        if(ver[j]) learnPlay(j, v.T, v.A, sqrt(v.V), v.proper);
                        else pend[j].push_back({v.T, v.A, sqrt(v.V), v.proper});
                        if(v.dir != eng[j]) resetRun(j);  //loaded on the other side: a new run of the dial
                        eng[j] = v.dir; v = Rev{};
                    } else {
                        //the play took at least T - (A + 2 sigma)/k of this reversal
                        double lb = v.T - (std::max(0.0, v.A) + 2 * sqrt(v.V)) / k[j];
                        v.needLo = std::max(v.needLo, lb);
                        if(v.proper) blo[j] = std::max(blo[j], lb);
                        v.clear = lb > 0.5 * v.T;
                        if(v.T > maxMove) throw GiveUp{};
                    }
                }
            }
            //gives up when moves that should show do not
            dud = ( !anySeen && expectedUnseen) ? dud + 1 : 0;
            if(dud >= 3) throw GiveUp{};
        }
    }
};

} //namespace servo
} //namespace detail

//! Adds \a fit to \a log and returns the step that follows it: the search
//! replayed over the whole log, up to its first request the log does not
//! answer.  A move is recorded in \a log before it is returned.
inline Step next(Search search, double ft, Log &log, const Circuit &fit) {
    Event e;
    e.isFit = true;
    e.fit = fit;
    log.push_back(e);
    detail::ReplayRig rig(ft, log);
    Step step;
    try {
        if(search == Search::Alternate)
            detail::alternate::Tuner<detail::ReplayRig>(rig).run();
        else
            detail::servo::Tuner<detail::ReplayRig>(rig).run();
        step.kind = Step::Kind::GiveUp; //never returns otherwise
    }
    catch(detail::ReplayRig::Frontier &f) {
        if(f.wantsFit) {
            step.kind = Step::Kind::Measure;
        }
        else {
            Event mv;
            mv.isFit = false;
            mv.d[0] = f.d[0]; mv.d[1] = f.d[1];
            log.push_back(mv);
            step.kind = Step::Kind::Move;
            step.d[0] = f.d[0]; step.d[1] = f.d[1];
        }
    }
    catch(GiveUp &) {
        step.kind = Step::Kind::GiveUp;
    }
    catch(detail::ReplayRig::Diverged &) {
        step.kind = Step::Kind::GiveUp; //cannot happen with a deterministic search
    }
    return step;
}

} //namespace lctuner

#endif /* LCTUNERSEARCH_H_ */
