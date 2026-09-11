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
#ifndef nmrrelaxmapH
#define nmrrelaxmapH
//---------------------------------------------------------------------------
//! \file Relaxation-time mapping shared by the T1/T2 measurement and the
//! frequency-swept spectrometer.
//!
//! The two drivers acquire completely different things -- one distributes P1
//! over a range and Fourier-transforms one pulse at a time, the other sweeps a
//! frequency and sums every record into a swept axis -- but once the data is a
//! set of decay/recovery curves, one per point of an abscissa, the rest is the
//! same: build the kernel from a relaxation function, regularize, invert row by
//! row, draw.  That "rest" lives here, as plain classes: the two drivers sit in
//! unrelated node hierarchies (XNMRFSpectrum is already a XNMRSpectrumBase) and
//! XNode does not admit multiple inheritance, so they share by composition.
#include "xnode.h"
#include "tikhonovreg.h"

#include <Eigen/Core>

class XRelaxFunc;
class XWaveNGraph;
class XComboNode;

//! Criterion for choosing the regularization parameter, as offered in the UI.
//! The values are stored in .kam files; append, never renumber.
enum class NMRRelaxMapMode {Off = 0, AllNonNegative = 1, NoiseAnalysis = 2, LCurve = 3, GCV = 4};

//! \return the TikhonovRegular criterion \a mode selects. \a Off has none.
TikhonovRegular::Method tikhonovMethodOf(NMRRelaxMapMode mode);
//! Fills a combo with the NMRRelaxMapMode labels, in enum order.
void addRelaxMapModeItems(Transaction &tr, const shared_ptr<XComboNode> &);
//! Fills a combo with the TikhonovRegular::TikhonovMatrix labels, in enum order.
void addTikhonovMatrixItems(Transaction &tr, const shared_ptr<XComboNode> &);

//! Time-resolved data to be inverted into a distribution of relaxation times:
//! for every point of an abscissa (frequency, field, ...) one decay or recovery
//! curve, sampled at the abscissae carried by \a timesOfBin.
//!
//! A bin may hold MORE THAN ONE time (\a timesOfBin[b].size() > 1): CPMG echoes
//! are grouped m at a time (2 tau n/m) to trade time resolution for S/N and for
//! a smaller problem.  The kernel row of a grouped bin is then the MEAN of the
//! rows of its members, which is exactly what summing their signals measures --
//! there is no representative-time bias to correct for, and a bin holding a
//! single time reduces to the plain kernel.
struct NMRRelaxMapData {
    //! [ms] or [us], per bin; every abscissa summed into that bin.
    std::vector<std::vector<double> > timesOfBin;
    //! Abscissa of every row, in the unit the graphs are to show.
    std::vector<double> xvalues;
    Eigen::MatrixXd y; //!< (x, bin) in-phase signal [V]; what is inverted.
    Eigen::MatrixXd yimag; //!< (x, bin) out-of-phase signal [V]; shown, not inverted.
    Eigen::MatrixXd isigma; //!< (x, bin) 1/sigma [1/V]; shown, not inverted.
    double noiseSq = 0.0; //!< mean <dy^2> per point [V^2], for MinGCV/KnownError.

    int binCount() const {return (int)timesOfBin.size();}
    int xCount() const {return (int)xvalues.size();}
    //! \return the mean of the abscissae summed into \a bin, for display.
    double timeOfBin(int bin) const;
    //! \return the row carrying the most signal.  The lambda criteria are
    //! evaluated on one row only, and the middle of a swept axis is as likely
    //! to be a valley as a peak.
    int strongestRow() const;
    //! Sizes every matrix and zeroes it.
    void resize(int xcount, int bincount);
    //! \return a log-spaced grid of relaxation times, the unknowns of the
    //! inversion; empty if the range is degenerate.
    static std::vector<double> makeTGrid(double tmin, double tmax, int count);
};

//! Inverts NMRRelaxMapData row by row into a distribution of relaxation times,
//! by Tikhonov regularization (\sa TikhonovRegular).
//!
//! The kernel and its SVD are cached and rebuilt only when the problem itself
//! changes -- the bin times, the T grid, the relaxation function or the
//! regularization matrix.  \a relax_coeff deliberately does NOT invalidate the
//! cache: on a recovery curve it follows the running fit and would demand an
//! SVD per record.
//!
//! exec() is heavy and calls XRelaxFunc::relax(); call it from visualize(),
//! never from an iterate_commit() closure.
class NMRRelaxMapSolver {
public:
    NMRRelaxMapSolver() = default;
    //! \param tgrid relaxation times, in the unit of NMRRelaxMapData::timesOfBin.
    //! \param relax_coeff f(t) enters the kernel as relax_coeff * f(t) + 1;
    //!   -1 for a decay, 1/(c + a) for a recovery fitted to c * f + a.
    //! \param lambda_row the row on which \a method is evaluated.
    //! \return the (x, T) density, empty if the problem is degenerate.
    Eigen::MatrixXd exec(const NMRRelaxMapData &data, const std::vector<double> &tgrid,
        const shared_ptr<XRelaxFunc> &relax_fn, double relax_coeff,
        TikhonovRegular::TikhonovMatrix mattype, TikhonovRegular::Method method,
        int lambda_row);
    //! Drops the cached kernel.  Safe to call from analyze(): the request is a
    //! flag, consumed by the next exec(), and never touches the kernel itself.
    void invalidate() {m_invalidated = 1;}
    //! One line on what the last exec() settled on, and on what it was asked:
    //! enough of the inversion to repeat it from a picture of the map -- the
    //! criterion, the regularization matrix and parameter, the relaxation
    //! function, the grid -- since none of it shows in the picture itself.
    //! \sa drawRelaxDensityMap()
    const XString &status() const {return m_status;}
private:
    bool isCacheValid(const NMRRelaxMapData &, const std::vector<double> &tgrid,
        const XRelaxFunc *, TikhonovRegular::TikhonovMatrix) const;

    shared_ptr<TikhonovRegular> m_regularization;
    std::vector<std::vector<double> > m_times;
    std::vector<double> m_tgrid;
    const XRelaxFunc *m_relaxFn = nullptr;
    TikhonovRegular::TikhonovMatrix m_matStype = TikhonovRegular::TikhonovMatrix::I;
    atomic<int> m_invalidated{0};
    XString m_status;
};

//! Sets \a graph up as the color map of the inverted density.
//! \return false if a plot could not be inserted, which obliges the caller to
//! abandon the transaction, as XWaveNGraph::insertPlot() does elsewhere.
bool setupRelaxDensityMapGraph(Transaction &tr, const shared_ptr<XWaveNGraph> &graph,
    const char *xlabel, const char *ylabel);
//! Sets \a graph up as the raw curves, one per abscissa, colored by time.
bool setupRelaxCurvesGraph(Transaction &tr, const shared_ptr<XWaveNGraph> &graph,
    const char *xlabel, const char *ylabel);
//! Draws the raw curves of \a data. \a tlabel names the time axis, e.g. "2tau [us]".
//! \a note goes on the graph itself: what it took to ACQUIRE these curves, the
//! inversion's own settings being on the density map instead.  Neither graph
//! holds much text before it runs off its edges, so keep both short.
void drawRelaxCurves(const shared_ptr<XWaveNGraph> &graph,
    const NMRRelaxMapData &data, const char *tlabel, const XString &note);
//! Draws the density from NMRRelaxMapSolver::exec(). \a tlabel e.g. "T2 [us]".
//! \a note goes on the graph itself (NMRRelaxMapSolver::status()), and is put
//! there even when \a density is empty -- that is when it has something to say.
void drawRelaxDensityMap(const shared_ptr<XWaveNGraph> &graph,
    const NMRRelaxMapData &data, const std::vector<double> &tgrid,
    const Eigen::MatrixXd &density, const char *tlabel, const XString &note);

//---------------------------------------------------------------------------
#endif
