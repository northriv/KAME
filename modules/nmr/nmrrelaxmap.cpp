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
#include "nmrrelaxmap.h"
#include "nmrrelaxfit.h"

#include "xitemnode.h"
#include "graph.h"
#include "xwavengraph.h"

TikhonovRegular::Method
tikhonovMethodOf(NMRRelaxMapMode mode) {
    switch(mode) {
    case NMRRelaxMapMode::NoiseAnalysis:
        return TikhonovRegular::Method::KnownError;
    case NMRRelaxMapMode::GCV:
        return TikhonovRegular::Method::MinGCV;
    case NMRRelaxMapMode::LCurve:
        return TikhonovRegular::Method::L_Curve;
    case NMRRelaxMapMode::Off:
    case NMRRelaxMapMode::AllNonNegative:
    default:
        return TikhonovRegular::Method::AllNonNegative;
    }
}
void
addRelaxMapModeItems(Transaction &tr, const shared_ptr<XComboNode> &combo) {
    tr[ *combo].add({"Off", "AllNonNegative", "Noise Analysis", "L Curve", "GCV"});
}
void
addTikhonovMatrixItems(Transaction &tr, const shared_ptr<XComboNode> &combo) {
    tr[ *combo].add({"Identity", "2nd Derivative Op."});
}

double
NMRRelaxMapData::timeOfBin(int bin) const {
    if((bin < 0) || (bin >= binCount()))
        return 0.0;
    const std::vector<double> &times(timesOfBin[bin]);
    if(times.empty())
        return 0.0;
    double sum = 0.0;
    for(double t: times)
        sum += t;
    return sum / times.size();
}
int
NMRRelaxMapData::strongestRow() const {
    int best = 0;
    double best_sum = -1.0;
    for(int i = 0; i < (int)y.rows(); ++i) {
        double sum = 0.0;
        for(int j = 0; j < (int)y.cols(); ++j)
            sum += fabs(y.coeff(i, j));
        if(sum > best_sum) {
            best_sum = sum;
            best = i;
        }
    }
    return best;
}
void
NMRRelaxMapData::resize(int xcount, int bincount) {
    timesOfBin.clear();
    timesOfBin.resize(std::max(0, bincount));
    xvalues.assign(std::max(0, xcount), 0.0);
    y.setZero(std::max(0, xcount), std::max(0, bincount));
    yimag.setZero(std::max(0, xcount), std::max(0, bincount));
    isigma.setZero(std::max(0, xcount), std::max(0, bincount));
    noiseSq = 0.0;
}
std::vector<double>
NMRRelaxMapData::makeTGrid(double tmin, double tmax, int count) {
    std::vector<double> grid;
    if((count < 2) || (tmin <= 0.0) || (tmax <= tmin))
        return grid;
    grid.resize(count);
    double step = log(tmax / tmin) / (count - 1);
    for(int i = 0; i < count; ++i)
        grid[i] = tmin * exp(step * i);
    return grid;
}

bool
NMRRelaxMapSolver::isCacheValid(const NMRRelaxMapData &data, const std::vector<double> &tgrid,
    const XRelaxFunc *relax_fn, TikhonovRegular::TikhonovMatrix mattype) const {
    if( !m_regularization)
        return false;
    if((m_relaxFn != relax_fn) || (m_matStype != mattype))
        return false;
    if(m_tgrid != tgrid)
        return false;
    return m_times == data.timesOfBin;
}
Eigen::MatrixXd
NMRRelaxMapSolver::exec(const NMRRelaxMapData &data, const std::vector<double> &tgrid,
    const shared_ptr<XRelaxFunc> &relax_fn, double relax_coeff,
    TikhonovRegular::TikhonovMatrix mattype, TikhonovRegular::Method method,
    int lambda_row) {
    Eigen::MatrixXd density;
    int nbin = data.binCount();
    int nx = data.xCount();
    int nt = (int)tgrid.size();
    //A kernel of fewer rows than unknowns is still solvable after
    //regularization, but one bin or one grid point is not a problem at all.
    m_status.clear();
    if( !relax_fn || (nbin < 2) || (nx < 1) || (nt < 2)) {
        m_status = "too few points to invert";
        return density;
    }
    if((data.y.rows() != nx) || (data.y.cols() != nbin))
        return density;
    if(data.y.cwiseAbs().maxCoeff() <= 0.0) {
        //nothing measured yet; the criteria would divide by zero.
        m_status = "no signal yet";
        return density;
    }

    if(m_invalidated.compare_set_strong(1, 0))
        m_regularization.reset();

    if( !isCacheValid(data, tgrid, relax_fn.get(), mattype)) {
        Eigen::MatrixXd mat_conv; //Matrix A; y = A x.
        mat_conv.setZero(nbin, nt);
        for(int j = 0; j < nt; ++j) {
            double it1 = 1.0 / tgrid[j];
            for(int i = 0; i < nbin; ++i) {
                const std::vector<double> &times(data.timesOfBin[i]);
                if(times.empty())
                    continue;
                //The mean over the abscissae summed into this bin.  The bin's
                //signal is the mean of its members' too, and the model is
                //linear in x, so this is exact rather than an approximation
                //around a representative time.
                double fsum = 0.0;
                for(double t: times) {
                    double f, df;
                    relax_fn->relax( &f, &df, t, it1);
                    fsum += f;
                }
                mat_conv.coeffRef(i, j) = relax_coeff * (fsum / times.size()) + 1.0;
            }
        }
        //very slow due to SVD.
        m_regularization = std::make_shared<TikhonovRegular>(mat_conv, mattype);
        m_times = data.timesOfBin;
        m_tgrid = tgrid;
        m_relaxFn = relax_fn.get();
        m_matStype = mattype;
    }

    int row = std::min(std::max(0, lambda_row), nx - 1);
    Eigen::VectorXd yrow = data.y.row(row).transpose();
    Eigen::VectorXd xrow = m_regularization->chooseLambda(method, yrow, data.noiseSq);
    //The parameter the whole map hangs on, and the one number of it that no
    //part of the picture shows.  With it, how much of the reference row it left
    //unexplained, against the noise there: about 1 is a fit, well above says
    //over-smoothed, well below says the noise is being fitted.
    m_status = formatString("lambda=%.3g", m_regularization->lambda());
    if(data.noiseSq > 0.0) {
        double rms = sqrt(m_regularization->residualSq(yrow, xrow) / nbin);
        m_status += formatString(" rms/sigma=%.2f", rms / sqrt(data.noiseSq));
    }

    density.setZero(nx, nt);
    for(int i = 0; i < nx; ++i) {
        yrow = data.y.row(i).transpose();
        density.row(i) = m_regularization->solve(yrow).transpose();
    }
    return density;
}

bool
setupRelaxCurvesGraph(Transaction &tr, const shared_ptr<XWaveNGraph> &graph,
    const char *xlabel, const char *ylabel) {
    const char *labels[] = {xlabel, ylabel, "Re [V]", "Im [V]", "Weight [1/V]"};
    tr[ *graph].setColCount(5, labels);
    if( !tr[ *graph].insertPlot(tr, i18n_noncontext("Relaxation"), 0, 2, -1, 4, 1)) return false;
    if( !tr[ *graph].insertPlot(tr, i18n_noncontext("Out-of-Phase"), 0, 3, -1, 4, 1)) return false;
    tr[ *tr[ *graph].axisx()->label()] = xlabel;
    tr[ *tr[ *graph].axisy()->label()] = i18n_noncontext("Intens [V]");
    tr[ *tr[ *graph].axisz()->label()] = ylabel;
    tr[ *tr[ *graph].axisz()->logScale()] = true;
    tr[ *tr[ *graph].plot(0)->drawLines()] = false;
    tr[ *tr[ *graph].plot(1)->drawLines()] = false;
    tr[ *tr[ *graph].plot(1)->intensity()] = 1.0;
    tr[ *graph].clearPoints();
    return true;
}
bool
setupRelaxDensityMapGraph(Transaction &tr, const shared_ptr<XWaveNGraph> &graph,
    const char *xlabel, const char *ylabel) {
    const char *labels[] = {xlabel, ylabel, "Density"};
    tr[ *graph].setColCount(3, labels);
    if( !tr[ *graph].insertPlot(tr, i18n_noncontext("Density"), 0, 1, -1, -1, 2)) return false;
    tr[ *tr[ *graph].axisx()->label()] = xlabel;
    tr[ *tr[ *graph].axisy()->label()] = ylabel;
    tr[ *tr[ *graph].axisy()->logScale()] = true;
    tr[ *tr[ *graph].plot(0)->drawLines()] = false;
    tr[ *tr[ *graph].plot(0)->intensity()] = 2;
    tr[ *tr[ *graph].plot(0)->colorPlot()] = true;
    tr[ *tr[ *graph].plot(0)->colorPlotColorHigh()] = QColor(0xFF, 0xFF, 0x2F).rgb();
    tr[ *tr[ *graph].plot(0)->colorPlotColorLow()] = QColor(0x00, 0x00, 0xFF).rgb();
    tr[ *tr[ *graph].plot(0)->pointColor()] = QColor(0x00, 0xFF, 0x00).rgb();
    tr[ *tr[ *graph].plot(0)->majorGridColor()] = QColor(0x4A, 0x4A, 0x4A).rgb();
    tr[ *graph->graph()->backGround()] = QColor(0,0,0).rgb();
    tr[ *graph->graph()->titleColor()] = clWhite;
    tr[ *tr[ *graph].axisx()->ticColor()] = clWhite;
    tr[ *tr[ *graph].axisx()->labelColor()] = clWhite;
    tr[ *tr[ *graph].axisx()->ticLabelColor()] = clWhite;
    tr[ *tr[ *graph].axisy()->ticColor()] = clWhite;
    tr[ *tr[ *graph].axisy()->labelColor()] = clWhite;
    tr[ *tr[ *graph].axisy()->ticLabelColor()] = clWhite;
    tr[ *tr[ *graph].axisz()->ticColor()] = clWhite;
    tr[ *tr[ *graph].axisz()->labelColor()] = clWhite;
    tr[ *tr[ *graph].axisz()->ticLabelColor()] = clWhite;
    tr[ *graph].clearPoints();
    return true;
}
void
drawRelaxCurves(const shared_ptr<XWaveNGraph> &graph,
    const NMRRelaxMapData &data, const char *tlabel) {
    int nx = data.xCount();
    int nbin = data.binCount();
    graph->iterate_commit([&](Transaction &tr){
        tr[ *graph].setLabel(1, tlabel);
        tr[ *tr[ *graph].axisz()->label()] = tlabel;
        size_t length = (size_t)nx * nbin;
        std::vector<float> colx(length, 0.0), colt(length, 0.0),
            colre(length, 0.0), colim(length, 0.0), colisigma(length, 0.0);
        int k = 0;
        for(int j = 0; j < nbin; ++j) {
            float t = data.timeOfBin(j);
            for(int i = 0; i < nx; ++i) {
                colx[k] = data.xvalues[i];
                colt[k] = t;
                colre[k] = data.y.coeff(i, j);
                colim[k] = data.yimag.coeff(i, j);
                colisigma[k] = data.isigma.coeff(i, j);
                ++k;
            }
        }
        tr[ *graph].setRowCount(length);
        tr[ *graph].setColumn(0, std::move(colx), 5);
        tr[ *graph].setColumn(1, std::move(colt), 5);
        tr[ *graph].setColumn(2, std::move(colre), 4);
        tr[ *graph].setColumn(3, std::move(colim), 4);
        tr[ *graph].setColumn(4, std::move(colisigma), 3);
        graph->drawGraph(tr);
    });
}
void
drawRelaxDensityMap(const shared_ptr<XWaveNGraph> &graph,
    const NMRRelaxMapData &data, const std::vector<double> &tgrid,
    const Eigen::MatrixXd &density, const char *tlabel, const XString &note) {
    int nx = data.xCount();
    int nt = (int)tgrid.size();
    bool drawable = (density.rows() == nx) && (density.cols() == nt);
    graph->iterate_commit([&](Transaction &tr){
        tr[ *graph->graph()->onScreenStrings()] = note;
        if( !drawable)
            return; //the note is the only thing there is to say.
        tr[ *graph].setLabel(1, tlabel);
        tr[ *tr[ *graph].axisy()->label()] = tlabel;
        size_t length = (size_t)nx * nt;
        std::vector<float> colx(length, 0.0), colt(length, 0.0), colval(length, 0.0);
        int k = 0;
        for(int i = 0; i < nx; ++i) {
            for(int j = 0; j < nt; ++j) {
                colx[k] = data.xvalues[i];
                colt[k] = tgrid[j];
                colval[k] = density.coeff(i, j);
                ++k;
            }
        }
        tr[ *graph].setRowCount(length);
        tr[ *graph].setColumn(0, std::move(colx), 5);
        tr[ *graph].setColumn(1, std::move(colt), 5);
        tr[ *graph].setColumn(2, std::move(colval), 4);
        graph->drawGraph(tr);
    });
}
