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
//---------------------------------------------------------------------------
#include "analyzer.h"
#include "xnodeconnector.h"
#include "montecarlo.h"
#include "kamemontecarlo.h"
#include "ui_montecarloform.h"
#include <QStatusBar>
#include "graph.h"
#include "graphwidget.h"

REGISTER_TYPE(XDriverList, MonteCarloDriver, "Monte-Carlo simulation");

struct FFTAxis {
    const char *desc;
    int h,k,l;
    int h0,k0,l0;
};
static const FFTAxis c_fftaxes[] = {
    {"h 0 0", 1, 0, 0, 0, 0, 0},
    {"0 k 0", 0, 1, 0, 0, 0, 0},
    {"0 0 l", 0, 0, 1, 0, 0, 0},
    {"h h 0", 1, 1, 0, 0, 0, 0},
    {"h h h", 1, 1, 1, 0, 0, 0},
    {0, 0,0,0, 0,0,0}
};
#define GRAPH_3D_FFT_INTENS_ABS "FFT3D-abs."
#define GRAPH_3D_FFT_INTENS_X "FFT3D-x"
#define GRAPH_3D_FFT_INTENS_Y "FFT3D-y"
#define GRAPH_3D_FFT_INTENS_Z "FFT3D-z"
#define GRAPH_3D_FFT_INTENS_M "FFT3D-along-H"
#define GRAPH_REAL_M "SPINS-along-H"
#define GRAPH_REAL_H "H-along-Ising"
#define GRAPH_REAL_P "Flipping Probability"
#define GRAPH_REAL_H_B_SITE "H-at-B-site"
#define GRAPH_REAL_H_8a_SITE "H-at-8a-site"
#define GRAPH_REAL_H_48f_SITE "H-at-48f-site"
#define GRAPH_FLIPS "Flip History"

XMonteCarloDriver::XMonteCarloDriver(const char *name, bool runtime, 
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XDummyDriver<XPrimaryDriver>(name, runtime, ref(tr_meas), meas),
    m_targetTemp(create<XDoubleNode>("TargetTemp", false)),
    m_targetField(create<XDoubleNode>("TargetField", false)),
    m_hdirx(create<XDoubleNode>("FieldDirX", false)),
    m_hdiry(create<XDoubleNode>("FieldDirY", false)),
    m_hdirz(create<XDoubleNode>("FieldDirZ", false)),
    m_L(create<XUIntNode>("Length", false)),
    m_cutoffReal(create<XDoubleNode>("CutoffReal", false)),
    m_cutoffRec(create<XDoubleNode>("CutoffRec", false)),
    m_alpha(create<XDoubleNode>("Alpha", false)),
    m_minTests(create<XDoubleNode>("MinTests", false)),
    m_minFlips(create<XDoubleNode>("MinFlips", false)),
    m_step(create<XTouchableNode>("Step", true)),
    m_graph3D(create<XComboNode>("Graph3D", false, true)),
	m_entryT(create<XScalarEntry>("Temp", false,
		dynamic_pointer_cast<XDriver>(shared_from_this()), "%.6g")),
	m_entryH(create<XScalarEntry>("Field", false,
		dynamic_pointer_cast<XDriver>(shared_from_this()), "%.6g")),
	m_entryU(create<XScalarEntry>("U", false,
		dynamic_pointer_cast<XDriver>(shared_from_this()), "%.6g")),
	m_entryC(create<XScalarEntry>("C", false,
		dynamic_pointer_cast<XDriver>(shared_from_this()), "%.4g")),
	m_entryCoT(create<XScalarEntry>("CoverT", false,
		dynamic_pointer_cast<XDriver>(shared_from_this()), "%.4g")),
	m_entryS(create<XScalarEntry>("S", false,
		dynamic_pointer_cast<XDriver>(shared_from_this()), "%.6g")),
	m_entryM(create<XScalarEntry>("M", false,
		dynamic_pointer_cast<XDriver>(shared_from_this()), "%.6g")),
	m_entry2in2(create<XScalarEntry>("2in2", false,
		dynamic_pointer_cast<XDriver>(shared_from_this()), "%.4g")),
	m_entry1in3(create<XScalarEntry>("1in3", false,
		dynamic_pointer_cast<XDriver>(shared_from_this()), "%.4g")),
    m_form(new FrmMonteCarlo),
    m_wave3D(create<XWaveNGraph>("Wave3D", false, 
        m_form->m_graph, m_form->m_edDump, m_form->m_tbDump, m_form->m_btnDump)),
    m_statusPrinter(XStatusPrinter::create(m_form.get())) {

	meas->scalarEntries()->insert(tr_meas, m_entryT);
	meas->scalarEntries()->insert(tr_meas, m_entryH);
	meas->scalarEntries()->insert(tr_meas, m_entryU);
	meas->scalarEntries()->insert(tr_meas, m_entryC);
	meas->scalarEntries()->insert(tr_meas, m_entryCoT);
	meas->scalarEntries()->insert(tr_meas, m_entryS);
	meas->scalarEntries()->insert(tr_meas, m_entryM);
	meas->scalarEntries()->insert(tr_meas, m_entry2in2);
	meas->scalarEntries()->insert(tr_meas, m_entry1in3);

	iterate_commit([=](Transaction &tr){
		tr[ *m_targetTemp] = 100.0;
		tr[ *m_hdirx] = 1.0;
		tr[ *m_hdiry] = 1.0;
		tr[ *m_hdirz] = 1.0;
		tr[ *m_L] = 8;
		tr[ *m_cutoffReal] = 4.0;
		tr[ *m_cutoffRec] = 2.0;
		tr[ *m_alpha] = 0.5;
		tr[ *m_minTests] = 1.0;
		tr[ *m_minFlips] = 0.2;

	//  for(const FFTAxis *axis = c_fftaxes; axis->desc; axis++) {
	//    m_fftAxisX].add(axis->desc);
	//    m_fftAxisY].add(axis->desc);
	//  }
	//  m_fftAxisX] = 0);
	//  m_fftAxisY] = 1);

		tr[ *m_graph3D].add(GRAPH_3D_FFT_INTENS_ABS);
		tr[ *m_graph3D].add(GRAPH_3D_FFT_INTENS_X);
		tr[ *m_graph3D].add(GRAPH_3D_FFT_INTENS_Y);
		tr[ *m_graph3D].add(GRAPH_3D_FFT_INTENS_Z);
		tr[ *m_graph3D].add(GRAPH_3D_FFT_INTENS_M);
		tr[ *m_graph3D].add(GRAPH_REAL_M);
		tr[ *m_graph3D].add(GRAPH_REAL_H);
		tr[ *m_graph3D].add(GRAPH_REAL_P);
		tr[ *m_graph3D].add(GRAPH_REAL_H_B_SITE);
		tr[ *m_graph3D].add(GRAPH_REAL_H_8a_SITE);
		tr[ *m_graph3D].add(GRAPH_REAL_H_48f_SITE);
		tr[ *m_graph3D].add(GRAPH_FLIPS);
    });
  
	m_conLength = xqcon_create<XQLineEditConnector>(m_L, m_form->m_edLength);
	m_conCutoffReal = xqcon_create<XQLineEditConnector>(m_cutoffReal, m_form->m_edCutoffReal);
	m_conCutoffRec = xqcon_create<XQLineEditConnector>(m_cutoffRec, m_form->m_edCutoffRec);
	m_conAlpha = xqcon_create<XQLineEditConnector>(m_alpha, m_form->m_edAlpha);
	m_conTargetTemp = xqcon_create<XQLineEditConnector>(m_targetTemp, m_form->m_edTargetTemp);
	m_conTargetField = xqcon_create<XQLineEditConnector>(m_targetField, m_form->m_edTargetField);
	m_conHDirX = xqcon_create<XQLineEditConnector>(m_hdirx, m_form->m_edHDirX);
	m_conHDirY = xqcon_create<XQLineEditConnector>(m_hdiry, m_form->m_edHDirY);
	m_conHDirZ = xqcon_create<XQLineEditConnector>(m_hdirz, m_form->m_edHDirZ);
	m_conMinTests = xqcon_create<XQLineEditConnector>(m_minTests, m_form->m_edMinTests);
	m_conMinFlips = xqcon_create<XQLineEditConnector>(m_minFlips, m_form->m_edMinFlips);
	m_conStep = xqcon_create<XQButtonConnector>(m_step, m_form->m_btnStep);
	m_conGraph3D = xqcon_create<XQComboBoxConnector>(m_graph3D, m_form->m_cmbGraph3D, Snapshot( *m_graph3D));

    m_L->setUIEnabled(true);
    m_cutoffReal->setUIEnabled(true);
    m_hdirx->setUIEnabled(true);
    m_hdiry->setUIEnabled(true);
    m_hdirz->setUIEnabled(true);
    m_minTests->setUIEnabled(true);
    m_minFlips->setUIEnabled(true);
    m_step->setUIEnabled(true);
    m_targetTemp->setUIEnabled(true);
    m_targetField->setUIEnabled(true);

    m_wave3D->iterate_commit([=](Transaction &tr){
		const char *s_trace_names[] = {
			"h or x", "k or y", "l or z", "intens.", "hx", "hy", "hz", "site"
		};
		tr[ *m_wave3D].setColCount(8, s_trace_names);
        if( !tr[ *m_wave3D].insertPlot(tr, "Intens.", 0, 1, -1, 3, 2))
            return;
		tr[ *tr[ *m_wave3D].plot(0)->drawLines()] = false;

		tr[ *m_wave3D->graph()->backGround()] = QColor(0,0,0).rgb();
		tr[ *tr[ *m_wave3D].plot(0)->intensity()] = 2;
		tr[ *tr[ *m_wave3D].plot(0)->colorPlot()] = true;
		tr[ *tr[ *m_wave3D].plot(0)->colorPlotColorHigh()] = QColor(0xFF, 0xFF, 0x2F).rgb();
		tr[ *tr[ *m_wave3D].plot(0)->colorPlotColorLow()] = QColor(0x00, 0x00, 0xFF).rgb();
		tr[ *tr[ *m_wave3D].plot(0)->pointColor()] = QColor(0x00, 0xFF, 0x00).rgb();
		tr[ *tr[ *m_wave3D].plot(0)->majorGridColor()] = QColor(0x4A, 0x4A, 0x4A).rgb();
		tr[ *m_wave3D->graph()->titleColor()] = clWhite;
		tr[ *tr[ *m_wave3D].axisx()->ticColor()] = clWhite;
		tr[ *tr[ *m_wave3D].axisx()->labelColor()] = clWhite;
		tr[ *tr[ *m_wave3D].axisx()->ticLabelColor()] = clWhite;
		tr[ *tr[ *m_wave3D].axisy()->ticColor()] = clWhite;
		tr[ *tr[ *m_wave3D].axisy()->labelColor()] = clWhite;
		tr[ *tr[ *m_wave3D].axisy()->ticLabelColor()] = clWhite;
		tr[ *tr[ *m_wave3D].axisz()->ticColor()] = clWhite;
		tr[ *tr[ *m_wave3D].axisz()->labelColor()] = clWhite;
		tr[ *tr[ *m_wave3D].axisz()->ticLabelColor()] = clWhite;
		tr[ *m_wave3D].clearPoints();
    });
}
XMonteCarloDriver::~XMonteCarloDriver() {
	Snapshot shot( *this);
    for(int d = 0; d < 3; d++) {
        if(shot[ *this].m_fftlen > 0) {
            fftw_destroy_plan(shot[ *this].m_fftplan[d]);
            fftw_free(shot[ *this].m_pFFTin[d]);
            fftw_free(shot[ *this].m_pFFTout[d]);
        }
    }	
}
void
XMonteCarloDriver::showForms() {
//! impliment form->show() here
    m_form->showNormal();
    m_form->raise();
}
void
XMonteCarloDriver::start() {
	Snapshot shot( *this);
    MonteCarlo::setupField(shot[ *m_L], 0.0, shot[ *m_cutoffReal], shot[ *m_cutoffRec], shot[ *m_alpha]);
    m_L->setUIEnabled(false);
    m_cutoffReal->setUIEnabled(false);
    m_cutoffRec->setUIEnabled(false);
    m_alpha->setUIEnabled(false);
    m_hdirx->setUIEnabled(false);
    m_hdiry->setUIEnabled(false);
    m_hdirz->setUIEnabled(false);
    iterate_commit([=](Transaction &tr){
    	const Snapshot &shot(tr);
        m_loop.reset(new MonteCarlo(8));
        tr[ *this].m_sumDU = m_loop->internalEnergy() * N_A;
        tr[ *this].m_sumDUav = tr[ *this].m_sumDU;
        tr[ *this].m_sumDS = 0.0;
        tr[ *this].m_testsTotal = 0.0;
        tr[ *this].m_flippedTotal = 0.0;
        tr[ *this].m_lastTemp = 300.0;
        tr[ *this].m_lastField = 0.0;
        tr[ *this].m_dU = 0.0;
        MonteCarlo::Vector3<double> field_dir(tr[ *m_hdirx], tr[ *m_hdiry], tr[ *m_hdirz]);
        field_dir.normalize();
        tr[ *this].m_lastMagnetization = m_loop->magnetization().innerProduct(field_dir);

        m_lsnStepTouched = tr[ *m_step].onTouch().connectWeakly(
    		shared_from_this(), &XMonteCarloDriver::onStepTouched);
        m_lsnTargetChanged = tr[ *m_targetTemp].onValueChanged().connectWeakly(
    		shared_from_this(), &XMonteCarloDriver::onTargetChanged);
        tr[ *m_targetField].onValueChanged().connect(m_lsnTargetChanged);
        m_lsnGraphChanged = tr[ *m_graph3D].onValueChanged().connectWeakly(
    		shared_from_this(), &XMonteCarloDriver::onGraphChanged);

        int fftlen = MonteCarlo::length() * 4;
        for(int d = 0; d < 3; d++) {
            if(shot[ *this].m_fftlen > 0) {
                fftw_destroy_plan(shot[ *this].m_fftplan[d]);
                fftw_free(shot[ *this].m_pFFTin[d]);
                fftw_free(shot[ *this].m_pFFTout[d]);
            }
            tr[ *this].m_pFFTin[d] = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftlen * fftlen * fftlen);
            tr[ *this].m_pFFTout[d] = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftlen * fftlen * fftlen);
            tr[ *this].m_fftplan[d] = fftw_plan_dft_3d(fftlen, fftlen, fftlen,
            	shot[ *this].m_pFFTin[d], shot[ *this].m_pFFTout[d],
            	FFTW_FORWARD, FFTW_ESTIMATE);
        }
        tr[ *this].m_fftlen = fftlen;
    });
}
void
XMonteCarloDriver::stop() {
    m_lsnTargetChanged.reset();
    m_lsnStepTouched.reset();
    m_lsnGraphChanged.reset();
    MonteCarlo::s_bAborting = true;
    m_L->setUIEnabled(true);
    m_cutoffReal->setUIEnabled(true);
    m_cutoffRec->setUIEnabled(true);
    m_alpha->setUIEnabled(true);
    m_hdirx->setUIEnabled(true);
    m_hdiry->setUIEnabled(true);
    m_hdirz->setUIEnabled(true);
    
    closeInterface();
}
void
XMonteCarloDriver::analyzeRaw(RawDataReader &reader, Transaction &tr) {
	const Snapshot &shot(tr);
    unsigned int size = MonteCarlo::length();
    unsigned int len = size*size*size*16;

    MonteCarlo::Vector3<double> field_dir(shot[ *m_hdirx], shot[ *m_hdiry], shot[ *m_hdirz]);
    field_dir.normalize();
    MonteCarlo::Vector3<double> field(field_dir);
    field *= shot[ *m_targetField];

    double dU = reader.pop<double>();
    tr[ *this].m_DUav = reader.pop<double>();
    tr[ *this].m_dU = dU;
    tr[ *this].m_flippedTotal += reader.pop<int>();
    tr[ *this].m_testsTotal += reader.pop<double>();
    double mav = reader.pop<double>();
    tr[ *this].m_Mav = mav;
    double m = reader.pop<double>();
    double onethree = reader.pop<double>();
    double twotwo = reader.pop<double>();
    double u_calc = reader.pop<double>();

    fprintf(stderr, "Total flips = %g (%g per spin).\n",
        ((double)tr[ *this].m_flippedTotal), ((double)tr[ *this].m_flippedTotal / len));
    fprintf(stderr, "Total tests = %g (%g per spin).\n",
        ((double)tr[ *this].m_testsTotal), ((double)tr[ *this].m_testsTotal / len));

    double dT = shot[ *m_targetTemp] - shot[ *this].m_lastTemp;
    tr[ *this].m_lastTemp = shot[ *m_targetTemp];
    double dM = m - shot[ *this].m_lastMagnetization;
    double dH = shot[ *m_targetField] - shot[ *this].m_lastField;
    // DS += (DU + HDM + MDH)/T
    tr[ *this].m_sumDS += (dU + dM * shot[ *m_targetField] * MU_B * N_A) / shot[ *m_targetTemp];
    // DU += -MDH.
    dU += -shot[ *this].m_lastMagnetization * MU_B * N_A * dH;
    double dUav = shot[ *this].m_DUav * N_A;
    dUav += -shot[ *this].m_Mav * MU_B * N_A * dH;
    double c = (fabs(dT) < 1e-10) ? 0.0 : ((shot[ *this].m_sumDU + dUav - shot[ *this].m_sumDUav) / dT);
    tr[ *this].m_sumDUav = shot[ *this].m_sumDU + dUav;
    tr[ *this].m_sumDU += dU;
    tr[ *this].m_lastMagnetization = m;
    tr[ *this].m_lastField = shot[ *m_targetField];

    long double u = shot[ *this].m_sumDU;
    if(u_calc != 0.0)  {
        double diff = (u - u_calc)/u_calc;
        if(fabs(diff) > 1e-5) {
            gErrPrint(formatString("SumDU Error = %g!", diff));
            tr[ *this].m_sumDU = u_calc;
        }
    }

    const char *spins = &*reader.popIterator();
    tr[ *this].m_spins.resize(len);
    std::memcpy( &tr[ *this].m_spins[0], spins, len);

    m_entryT->value(tr, shot[ *this].m_lastTemp);
    m_entryH->value(tr, shot[ *this].m_lastField);
    m_entryU->value(tr, u);
    m_entryC->value(tr, c);
    m_entryCoT->value(tr, c / shot[ *this].m_lastTemp);
    m_entryS->value(tr, shot[ *this].m_sumDS);
    m_entryM->value(tr, shot[ *this].m_Mav);
    m_entry2in2->value(tr, 100.0*twotwo);
    m_entry1in3->value(tr, 100.0*onethree);
}
void
XMonteCarloDriver::visualize(const Snapshot &shot) {
	if( !shot[ *this].time()) {
		iterate_commit([=](Transaction &tr){
			tr[ *m_wave3D].clearPoints();
        });
		return;
	}

	bool fftx = false;
	bool ffty = false;
	bool fftz = false;
	bool fft_intens_abs = false;
	bool along_field_dir = false;
	bool calch = false;
	bool calcp = false;
	bool calcasite = false;
	bool calcbsite = false;
	bool calc8asite = false;
	bool calc48fsite = false;
	bool writeflips = false;
	if(shot[ *m_graph3D].to_str() == GRAPH_3D_FFT_INTENS_ABS) {
		fftx = true;
		ffty = true;
		fftz = true;
		fft_intens_abs = true;
	}
	if(shot[ *m_graph3D].to_str() == GRAPH_3D_FFT_INTENS_X) {
		fftx = true;
	}
	if(shot[ *m_graph3D].to_str() == GRAPH_3D_FFT_INTENS_Y) {
		ffty = true;
	}
	if(shot[ *m_graph3D].to_str() == GRAPH_3D_FFT_INTENS_Z) {
		fftz = true;
	}
	if(shot[ *m_graph3D].to_str() == GRAPH_3D_FFT_INTENS_M) {
		fftx = true;
		ffty = true;
		fftz = true;
		along_field_dir = true;
	}
	if(shot[ *m_graph3D].to_str() == GRAPH_REAL_M) {
		calcasite = true;
		along_field_dir = true;
	}
	if(shot[ *m_graph3D].to_str() == GRAPH_REAL_H) {
		calcasite = true;
		calch = true;
	}
	if(shot[ *m_graph3D].to_str() == GRAPH_REAL_P) {
		calcasite = true;
		calch = true;
		calcp = true;
	}
	if(shot[ *m_graph3D].to_str() == GRAPH_REAL_H_B_SITE) {
		calcbsite = true;
	}
	if(shot[ *m_graph3D].to_str() == GRAPH_REAL_H_8a_SITE) {
		calc8asite = true;
	}
	if(shot[ *m_graph3D].to_str() == GRAPH_REAL_H_48f_SITE) {
		calc48fsite = true;
	}
	if(shot[ *m_graph3D].to_str() == GRAPH_FLIPS) {
		writeflips = true;
	}

    int size = MonteCarlo::length();
    MonteCarlo::Vector3<double> field_dir(shot[ *m_hdirx], shot[ *m_hdiry], shot[ *m_hdirz]);
    field_dir.normalize();
    MonteCarlo::Vector3<double> field(field_dir);
    field *= shot[ *m_targetField];
    MonteCarlo stored(0, false);
    stored.read( &shot[ *this].m_spins[0], shot[ *m_targetTemp], field);

    std::vector<char> spins(size*size*size*16);

    std::vector<double> fields;
    if(calch) fields.resize(size*size*size*16);
    std::vector<double> probabilities;
    if(calcp) probabilities.resize(size*size*size*16);
    stored.write((char*)&spins[0]
                   , fields.size() ? &fields[0] : 0, probabilities.size() ? &probabilities[0] : 0);
    int fftlen = shot[ *this].m_fftlen;
    for(int d = 0; d < 3; d++) {
        for(int site = 0; site < 16; site++) {
            const int *pos = cg_ASitePositions[site];
            int ising = cg_ASiteIsingAxes[site][d];
            for(int k = 0; k < size; k++) {
				for(int j = 0; j < size; j++) {
					for(int i = 0; i < size; i++) {
						int fftidx =fftlen*(fftlen*(4*k+pos[2]) + 4*j+pos[1]) + 4*i + pos[0];
						int spin = spins[size*(size*(size*site + k) + j) + i];
						shot[ *this].m_pFFTin[d][fftidx][0] = spin* ising / sqrt(3.0);
					}
				}
            }
        }
    }
        
    std::vector<double> colh(fftlen), colk(fftlen), coll(fftlen),
        colv(fftlen), cols(fftlen),
        colx(fftlen), coly(fftlen), colz(fftlen);

	if(fftx || ffty || fftz) {
		fftw_execute(shot[ *this].m_fftplan[0]);
		fftw_execute(shot[ *this].m_fftplan[1]);
		fftw_execute(shot[ *this].m_fftplan[2]);
    
        m_wave3D->iterate_commit([&](Transaction &tr){
			tr[ *m_wave3D].setRowCount(fftlen * fftlen * fftlen );
            unsigned int len = tr[ *m_wave3D].rowCount();
            std::vector<double> colh(len), colk(len), coll(len),
                colv(len), cols(len),
                colx(len), coly(len), colz(len);
            double normalize = A_MOMENT / len;
			int idx = 0;
			for(int l = 0; l < fftlen; l++) {
				for(int k = 0; k < fftlen; k++) {
					for(int h = 0; h < fftlen; h++) {
						int qidx = fftlen*(fftlen*l + k) + h;
						fftw_complex *ix = &shot[ *this].m_pFFTout[0][qidx];
						fftw_complex *iy = &shot[ *this].m_pFFTout[1][qidx];
						fftw_complex *iz = &shot[ *this].m_pFFTout[2][qidx];
                        colh[idx] = (double)h * 8.0/fftlen;
                        colk[idx] = (double)k * 8.0/fftlen;
                        coll[idx] = (double)l * 8.0/fftlen;
						double v = 0.0;
						if(along_field_dir) {
							double vr = field_dir.innerProduct(MonteCarlo::Vector3<double>
															   ((*ix)[0], (*iy)[0], (*iz)[0]));
							double vi = field_dir.innerProduct(MonteCarlo::Vector3<double>
															   ((*ix)[1], (*iy)[1], (*iz)[1]));
							v = vr*vr + vi * vi;
						}
						else {
							if(fftx) v+= (*ix)[0]*(*ix)[0] + (*ix)[1]*(*ix)[1];
							if(ffty) v+= (*iy)[0]*(*iy)[0] + (*iy)[1]*(*iy)[1];
							if(fftz) v+= (*iz)[0]*(*iz)[0] + (*iz)[1]*(*iz)[1];
						}
						v = sqrt(v);
                        colv[idx] = v * normalize;
                        colx[idx] = ((*ix)[0]*(*ix)[0] + (*ix)[1]*(*ix)[1]) * normalize;
                        coly[idx] = ((*iy)[0]*(*iy)[0] + (*iy)[1]*(*iy)[1]) * normalize;
                        colz[idx] = ((*iz)[0]*(*iz)[0] + (*iz)[1]*(*iz)[1]) * normalize;
                        cols[idx] = 0;
						idx++;
					}
				}
			}
            tr[ *m_wave3D].setColumn(0, std::move(colh), 5);
            tr[ *m_wave3D].setColumn(1, std::move(colk), 5);
            tr[ *m_wave3D].setColumn(2, std::move(coll), 5);
            tr[ *m_wave3D].setColumn(3, std::move(colv), 5);
            tr[ *m_wave3D].setColumn(4, std::move(colx), 5);
            tr[ *m_wave3D].setColumn(5, std::move(coly), 5);
            tr[ *m_wave3D].setColumn(6, std::move(colz), 5);
            tr[ *m_wave3D].setColumn(7, std::move(cols), 5);
            m_wave3D->drawGraph(tr);
        });
	}
	if(calcasite) {
        m_wave3D->iterate_commit([&](Transaction &tr){
            int idx = 0;
            tr[ *m_wave3D].setRowCount(16*size*size*size);
            unsigned int len = tr[ *m_wave3D].rowCount();
            std::vector<double> colh(len), colk(len), coll(len),
                colv(len), cols(len),
                colx(len), coly(len), colz(len);
            for(int site = 0; site < 16; site++) {
                const int *pos = cg_ASitePositions[site];
                for(int k = 0; k < size; k++) {
                    for(int j = 0; j < size; j++) {
                        for(int i = 0; i < size; i++) {
                            int sidx = size*(size*(size*site + k) + j) + i;
                            int fftidx = fftlen*(fftlen*(4*k+pos[2]) + 4*j+pos[1]) + 4*i + pos[0];
                            double x = i + pos[0] * 0.25;
                            double y = j + pos[1] * 0.25;
                            double z = k + pos[2] * 0.25;
                            colh[idx] = x;
                            colk[idx] = y;
                            coll[idx] = z;
                            double sx = shot[ *this].m_pFFTin[0][fftidx][0];
                            double sy = shot[ *this].m_pFFTin[1][fftidx][0];
                            double sz = shot[ *this].m_pFFTin[2][fftidx][0];
                            double v = 0.0;
                            if(along_field_dir) {
                                v = field_dir.innerProduct(MonteCarlo::Vector3<double>
                                                           (sx,sy,sz)) * A_MOMENT;
                            }
                            else {
                                if(calcp) {
                                    v = probabilities[sidx];
                                }
                                else {
                                    if(calch) {
                                        v = fields[sidx];
                                    }
                                }
                            }
                            colv[idx] = v;
                            colx[idx] = sx;
                            coly[idx] = sy;
                            colz[idx] = sz;
                            cols[idx] = site;
                            idx++;
                        }
                    }
                }
            }
            tr[ *m_wave3D].setColumn(0, std::move(colh), 5);
            tr[ *m_wave3D].setColumn(1, std::move(colk), 5);
            tr[ *m_wave3D].setColumn(2, std::move(coll), 5);
            tr[ *m_wave3D].setColumn(3, std::move(colv), 5);
            tr[ *m_wave3D].setColumn(4, std::move(colx), 5);
            tr[ *m_wave3D].setColumn(5, std::move(coly), 5);
            tr[ *m_wave3D].setColumn(6, std::move(colz), 5);
            tr[ *m_wave3D].setColumn(7, std::move(cols), 5);
            m_wave3D->drawGraph(tr);
        });
    }
	if(calcbsite) {
		std::vector<MonteCarlo::Vector3<double> > fields(16*size*size*size);
        stored.write_bsite(&fields[0]);
        m_wave3D->iterate_commit([&](Transaction &tr){
			int idx = 0;
			tr[ *m_wave3D].setRowCount(16*size*size*size);
            unsigned int len = tr[ *m_wave3D].rowCount();
            std::vector<double> colh(len), colk(len), coll(len),
                colv(len), cols(len),
                colx(len), coly(len), colz(len);
            for(int site = 0; site < 16; site++) {
				const int *pos = cg_BSitePositions[site];
				for(int k = 0; k < size; k++) {
					for(int j = 0; j < size; j++) {
						for(int i = 0; i < size; i++) {
							int sidx = size*(size*(size*site + k) + j) + i;
							MonteCarlo::Vector3<double> h(fields[sidx]);
							double x = i + pos[0] * 0.25;
							double y = j + pos[1] * 0.25;
							double z = k + pos[2] * 0.25;
                            colh[idx] = x;
                            colk[idx] = y;
                            coll[idx] = z;
                            colv[idx] = h.abs();
                            colx[idx] = h.x;
                            coly[idx] = h.y;
                            colz[idx] = h.z;
                            cols[idx] = site;
							idx++;
						}
					}
				}
			}
            tr[ *m_wave3D].setColumn(0, std::move(colh), 5);
            tr[ *m_wave3D].setColumn(1, std::move(colk), 5);
            tr[ *m_wave3D].setColumn(2, std::move(coll), 5);
            tr[ *m_wave3D].setColumn(3, std::move(colv), 5);
            tr[ *m_wave3D].setColumn(4, std::move(colx), 5);
            tr[ *m_wave3D].setColumn(5, std::move(coly), 5);
            tr[ *m_wave3D].setColumn(6, std::move(colz), 5);
            tr[ *m_wave3D].setColumn(7, std::move(cols), 5);
            m_wave3D->drawGraph(tr);
        });
	}
	if(calc8asite) {
		std::vector<MonteCarlo::Vector3<double> > fields(8*size*size*size);
        stored.write_8asite(&fields[0]);
        m_wave3D->iterate_commit([&](Transaction &tr){
			int idx = 0;
			tr[ *m_wave3D].setRowCount(8*size*size*size);
            unsigned int len = tr[ *m_wave3D].rowCount();
            std::vector<double> colh(len), colk(len), coll(len),
                colv(len), cols(len),
                colx(len), coly(len), colz(len);
            for(int site = 0; site < 8; site++) {
				const int *pos = cg_8aSitePositions[site];
				for(int k = 0; k < size; k++) {
					for(int j = 0; j < size; j++) {
						for(int i = 0; i < size; i++) {
							int sidx = size*(size*(size*site + k) + j) + i;
							MonteCarlo::Vector3<double> h(fields[sidx]);
							double x = i + pos[0] * 0.125;
							double y = j + pos[1] * 0.125;
							double z = k + pos[2] * 0.125;
                            colh[idx] = x;
                            colk[idx] = y;
                            coll[idx] = z;
                            colv[idx] = h.abs();
                            colx[idx] = h.x;
                            coly[idx] = h.y;
                            colz[idx] = h.z;
                            cols[idx] = site;
							idx++;
						}
					}
				}
			}
            tr[ *m_wave3D].setColumn(0, std::move(colh), 5);
            tr[ *m_wave3D].setColumn(1, std::move(colk), 5);
            tr[ *m_wave3D].setColumn(2, std::move(coll), 5);
            tr[ *m_wave3D].setColumn(3, std::move(colv), 5);
            tr[ *m_wave3D].setColumn(4, std::move(colx), 5);
            tr[ *m_wave3D].setColumn(5, std::move(coly), 5);
            tr[ *m_wave3D].setColumn(6, std::move(colz), 5);
            tr[ *m_wave3D].setColumn(7, std::move(cols), 5);
            m_wave3D->drawGraph(tr);
        });
	}
	if(calc48fsite) {
		std::vector<MonteCarlo::Vector3<double> > fields(48*size*size*size);
        stored.write_48fsite(&fields[0]);
        m_wave3D->iterate_commit([&](Transaction &tr){
			int idx = 0;
			tr[ *m_wave3D].setRowCount(48*size*size*size);
            unsigned int len = tr[ *m_wave3D].rowCount();
            std::vector<double> colh(len), colk(len), coll(len),
                colv(len), cols(len),
                colx(len), coly(len), colz(len);
            for(int site = 0; site < 48; site++) {
				const double *pos = cg_48fSitePositions[site];
				for(int k = 0; k < size; k++) {
					for(int j = 0; j < size; j++) {
						for(int i = 0; i < size; i++) {
							int sidx = size*(size*(size*site + k) + j) + i;
							MonteCarlo::Vector3<double> h(fields[sidx]);
							double x = i + pos[0] * 0.125;
							double y = j + pos[1] * 0.125;
							double z = k + pos[2] * 0.125;
                            colh[idx] = x;
                            colk[idx] = y;
                            coll[idx] = z;
                            colv[idx] = h.abs();
                            colx[idx] = h.x;
                            coly[idx] = h.y;
                            colz[idx] = h.z;
                            cols[idx] = site;
							idx++;
						}
					}
				}
			}
            tr[ *m_wave3D].setColumn(0, std::move(colh), 5);
            tr[ *m_wave3D].setColumn(1, std::move(colk), 5);
            tr[ *m_wave3D].setColumn(2, std::move(coll), 5);
            tr[ *m_wave3D].setColumn(3, std::move(colv), 5);
            tr[ *m_wave3D].setColumn(4, std::move(colx), 5);
            tr[ *m_wave3D].setColumn(5, std::move(coly), 5);
            tr[ *m_wave3D].setColumn(6, std::move(colz), 5);
            tr[ *m_wave3D].setColumn(7, std::move(cols), 5);
            m_wave3D->drawGraph(tr);
        });
	}
	if(writeflips) {
		std::deque<MonteCarlo::FlipHistory> flips;
        stored.write_flips(flips);

        m_wave3D->iterate_commit([&](Transaction &tr){
			tr[ *m_wave3D].setRowCount(flips.size());
            unsigned int len = tr[ *m_wave3D].rowCount();
            std::vector<double> colh(len), colk(len), coll(len),
                colv(len), cols(len),
                colx(len), coly(len), colz(len);
            for(int idx = 0; idx < (int)flips.size(); idx++) {
				int lidx = flips[idx].lidx;
				int site = flips[idx].site;
				int i = lidx % size;
				lidx /= size;
				int j = lidx % size;
				lidx /= size;
				int k = lidx % size;

				const int *pos = cg_ASitePositions[site];
				double x = i + pos[0] * 0.25;
				double y = j + pos[1] * 0.25;
				double z = k + pos[2] * 0.25;
                colh[idx] = x;
                colk[idx] = y;
                coll[idx] = z;
                colv[idx] = (flips[idx].delta > 0.0) ? 2.0 : 1.0;
                colx[idx] = flips[idx].delta;
                coly[idx] = flips[idx].tests;
                colz[idx] = 0;
                cols[idx] = site;
			}
            tr[ *m_wave3D].setColumn(0, std::move(colh), 5);
            tr[ *m_wave3D].setColumn(1, std::move(colk), 5);
            tr[ *m_wave3D].setColumn(2, std::move(coll), 5);
            tr[ *m_wave3D].setColumn(3, std::move(colv), 5);
            tr[ *m_wave3D].setColumn(4, std::move(colx), 5);
            tr[ *m_wave3D].setColumn(5, std::move(coly), 5);
            tr[ *m_wave3D].setColumn(6, std::move(colz), 5);
            tr[ *m_wave3D].setColumn(7, std::move(cols), 5);
            m_wave3D->drawGraph(tr);
        });
	}
}
void
XMonteCarloDriver::onGraphChanged(const Snapshot &shot, XValueNodeBase *) {
	Snapshot shot_this( *this);
	visualize(shot_this);
  
}
void
XMonteCarloDriver::onTargetChanged(const Snapshot &shot, XValueNodeBase *) {
	Snapshot shot_this( *this);
    int size = m_loop->length();
	int spin_size = size*size*size*4*4;
	int flips = (int)(shot_this[ *m_minFlips] * spin_size);
	long double tests = shot_this[ *m_minTests] * spin_size;
	execute(flips, tests);
}
void
XMonteCarloDriver::onStepTouched(const Snapshot &shot, XTouchableNode *) {
	execute(1, 1);
}
void
XMonteCarloDriver::execute(int flips, long double tests) {
    Snapshot shot_this( *this);
    unsigned int size = m_loop->length();
    MonteCarlo::Vector3<double> field_dir(shot_this[ *m_hdirx], shot_this[ *m_hdiry], shot_this[ *m_hdirz]);
    field_dir.normalize();
    MonteCarlo::Vector3<double> field(field_dir);
    MonteCarlo::Vector3<double> mav;
    field *= shot_this[ *m_targetField];
    double dU_av;
    double dU = m_loop->exec(shot_this[ *m_targetTemp], field, &flips, &tests, &dU_av, &mav) * N_A;
    double mag = m_loop->magnetization().innerProduct(field_dir);
    auto writer = std::make_shared<RawData>();
    std::vector<char> buf(size*size*size*16);
    writer->push(dU);
    writer->push(dU_av);
    writer->push(flips);
    writer->push((double)tests);
    writer->push(mav.innerProduct(field_dir));
    writer->push(mag);
    MonteCarlo::Quartet quartet = m_loop->siteMagnetization();
    writer->push(quartet.onethree);
    writer->push(quartet.twotwo);
    if(rand() % 20 == 0) {
        std::cerr << "Recalculate Internal Energy." << std::endl;
        double u = m_loop->internalEnergy() * N_A;
        writer->push(u);
    }
    else
        writer->push(0.0);
    m_loop->write((char*)&buf.at(0));
    writer->insert(writer->end(), buf.begin(), buf.end());
    finishWritingRaw(writer, XTime::now(), XTime::now());
}
