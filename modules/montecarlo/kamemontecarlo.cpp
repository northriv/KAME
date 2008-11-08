/***************************************************************************
		Copyright (C) 2002-2008 Kentaro Kitagawa
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
#include <qstatusbar.h>
#include <qpushbutton.h>
#include "montecarlo.h"
#include "kamemontecarlo.h"
#include "forms/montecarloform.h"
#include "graph.h"
#include "graphwidget.h"
#include "xwavengraph.h"

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
									 const shared_ptr<XScalarEntryList> &scalarentries,
									 const shared_ptr<XInterfaceList> &interfaces,
									 const shared_ptr<XThermometerList> &thermometers,
									 const shared_ptr<XDriverList> &drivers) :
    XDummyDriver<XPrimaryDriver>(name, runtime, scalarentries, interfaces, thermometers, drivers),
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
    m_step(create<XNode>("Step", true)),
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
    m_form(new FrmMonteCarlo(g_pFrmMain)),
    m_wave3D(create<XWaveNGraph>("Wave3D", false, 
								 m_form->m_graph, m_form->m_urlDump, m_form->m_btnDump)),
    m_statusPrinter(XStatusPrinter::create(m_form.get())),
    m_fftlen(-1)
{
	scalarentries->insert(m_entryT);
	scalarentries->insert(m_entryH);
	scalarentries->insert(m_entryU);
	scalarentries->insert(m_entryC);
	scalarentries->insert(m_entryCoT);
	scalarentries->insert(m_entryS);
	scalarentries->insert(m_entryM);
	scalarentries->insert(m_entry2in2);
	scalarentries->insert(m_entry1in3);
	m_targetTemp->value(100.0);
	m_hdirx->value(1.0);
	m_hdiry->value(1.0);
	m_hdirz->value(1.0);
	m_L->value(8);
	m_cutoffReal->value(4.0);
	m_cutoffRec->value(2.0);
	m_alpha->value(0.5);
	m_minTests->value(1.0);
	m_minFlips->value(0.2);
  
//  for(const FFTAxis *axis = c_fftaxes; axis->desc; axis++) {
//    m_fftAxisX->add(axis->desc);
//    m_fftAxisY->add(axis->desc);
//  }
//  m_fftAxisX->value(0);
//  m_fftAxisY->value(1);
  
	m_graph3D->add(GRAPH_3D_FFT_INTENS_ABS);
	m_graph3D->add(GRAPH_3D_FFT_INTENS_X);
	m_graph3D->add(GRAPH_3D_FFT_INTENS_Y);
	m_graph3D->add(GRAPH_3D_FFT_INTENS_Z);
	m_graph3D->add(GRAPH_3D_FFT_INTENS_M);
	m_graph3D->add(GRAPH_REAL_M);
	m_graph3D->add(GRAPH_REAL_H);
	m_graph3D->add(GRAPH_REAL_P);
	m_graph3D->add(GRAPH_REAL_H_B_SITE);
	m_graph3D->add(GRAPH_REAL_H_8a_SITE);
	m_graph3D->add(GRAPH_REAL_H_48f_SITE);
	m_graph3D->add(GRAPH_FLIPS);
  
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
	m_conGraph3D = xqcon_create<XQComboBoxConnector>(m_graph3D, m_form->m_cmbGraph3D);

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

    const char *s_trace_names[] = {
		"h or x", "k or y", "l or z", "intens.", "hx", "hy", "hz", "site"
    };
	m_wave3D->setColCount(8, s_trace_names); 
	m_wave3D->insertPlot("Intens.", 0, 1, -1, 3, 2);
	m_wave3D->clear();
	m_wave3D->plot(0)->drawLines()->value(false);

	m_wave3D->graph()->backGround()->value(QColor(0,0,0).rgb());
	m_wave3D->plot(0)->intensity()->value(2);
	m_wave3D->plot(0)->colorPlot()->value(true);
	m_wave3D->plot(0)->colorPlotColorHigh()->value(QColor(0xFF, 0xFF, 0x2F).rgb());
	m_wave3D->plot(0)->colorPlotColorLow()->value(QColor(0x00, 0x00, 0xFF).rgb());
	m_wave3D->plot(0)->pointColor()->value(QColor(0x00, 0xFF, 0x00).rgb());
	m_wave3D->plot(0)->majorGridColor()->value(QColor(0x4A, 0x4A, 0x4A).rgb());
	m_wave3D->graph()->titleColor()->value(clWhite);
	m_wave3D->axisx()->ticColor()->value(clWhite);
	m_wave3D->axisx()->labelColor()->value(clWhite);
	m_wave3D->axisx()->ticLabelColor()->value(clWhite);  
	m_wave3D->axisy()->ticColor()->value(clWhite);
	m_wave3D->axisy()->labelColor()->value(clWhite);
	m_wave3D->axisy()->ticLabelColor()->value(clWhite);  
	m_wave3D->axisz()->ticColor()->value(clWhite);
	m_wave3D->axisz()->labelColor()->value(clWhite);
	m_wave3D->axisz()->ticLabelColor()->value(clWhite);  
}
XMonteCarloDriver::~XMonteCarloDriver() {
    for(int d = 0; d < 3; d++) {
        if(m_fftlen > 0) {
            fftw_destroy_plan(m_fftplan[d]);
            fftw_free(m_pFFTin[d]);
            fftw_free(m_pFFTout[d]);
        }
    }	
}
void
XMonteCarloDriver::showForms() {
//! impliment form->show() here
    m_form->show();
    m_form->raise();
}
void
XMonteCarloDriver::start()
{
    MonteCarlo::setupField(*m_L, 0.0, *m_cutoffReal, *m_cutoffRec, *m_alpha);
    m_L->setUIEnabled(false);
    m_loop.reset(new MonteCarlo(2));
    m_store.reset(new MonteCarlo(1));
    m_cutoffReal->setUIEnabled(false);
    m_cutoffRec->setUIEnabled(false);
    m_alpha->setUIEnabled(false);
    m_sumDU = m_loop->internalEnergy() * N_A;
    m_sumDUav = m_sumDU;
    m_sumDS = 0.0;
    m_testsTotal = 0.0;
    m_flippedTotal = 0.0;
    m_lastTemp = 300.0;
    m_lastField = 0.0;
    m_dU = 0.0;
    m_hdirx->setUIEnabled(false);
    m_hdiry->setUIEnabled(false);
    m_hdirz->setUIEnabled(false);
    MonteCarlo::Vector3<double> field_dir(*m_hdirx,*m_hdiry,*m_hdirz);
    field_dir.normalize();
    m_lastMagnetization = m_loop->magnetization().innerProduct(field_dir);

    m_lsnTargetChanged = m_targetTemp->onValueChanged().connectWeak(
		shared_from_this(), &XMonteCarloDriver::onTargetChanged);
    m_targetField->onValueChanged().connect(m_lsnTargetChanged);
    m_lsnStepTouched = m_step->onTouch().connectWeak(
		shared_from_this(), &XMonteCarloDriver::onStepTouched);
    m_lsnGraphChanged = m_graph3D->onValueChanged().connectWeak(
		shared_from_this(), &XMonteCarloDriver::onGraphChanged);

    int fftlen = MonteCarlo::length() * 4;
    for(int d = 0; d < 3; d++) {
        if(m_fftlen > 0) {
            fftw_destroy_plan(m_fftplan[d]);
            fftw_free(m_pFFTin[d]);
            fftw_free(m_pFFTout[d]);
        }
        m_pFFTin[d] = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftlen * fftlen * fftlen);
        m_pFFTout[d] = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftlen * fftlen * fftlen);
        m_fftplan[d] = fftw_plan_dft_3d(fftlen, fftlen, fftlen, m_pFFTin[d], m_pFFTout[d],
        	FFTW_FORWARD, FFTW_ESTIMATE);
    }
    m_fftlen = fftlen;
}
void
XMonteCarloDriver::stop()
{
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
    
    afterStop();
}
void
XMonteCarloDriver::analyzeRaw() throw (XRecordError&)
{
    unsigned int size = MonteCarlo::length();
    if(rawData().size() != size*size*size*16)
        throw XRecordError("Size Mismatch", __FILE__, __LINE__);
    MonteCarlo::Vector3<double> field_dir(*m_hdirx,*m_hdiry,*m_hdirz);
    field_dir.normalize();
    MonteCarlo::Vector3<double> field(field_dir);
    field *= *m_targetField;
    char *spins = (char*)&rawData()[0];
    m_store->read(spins, *m_targetTemp, field);

    double dT = *m_targetTemp - m_lastTemp;
    m_lastTemp = *m_targetTemp;
    double m = m_store->magnetization().innerProduct(field_dir);
    double dM = m - m_lastMagnetization;
    double dH = *m_targetField - m_lastField;
    double dU = m_dU;
    // DS += (DU + HDM + MDH)/T
    m_sumDS += (dU + dM * *m_targetField * MU_B * N_A) / *m_targetTemp; 
    // DU += -MDH.
    dU += -m_lastMagnetization * MU_B * N_A * dH;
    double dUav = m_DUav * N_A;
    dUav += -m_Mav * MU_B * N_A * dH;
    double c = (fabs(dT) < 1e-10) ? 0.0 : ((m_sumDU + dUav - m_sumDUav) / dT);
    m_sumDUav = m_sumDU + dUav;
    m_sumDU += dU;
    m_lastMagnetization = m;
    m_lastField = *m_targetField;
    double u = m_sumDU;
    if(rand() % 20 == 0) {
        std::cerr << "Recalculate Internal Energy." << std::endl;
        u = m_store->internalEnergy() * N_A;
        if(fabs((u - m_sumDU)/u) > 1e-5) {
            gErrPrint(formatString("SumDU Error = %g!", (double)(u - m_sumDU)/u)); 
        }
    }
    MonteCarlo::Quartet quartet = m_store->siteMagnetization();
    m_entryT->value(m_lastTemp);
    m_entryH->value(m_lastField);
    m_entryU->value(u);
    m_entryC->value(c);
    m_entryCoT->value(c / m_lastTemp);
    m_entryS->value(m_sumDS);
    m_entryM->value(m_Mav);
    m_entry2in2->value(100.0*quartet.twotwo);
    m_entry1in3->value(100.0*quartet.onethree);
}
void
XMonteCarloDriver::visualize()
{
	//! impliment extra codes which do not need write-lock of record
	//! record is read-locked
	if(!time()) {
		m_wave3D->clear();
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
	if(m_graph3D->to_str() == GRAPH_3D_FFT_INTENS_ABS) {
		fftx = true;
		ffty = true;
		fftz = true;
		fft_intens_abs = true;
	}
	if(m_graph3D->to_str() == GRAPH_3D_FFT_INTENS_X) {
		fftx = true;
	}
	if(m_graph3D->to_str() == GRAPH_3D_FFT_INTENS_Y) {
		ffty = true;
	}
	if(m_graph3D->to_str() == GRAPH_3D_FFT_INTENS_Z) {
		fftz = true;
	}
	if(m_graph3D->to_str() == GRAPH_3D_FFT_INTENS_M) {
		fftx = true;
		ffty = true;
		fftz = true;
		along_field_dir = true;
	}
	if(m_graph3D->to_str() == GRAPH_REAL_M) {
		calcasite = true;
		along_field_dir = true;
	}
	if(m_graph3D->to_str() == GRAPH_REAL_H) {
		calcasite = true;
		calch = true;
	}
	if(m_graph3D->to_str() == GRAPH_REAL_P) {
		calcasite = true;
		calch = true;
		calcp = true;
	}
	if(m_graph3D->to_str() == GRAPH_REAL_H_B_SITE) {
		calcbsite = true;
	}
	if(m_graph3D->to_str() == GRAPH_REAL_H_8a_SITE) {
		calc8asite = true;
	}
	if(m_graph3D->to_str() == GRAPH_REAL_H_48f_SITE) {
		calc48fsite = true;
	}
	if(m_graph3D->to_str() == GRAPH_FLIPS) {
		writeflips = true;
	}

    int size = m_store->length();
    std::vector<char> spins(size*size*size*16);
    std::vector<double> fields;
    if(calch) fields.resize(size*size*size*16);
    std::vector<double> probabilities;
    if(calcp) probabilities.resize(size*size*size*16);
    m_store->write((char*)&spins[0]
				   , fields.size() ? &fields[0] : 0, probabilities.size() ? &probabilities[0] : 0);
    for(int d = 0; d < 3; d++) {
        for(int site = 0; site < 16; site++) {
            const int *pos = cg_ASitePositions[site];
            int ising = cg_ASiteIsingAxes[site][d];
            for(int k = 0; k < size; k++) {
				for(int j = 0; j < size; j++) {
					for(int i = 0; i < size; i++) {
						int fftidx = m_fftlen*(m_fftlen*(4*k+pos[2]) + 4*j+pos[1]) + 4*i + pos[0];
						int spin = spins[size*(size*(size*site + k) + j) + i];
						m_pFFTin[d][fftidx][0] = spin* ising / sqrt(3.0);
					}
				}
            }
        }
    }
    
	MonteCarlo::Vector3<double> field_dir(*m_hdirx,*m_hdiry,*m_hdirz);
	field_dir.normalize();
    

	if(fftx || ffty || fftz)
	{
		fftw_execute(m_fftplan[0]);
		fftw_execute(m_fftplan[1]);
		fftw_execute(m_fftplan[2]);
    
		XScopedWriteLock<XWaveNGraph> lock(*m_wave3D);
		m_wave3D->setRowCount(m_fftlen * m_fftlen * m_fftlen );
		double normalize = A_MOMENT / m_fftlen / m_fftlen / m_fftlen;
		int idx = 0;
		for(int l = 0; l < m_fftlen; l++) {
			for(int k = 0; k < m_fftlen; k++) {
				for(int h = 0; h < m_fftlen; h++) {
					int qidx = m_fftlen*(m_fftlen*l + k) + h;
					fftw_complex *ix = &m_pFFTout[0][qidx];
					fftw_complex *iy = &m_pFFTout[1][qidx];
					fftw_complex *iz = &m_pFFTout[2][qidx];
					m_wave3D->cols(0)[idx] = (double)h * 8.0/m_fftlen;
					m_wave3D->cols(1)[idx] = (double)k * 8.0/m_fftlen;
					m_wave3D->cols(2)[idx] = (double)l * 8.0/m_fftlen;
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
					m_wave3D->cols(3)[idx] = v * normalize;
					m_wave3D->cols(4)[idx] = ((*ix)[0]*(*ix)[0] + (*ix)[1]*(*ix)[1]) * normalize;
					m_wave3D->cols(5)[idx] = ((*iy)[0]*(*iy)[0] + (*iy)[1]*(*iy)[1]) * normalize;
					m_wave3D->cols(6)[idx] = ((*iz)[0]*(*iz)[0] + (*iz)[1]*(*iz)[1]) * normalize;
					m_wave3D->cols(7)[idx] = 0;
					idx++;
				}
			}
		}
	}
	if(calcasite)
	{   XScopedWriteLock<XWaveNGraph> lock(*m_wave3D);
	int idx = 0;
	m_wave3D->setRowCount(16*size*size*size);
	for(int site = 0; site < 16; site++) {
		const int *pos = cg_ASitePositions[site];
		for(int k = 0; k < size; k++) {
			for(int j = 0; j < size; j++) {
				for(int i = 0; i < size; i++) {
					int sidx = size*(size*(size*site + k) + j) + i;
					int fftidx = m_fftlen*(m_fftlen*(4*k+pos[2]) + 4*j+pos[1]) + 4*i + pos[0];
					double x = i + pos[0] * 0.25;
					double y = j + pos[1] * 0.25;
					double z = k + pos[2] * 0.25;
					m_wave3D->cols(0)[idx] = x;
					m_wave3D->cols(1)[idx] = y;
					m_wave3D->cols(2)[idx] = z;
					double sx = m_pFFTin[0][fftidx][0];
					double sy = m_pFFTin[1][fftidx][0];
					double sz = m_pFFTin[2][fftidx][0];
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
					m_wave3D->cols(3)[idx] = v;
					m_wave3D->cols(4)[idx] = sx;
					m_wave3D->cols(5)[idx] = sy;
					m_wave3D->cols(6)[idx] = sz;
					m_wave3D->cols(7)[idx] = site;
					idx++;
				}
			}
		}
	}
	}
	if(calcbsite)
	{
		std::vector<MonteCarlo::Vector3<double> > fields(16*size*size*size);
		m_store->write_bsite(&fields[0]);
		XScopedWriteLock<XWaveNGraph> lock(*m_wave3D);
		int idx = 0;
		m_wave3D->setRowCount(16*size*size*size);
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
						m_wave3D->cols(0)[idx] = x;
						m_wave3D->cols(1)[idx] = y;
						m_wave3D->cols(2)[idx] = z;
						m_wave3D->cols(3)[idx] = h.abs();
						m_wave3D->cols(4)[idx] = h.x;
						m_wave3D->cols(5)[idx] = h.y;
						m_wave3D->cols(6)[idx] = h.z;
						m_wave3D->cols(7)[idx] = site;
						idx++;
					}
				}
			}
		}
	}
	if(calc8asite)
	{
		std::vector<MonteCarlo::Vector3<double> > fields(8*size*size*size);
		m_store->write_8asite(&fields[0]);
		XScopedWriteLock<XWaveNGraph> lock(*m_wave3D);
		int idx = 0;
		m_wave3D->setRowCount(8*size*size*size);
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
						m_wave3D->cols(0)[idx] = x;
						m_wave3D->cols(1)[idx] = y;
						m_wave3D->cols(2)[idx] = z;
						m_wave3D->cols(3)[idx] = h.abs();
						m_wave3D->cols(4)[idx] = h.x;
						m_wave3D->cols(5)[idx] = h.y;
						m_wave3D->cols(6)[idx] = h.z;
						m_wave3D->cols(7)[idx] = site;
						idx++;
					}
				}
			}
		}
	}
	if(calc48fsite)
	{
		std::vector<MonteCarlo::Vector3<double> > fields(48*size*size*size);
		m_store->write_48fsite(&fields[0]);
		XScopedWriteLock<XWaveNGraph> lock(*m_wave3D);
		int idx = 0;
		m_wave3D->setRowCount(48*size*size*size);
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
						m_wave3D->cols(0)[idx] = x;
						m_wave3D->cols(1)[idx] = y;
						m_wave3D->cols(2)[idx] = z;
						m_wave3D->cols(3)[idx] = h.abs();
						m_wave3D->cols(4)[idx] = h.x;
						m_wave3D->cols(5)[idx] = h.y;
						m_wave3D->cols(6)[idx] = h.z;
						m_wave3D->cols(7)[idx] = site;
						idx++;
					}
				}
			}
		}
	}
	if(writeflips) {
		std::deque<MonteCarlo::FlipHistory> flips;
		m_loop->write_flips(flips);

		XScopedWriteLock<XWaveNGraph> lock(*m_wave3D);
		m_wave3D->setRowCount(flips.size());
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
			m_wave3D->cols(0)[idx] = x;
			m_wave3D->cols(1)[idx] = y;
			m_wave3D->cols(2)[idx] = z;
			m_wave3D->cols(3)[idx] = (flips[idx].delta > 0.0) ? 2.0 : 1.0;
			m_wave3D->cols(4)[idx] = flips[idx].delta;
			m_wave3D->cols(5)[idx] = flips[idx].tests;
			m_wave3D->cols(6)[idx] = 0;
			m_wave3D->cols(7)[idx] = site;
		}
	}
}
void
XMonteCarloDriver::onGraphChanged(const shared_ptr<XValueNodeBase> &)
{
	readLockRecord();
	visualize();
	readUnlockRecord();
  
}
void
XMonteCarloDriver::onTargetChanged(const shared_ptr<XValueNodeBase> &)
{
    int size = m_loop->length();
    int spin_size = size*size*size*4*4;
    int flips = (int)(*m_minFlips * spin_size);
    long double tests = *m_minTests * spin_size;
    execute(flips, tests);
}
void
XMonteCarloDriver::onStepTouched(const shared_ptr<XNode> &)
{
    execute(1, 1);
}
void
XMonteCarloDriver::execute(int flips, long double tests)
{
    unsigned int size = m_loop->length();
    int spin_size = size*size*size*4*4;
    MonteCarlo::Vector3<double> field_dir(*m_hdirx,*m_hdiry,*m_hdirz);
    field_dir.normalize();
    MonteCarlo::Vector3<double> field(field_dir);
    field *= *m_targetField;
    MonteCarlo::Vector3<double> m;
    m_dU = m_loop->exec(*m_targetTemp, field, &flips, &tests, &m_DUav, &m) * N_A;
    m_testsTotal += tests;
    m_flippedTotal += flips;
    fprintf(stderr, "Total flips = %g (%g per spin).\n",
    	((double)m_flippedTotal), ((double)m_flippedTotal / spin_size));
    m_Mav = m.innerProduct(field_dir);
    fprintf(stderr, "Total tests = %g (%g per spin).\n",
    	((double)m_testsTotal), ((double)m_testsTotal / spin_size));
    rawData().resize(size*size*size*16);
    m_loop->write((char*)&rawData()[0]);
    finishWritingRaw(XTime::now(), XTime::now());
}
