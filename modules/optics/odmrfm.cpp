/***************************************************************************
        Copyright (C) 2002-2015 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------
#include "odmrfm.h"
#include "ui_odmrfmform.h"

#include "analyzer.h"
#include "xnodeconnector.h"

REGISTER_TYPE(XDriverList, ODMRFMControl, "ODMR peak tracker by FM");

//---------------------------------------------------------------------------
XODMRFMControl::XODMRFMControl(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XSecondaryDriver(name, runtime, ref(tr_meas), meas),
        m_entryFreq(create<XScalarEntry>("Freq", false,
            dynamic_pointer_cast<XDriver>(shared_from_this()))),
        m_entryTesla(create<XScalarEntry>("Tesla", false,
            dynamic_pointer_cast<XDriver>(shared_from_this()))),
        m_entryTeslaErr(create<XScalarEntry>("TeslaErr", false,
            dynamic_pointer_cast<XDriver>(shared_from_this()))),
        m_entryFMIntens(create<XScalarEntry>("FMIntens", false,
            dynamic_pointer_cast<XDriver>(shared_from_this()))),
        m_sg(create<XItemNode<XDriverList, XSG> >("SG", false, ref(tr_meas), meas->drivers(), true)),
        m_lia(create<XItemNode<XDriverList, XLIA> >("LIA", false, ref(tr_meas), meas->drivers(), true)),
        m_gamma2pi(create<XDoubleNode>("Gamma2pi", false)),
        m_fmIntensRequired(create<XDoubleNode>("Gamma2pi", false)),
        m_numReadings(create<XUIntNode>("NumReadings", false)),
        m_form(new FrmODMRFM) {

    connect(sg());
    connect(lia());

    meas->scalarEntries()->insert(tr_meas, entryFreq());
    meas->scalarEntries()->insert(tr_meas, entryTesla());
    meas->scalarEntries()->insert(tr_meas, entryTeslaErr());
    meas->scalarEntries()->insert(tr_meas, entryFMIntens());

    iterate_commit([=](Transaction &tr){
        tr[ *gamma2pi()] = 28024.95142; //g=2.002319
        tr[ *fmIntensRequired()] = 0.1;
        tr[ *numReadings()] = 20;
    });

    m_form->setWindowTitle(i18n("ODMR peak tracker by FM - ") + getLabel() );

    m_conUIs = {
        xqcon_create<XQLineEditConnector>(m_gamma2pi, m_form->m_gamma2pi),
        xqcon_create<XQLineEditConnector>(m_fmIntensRequired, m_form->m_fmIntensRequired),
        xqcon_create<XQSpinBoxUnsignedConnector>(numReadings(), m_form->m_numReadings),
    };


}
XODMRFMControl::~XODMRFMControl() {
}


bool XODMRFMControl::checkDependency(const Snapshot &shot_this,
    const Snapshot &shot_emitter, const Snapshot &shot_others,
    XDriver *emitter) const {
    const shared_ptr<XSG> sg__ = shot_this[ *sg()];
    if( !sg__)
        return false;
    const shared_ptr<XLIA> lia__ = shot_this[ *lia()];
    if (emitter == lia__.get())
        return true;
    return false;
}
void XODMRFMControl::analyze(Transaction &tr, const Snapshot &shot_emitter,
    const Snapshot &shot_others,
    XDriver *emitter) {
    const Snapshot &shot_this(tr);
    const shared_ptr<XLIA> lia__ = shot_this[ *lia()];
    assert(lia__);
    const shared_ptr<XSG> sg__ = shot_this[ *sg()];

    if(pos >= dso_len) {
        throw XSkippedRecordError(i18n("Position beyond waveforms."), __FILE__, __LINE__);
    }
    if(pos < 0) {
        throw XSkippedRecordError(i18n("Position beyond waveforms."), __FILE__, __LINE__);
    }
    int length = lrint( shot_this[ *width()] / 1000 / interval);
    if(pos + length >= dso_len) {
        throw XSkippedRecordError(i18n("Invalid length."), __FILE__, __LINE__);
    }
    if(length <= 0) {
        throw XSkippedRecordError(i18n("Invalid length."), __FILE__, __LINE__);
    }

    int bgpos = lrint((shot_this[ *bgPos()] - shot_this[ *fromTrig()]) / 1000 / interval);
    if(pos + bgpos >= dso_len) {
        throw XSkippedRecordError(i18n("Position for BG. sub. beyond waveforms."), __FILE__, __LINE__);
    }
    if(pos + bgpos < 0) {
        throw XSkippedRecordError(i18n("Position for BG. sub. beyond waveforms."), __FILE__, __LINE__);
    }
    int bglength = lrint(shot_this[ *bgWidth()] / 1000 / interval);
    if(pos + bgpos + bglength >= dso_len) {
        throw XSkippedRecordError(i18n("Invalid Length for BG. sub."), __FILE__, __LINE__);
    }
    if(bglength < 0) {
        throw XSkippedRecordError(i18n("Invalid Length for BG. sub."), __FILE__, __LINE__);
    }

    shared_ptr<XPulser> pulse__(shot_this[ *pulser()]);

    int echoperiod = lrint(shot_this[ *echoPeriod()] / 1000 /interval);
    int numechoes = shot_this[ *numEcho()];
    numechoes = std::max(1, numechoes);
    int numechoes_pulse = numechoes;
    if(pulse__) numechoes_pulse = shot_others[ *pulse__].echoNum();
    bool bg_off_echotrain = (pos + length + echoperiod * (numechoes_pulse - 1) < pos + bgpos) || (pos + bgpos + bglength < pos);

    if(bglength && (bglength < length * numechoes * 3))
        m_statusPrinter->printWarning(i18n("Maybe, length for BG. sub. is too short."));

    if(bglength && !bg_off_echotrain)
        m_statusPrinter->printWarning(i18n("Maybe, position for BG. sub. is overrapped against echoes"), true);

    if(numechoes_pulse > 1) {
        if(pos + echoperiod * (numechoes_pulse - 1) + length >= dso_len) {
            throw XSkippedRecordError(i18n("Invalid Multiecho settings."), __FILE__, __LINE__);
        }
        if(echoperiod < length) {
            throw XSkippedRecordError(i18n("Invalid Multiecho settings."), __FILE__, __LINE__);
        }
        if(pulse__) {
            if((numechoes > shot_others[ *pulse__].echoNum()))
                throw XSkippedRecordError(i18n("Invalid Multiecho settings."), __FILE__, __LINE__);
            if(fabs(shot_this[ *echoPeriod()] * 1e3 / (shot_others[ *pulse__].tau() * 2.0) - 1.0) > 1e-4)
                m_statusPrinter->printWarning(i18n("Invalid Multiecho settings."), true);
        }
    }

    if((shot_this[ *this].m_startTime != starttime) || (length != shot_this[ *this].m_waveWidth)) {
        double t = length * interval * 1e3;
        tr[ *tr[ *waveGraph()].axisx()->autoScale()] = false;
        tr[ *tr[ *waveGraph()].axisx()->minValue()] = starttime * 1e3 - t * 0.3;
        tr[ *tr[ *waveGraph()].axisx()->maxValue()] = starttime * 1e3 + t * 1.3;
    }
    tr[ *this].m_waveWidth = length;
    bool skip = (shot_this[ *this].m_timeClearRequested);
    tr[ *this].m_timeClearRequested = {};
    bool avgclear = skip;

    if(interval != shot_this[ *this].m_interval) {
        //[sec]
        tr[ *this].m_interval = interval;
        avgclear = true;
    }
    if(shot_this[ *this].m_startTime != starttime) {
        //[sec]
        tr[ *this].m_startTime = starttime;
        avgclear = true;
    }

    if(length > (int)shot_this[ *this].m_waveSum.size()) {
        avgclear = true;
    }
    tr[ *this].m_wave.resize(length);
    tr[ *this].m_waveSum.resize(length);
    auto &echoesT2(tr[ *this].m_echoesT2);
    echoesT2.resize(numechoes_pulse);
    auto &echoesT2Sum(tr[ *this].m_echoesT2Sum);
    echoesT2Sum.resize(numechoes_pulse);
    for(int i = 0; i < numechoes_pulse; i++){
        echoesT2[i].resize(length);
        echoesT2Sum[i].resize(length);
    }
    int fftlen = FFT::fitLength(shot_this[ *fftLen()]);
    if(fftlen != shot_this[ *this].m_darkPSD.size()) {
        avgclear = true;
    }
    if(length > fftlen) {
        throw XSkippedRecordError(i18n("FFT length is too short."), __FILE__, __LINE__);
    }
    tr[ *this].m_darkPSD.resize(fftlen);
    tr[ *this].m_darkPSDSum.resize(fftlen);
    std::fill(tr[ *this].m_wave.begin(), tr[ *this].m_wave.end(), std::complex<double>(0.0));

    // Phase Inversion Cycling
    bool picenabled = shot_this[ *m_picEnabled];
    bool inverted = false;
    if(picenabled && ( !pulse__ || !shot_others[ *pulse__].time())) {
        picenabled = false;
        gErrPrint(getLabel() + ": " + i18n("No active pulser!"));
    }
    if(pulse__) {
        inverted = shot_others[ *pulse__].invertPhase();
    }

    int avgnum = std::max((unsigned int)shot_this[ *extraAvg()], 1u) * (picenabled ? 2 : 1);

    if( !shot_this[ *exAvgIncr()] && (avgnum <= shot_this[ *this].m_avcount)) {
        avgclear = true;
    }
    if(avgclear) {
        std::fill(tr[ *this].m_waveSum.begin(), tr[ *this].m_waveSum.end(), std::complex<double>(0.0));
        std::fill(tr[ *this].m_darkPSDSum.begin(), tr[ *this].m_darkPSDSum.end(), 0.0);
        for(int i = 0; i < numechoes_pulse; i++){
            std::fill(echoesT2Sum[i].begin(), echoesT2Sum[i].end(), std::complex<double>(0.0));
        }
        tr[ *this].m_avcount = 0;
        if(shot_this[ *exAvgIncr()]) {
            tr[ *extraAvg()] = 0;
        }
    }

    if(skip) {
        throw XSkippedRecordError(__FILE__, __LINE__);
    }

    tr[ *this].m_dsoWave.resize(dso_len);
    std::complex<double> *dsowave( &tr[ *this].m_dsoWave[0]);
    {
        const double *rawwavecos, *rawwavesin = NULL;
        assert(shot_dso[ *dso__].numChannels() );
        rawwavecos = shot_dso[ *dso__].wave(0);
        rawwavesin = shot_dso[ *dso__].wave(1);
        for(unsigned int i = 0; i < dso_len; i++) {
            dsowave[i] = std::complex<double>(rawwavecos[i], rawwavesin[i]) * (inverted ? -1.0 : 1.0);
        }
    }
    tr[ *this].m_dsoWaveStartPos = pos;

    //Background subtraction or dynamic noise reduction
    if(bg_off_echotrain)
        backgroundSub(tr, tr[ *this].m_dsoWave, pos, length, bgpos, bglength);
    for(int i = 0; i < numechoes_pulse; i++){
        int rpos = pos + i * echoperiod;
        std::complex<double> *pechoesT2(&echoesT2[i][0]);
        std::complex<double> *pechoesT2Sum(&echoesT2Sum[i][0]);
        for(int j = 0; j < length; j++){
            int k = rpos + j;
            assert(k < dso_len);
            pechoesT2[j] = dsowave[k];
            //increments for multi-echo T2
            if((emitter == dso__.get()) || ( !shot_this[ *this].m_avcount))
                pechoesT2Sum[j] += dsowave[k];
            if(i == 0)
                dsowave[pos + j] /= (double)numechoes;
            else if(numechoes > i)
                dsowave[pos + j] += dsowave[k] / (double)numechoes;
        }
    }

    std::complex<double> *wavesum( &tr[ *this].m_waveSum[0]);
    double *darkpsdsum( &tr[ *this].m_darkPSDSum[0]);
    //Incremental/Sequential average.
    if((emitter == dso__.get()) || ( !shot_this[ *this].m_avcount)) {
        double max_volt_abs = 0.0;
        for(int i = 0; i < length; i++) {
            auto z = dsowave[pos + i];
            wavesum[i] += z;
            max_volt_abs = std::max(max_volt_abs, std::abs(z));
        }
        if(max_volt_abs > shot_this[ *voltLimit()]) {
            throw XRecordError(i18n("Peak height exceeded limit voltage."), __FILE__, __LINE__);
        }
        if(bglength) {
            //Estimates power spectral density in the background.
            if( !shot_this[ *this].m_ftDark ||
                (shot_this[ *this].m_ftDark->length() != shot_this[ *this].m_darkPSD.size())) {
                tr[ *this].m_ftDark.reset(new FFT(-1, shot_this[ *fftLen()]));
            }
            std::vector<std::complex<double> > darkin(fftlen, 0.0), darkout(fftlen);
            int bginplen = std::min(bglength, fftlen);
            double normalize = 0.0;
            //Twists background not to be affected by the dc subtraction.
            for(int i = 0; i < bginplen; i++) {
                double tw = sin(2.0*M_PI*i/(double)bginplen);
                darkin[i] = dsowave[pos + i + bgpos] * tw;
                normalize += tw * tw;
            }
            normalize = 1.0 / normalize * interval;
            tr[ *this].m_ftDark->exec(darkin, darkout);
            //Convolution for the rectangular window.
            for(int i = 0; i < fftlen; i++) {
                darkin[i] = std::norm(darkout[i]) * normalize;
            }
            tr[ *this].m_ftDark->exec(darkin, darkout); //FT of PSD.
            std::vector<std::complex<double> > sigma2(darkout);
            std::fill(darkin.begin(), darkin.end(), std::complex<double>(0.0));
            double x = sqrt(1.0 / length / fftlen);
            for(int i = 0; i < length; i++) {
                darkin[i] = x;
            }
            tr[ *this].m_ftDark->exec(darkin, darkout); //FT of rect. window.
            for(int i = 0; i < fftlen; i++) {
                darkin[i] = std::norm(darkout[i]);
            }
            tr[ *this].m_ftDark->exec(darkin, darkout); //FT of norm of (FT of rect. window).
            for(int i = 0; i < fftlen; i++) {
                darkin[i] = std::conj(darkout[i] * sigma2[i]);
            }
            tr[ *this].m_ftDark->exec(darkin, darkout); //Convolution.
            normalize = 1.0 / fftlen;
            for(int i = 0; i < fftlen; i++) {
                darkpsdsum[i] += std::real(darkout[i]) * normalize; //[V^2/Hz]
            }
        }
        tr[ *this].m_avcount++;
        if( shot_this[ *exAvgIncr()]) {
            tr[ *extraAvg()] = shot_this[ *this].m_avcount;
        }
    }
    std::complex<double> *wave( &tr[ *this].m_wave[0]);
    double normalize = 1.0 / shot_this[ *this].m_avcount;
    for(int i = 0; i < length; i++) {
        wave[i] = wavesum[i] * normalize;
        //multi-echo T2
        for(int j=0; j < numechoes_pulse; j++){
            echoesT2[j][i] = echoesT2Sum[j][i] * normalize;
        }
    }
    double darknormalize = normalize * normalize;
    if(bg_off_echotrain)
        darknormalize /= (double)numechoes;
    double *darkpsd( &tr[ *this].m_darkPSD[0]);
    for(int i = 0; i < fftlen; i++) {
        darkpsd[i] = darkpsdsum[i] * darknormalize;
    }
    int ftpos = lrint( shot_this[ *fftPos()] * 1e-3 / interval + shot_dso[ *dso__].trigPos() - pos);

    if(shot_this[ *difFreq()] != 0.0) {
        //Digital IF.
        double omega = -2.0 * M_PI * shot_this[ *difFreq()] * 1e3 * interval;
        for(int i = 0; i < length; i++) {
            wave[i] *= std::polar(1.0, omega * (i - ftpos));
        }
    }

    //	if((windowfunc != &windowFuncRect) && (abs(ftpos - length/2) > length*0.1))
    //		m_statusPrinter->printWarning(i18n("FFTPos is off-centered for window func."));
    double ph = shot_this[ *phaseAdv()] * M_PI / 180;
    tr[ *this].m_waveFTPos = ftpos;
    //[Hz]
    tr[ *this].m_dFreq = 1.0 / fftlen / interval;
    tr[ *this].m_ftWave.resize(fftlen);

    rotNFFT(tr, ftpos, ph, tr[ *this].m_wave, tr[ *this].m_ftWave); //Generates a new SpectrumSolver.
    const SpectrumSolver &solver(shot_this[ *m_solver].solver());
    if(solver.peaks().size()) {
        entryPeakAbs()->value(tr,
            solver.peaks()[0].first / (double)shot_this[ *this].m_wave.size());
        double x = solver.peaks()[0].second;
        x = (x > fftlen / 2) ? (x - fftlen) : x;
        entryPeakFreq()->value(tr,
            0.001 * x * shot_this[ *this].m_dFreq);
    }

    m_isPulseInversionRequested = picenabled && (shot_this[ *this].m_avcount % 2 == 1) && (emitter == dso__.get());

    if( !shot_this[ *exAvgIncr()] && (avgnum != shot_this[ *this].m_avcount))
        throw XSkippedRecordError(__FILE__, __LINE__);
}
void XODMRFMControl::visualize(const Snapshot &shot) {
    iterate_commit_while([=](Transaction &tr)->bool{
        Snapshot &shot(tr);
        if(shot[ *this].time() && shot[ *this].m_avcount)
            return false;

        tr[ *ftWaveGraph()].clearPoints();
        tr[ *waveGraph()].clearPoints();
        tr[ *m_peakPlot->maxCount()] = 0;
        return true;
    });

    if(m_isPulseInversionRequested.compare_set_strong((int)true, (int)false)) {
        shared_ptr<XPulser> pulse__ = shot[ *pulser()];
        if(pulse__) {
            pulse__->iterate_commit([=](Transaction &tr){
                if(tr[ *pulse__].time()) {
                    tr[ *pulse__->invertPhase()] = !tr[ *pulse__->invertPhase()];
                }
            });
        }
    }

    iterate_commit([=](Transaction &tr){
        Snapshot &shot(tr);
        const SpectrumSolver &solver(shot[ *m_solver].solver());

        int ftsize = shot[ *this].m_ftWave.size();

        tr[ *ftWaveGraph()].setRowCount(ftsize);
        double normalize = 1.0 / shot[ *this].m_wave.size();
        double darknormalize = shot[ *this].darkPSDFactorToVoltSq();
        double dfreq = shot[ *this].m_dFreq;
        const double *darkpsd( &shot[ *this].m_darkPSD[0]);
        const std::complex<double> *ftwave( &shot[ *this].m_ftWave[0]);
        std::vector<double> colf(ftsize);
        std::vector<float> colr(ftsize), coli(ftsize), colarg(ftsize),
            colabs(ftsize), coldark(ftsize);
        for (int i = 0; i < ftsize; i++) {
            int j = (i - ftsize/2 + ftsize) % ftsize;
            colf[i] = 0.001 * (i - ftsize/2) * dfreq;
            std::complex<double> z = ftwave[j] * normalize;
            colr[i] = std::real(z);
            coli[i] = std::imag(z);
            colabs[i] = std::abs(z);
            colarg[i] = std::arg(z) / M_PI * 180;
            coldark[i] = sqrt(darkpsd[j] * darknormalize);
        }
        tr[ *ftWaveGraph()].setColumn(0, std::move(colf), 8);
        tr[ *ftWaveGraph()].setColumn(1, std::move(colr), 5);
        tr[ *ftWaveGraph()].setColumn(2, std::move(coli), 5);
        tr[ *ftWaveGraph()].setColumn(3, std::move(colabs), 5);
        tr[ *ftWaveGraph()].setColumn(4, std::move(colarg), 4);
        tr[ *ftWaveGraph()].setColumn(5, std::move(coldark), 4);
        const std::vector<std::pair<double, double> > &peaks(solver.peaks());
        int peaks_size = peaks.size();
        tr[ *m_peakPlot->maxCount()] = peaks_size;
        auto &points(tr[ *m_peakPlot].points());
        points.resize(peaks_size);
        for(int i = 0; i < peaks_size; i++) {
            double x = peaks[i].second;
            x = (x > ftsize / 2) ? (x - ftsize) : x;
            points[i] = XGraph::ValPoint(0.001 * x * dfreq, peaks[i].first * normalize);
        }
        ftWaveGraph()->drawGraph(tr);

        int length = shot[ *this].m_dsoWave.size();
        const std::complex<double> *dsowave( &shot[ *this].m_dsoWave[0]);
        if(solver.ifft().size() < ftsize)
            return; //solver has been failed.
        const std::complex<double> *ifft( &solver.ifft()[0]);
        int dsowavestartpos = shot[ *this].m_dsoWaveStartPos;
        double interval = shot[ *this].m_interval;
        double starttime = shot[ *this].startTime();
        int waveftpos = shot[ *this].m_waveFTPos;
        tr[ *waveGraph()].setRowCount(length);
        std::vector<float> colt(length);
        std::vector<float> colfr(length), colfi(length),
            colrr(length), colri(length);
        for (int i = 0; i < length; i++) {
            int j = i - dsowavestartpos;
            colt[i] = (starttime + j * interval) * 1e3;
            if(abs(j) < ftsize / 2) {
                j = (j - waveftpos + ftsize) % ftsize;
                colfr[i] = std::real(ifft[j]);
                colfi[i] = std::imag(ifft[j]);
            }
            else {
                colfr[i] = 0.0;
                colfi[i] = 0.0;
            }
            colrr[i] = dsowave[i].real();
            colri[i] = dsowave[i].imag();
        }
        tr[ *waveGraph()].setColumn(0, std::move(colt), 7);
        tr[ *waveGraph()].setColumn(1, std::move(colfr), 5);
        tr[ *waveGraph()].setColumn(2, std::move(colfi), 5);
        tr[ *waveGraph()].setColumn(3, std::move(colrr), 4);
        tr[ *waveGraph()].setColumn(4, std::move(colri), 4);
        waveGraph()->drawGraph(tr);
    });
}

