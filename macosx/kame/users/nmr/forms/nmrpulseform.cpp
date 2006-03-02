#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../../../kame/users/nmr/forms/nmrpulseform.ui'
**
** Created: 木  3 2 16:40:49 2006
**      by: The User Interface Compiler ($Id: nmrpulseform.cpp,v 1.1.2.1 2006/03/02 09:19:17 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "nmrpulseform.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qgroupbox.h>
#include <qlabel.h>
#include <qlineedit.h>
#include <qspinbox.h>
#include <qcheckbox.h>
#include <qcombobox.h>
#include <knuminput.h>
#include <kurlrequester.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include <qaction.h>
#include <qmenubar.h>
#include <qpopupmenu.h>
#include <qtoolbar.h>
#include "../../graph/graphwidget.h"

/*
 *  Constructs a FrmNMRPulse as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
FrmNMRPulse::FrmNMRPulse( QWidget* parent, const char* name, WFlags fl )
    : QMainWindow( parent, name, fl )
{
    (void)statusBar();
    if ( !name )
	setName( "FrmNMRPulse" );
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );
    FrmNMRPulseLayout = new QGridLayout( centralWidget(), 1, 1, 2, 6, "FrmNMRPulseLayout"); 

    groupBox1_2 = new QGroupBox( centralWidget(), "groupBox1_2" );
    groupBox1_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, groupBox1_2->sizePolicy().hasHeightForWidth() ) );
    groupBox1_2->setColumnLayout(0, Qt::Vertical );
    groupBox1_2->layout()->setSpacing( 6 );
    groupBox1_2->layout()->setMargin( 11 );
    groupBox1_2Layout = new QGridLayout( groupBox1_2->layout() );
    groupBox1_2Layout->setAlignment( Qt::AlignTop );

    layout23 = new QHBoxLayout( 0, 0, 6, "layout23"); 

    textLabel1_5_2_2 = new QLabel( groupBox1_2, "textLabel1_5_2_2" );
    textLabel1_5_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_5_2_2->sizePolicy().hasHeightForWidth() ) );
    layout23->addWidget( textLabel1_5_2_2 );

    layout1_2_3 = new QHBoxLayout( 0, 0, 6, "layout1_2_3"); 

    m_edEchoPeriod = new QLineEdit( groupBox1_2, "m_edEchoPeriod" );
    m_edEchoPeriod->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edEchoPeriod->sizePolicy().hasHeightForWidth() ) );
    m_edEchoPeriod->setMaximumSize( QSize( 80, 32767 ) );
    layout1_2_3->addWidget( m_edEchoPeriod );

    textLabel2_2_3 = new QLabel( groupBox1_2, "textLabel2_2_3" );
    textLabel2_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_2_3->sizePolicy().hasHeightForWidth() ) );
    layout1_2_3->addWidget( textLabel2_2_3 );
    layout23->addLayout( layout1_2_3 );

    groupBox1_2Layout->addLayout( layout23, 1, 0 );

    layout24 = new QHBoxLayout( 0, 0, 6, "layout24"); 

    textLabel1_5_2_2_2 = new QLabel( groupBox1_2, "textLabel1_5_2_2_2" );
    textLabel1_5_2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_5_2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout24->addWidget( textLabel1_5_2_2_2 );

    m_numEcho = new QSpinBox( groupBox1_2, "m_numEcho" );
    m_numEcho->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)1, (QSizePolicy::SizeType)0, 0, 0, m_numEcho->sizePolicy().hasHeightForWidth() ) );
    m_numEcho->setMinValue( 1 );
    m_numEcho->setValue( 1 );
    layout24->addWidget( m_numEcho );

    groupBox1_2Layout->addLayout( layout24, 0, 0 );

    FrmNMRPulseLayout->addWidget( groupBox1_2, 8, 0 );

    layout21 = new QHBoxLayout( 0, 0, 6, "layout21"); 

    layout2_3_2 = new QVBoxLayout( 0, 0, 6, "layout2_3_2"); 

    textLabel1_5_2 = new QLabel( centralWidget(), "textLabel1_5_2" );
    textLabel1_5_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_5_2->sizePolicy().hasHeightForWidth() ) );
    layout2_3_2->addWidget( textLabel1_5_2 );

    layout1_3_2 = new QHBoxLayout( 0, 0, 6, "layout1_3_2"); 

    m_edFFTPos = new QLineEdit( centralWidget(), "m_edFFTPos" );
    m_edFFTPos->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edFFTPos->sizePolicy().hasHeightForWidth() ) );
    m_edFFTPos->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2->addWidget( m_edFFTPos );

    textLabel2_3_2 = new QLabel( centralWidget(), "textLabel2_3_2" );
    textLabel2_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2->addWidget( textLabel2_3_2 );
    layout2_3_2->addLayout( layout1_3_2 );
    layout21->addLayout( layout2_3_2 );

    layout2_4 = new QVBoxLayout( 0, 0, 6, "layout2_4"); 

    textLabel1_4 = new QLabel( centralWidget(), "textLabel1_4" );
    textLabel1_4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_4->sizePolicy().hasHeightForWidth() ) );
    layout2_4->addWidget( textLabel1_4 );

    layout1_4 = new QHBoxLayout( 0, 0, 6, "layout1_4"); 

    m_edFFTLen = new QLineEdit( centralWidget(), "m_edFFTLen" );
    m_edFFTLen->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edFFTLen->sizePolicy().hasHeightForWidth() ) );
    m_edFFTLen->setMaximumSize( QSize( 80, 32767 ) );
    layout1_4->addWidget( m_edFFTLen );

    textLabel2_4 = new QLabel( centralWidget(), "textLabel2_4" );
    textLabel2_4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_4->sizePolicy().hasHeightForWidth() ) );
    layout1_4->addWidget( textLabel2_4 );
    layout2_4->addLayout( layout1_4 );
    layout21->addLayout( layout2_4 );

    FrmNMRPulseLayout->addLayout( layout21, 4, 0 );

    layout23_2 = new QHBoxLayout( 0, 0, 6, "layout23_2"); 

    textLabel1_5_2_2_3 = new QLabel( centralWidget(), "textLabel1_5_2_2_3" );
    textLabel1_5_2_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_5_2_2_3->sizePolicy().hasHeightForWidth() ) );
    layout23_2->addWidget( textLabel1_5_2_2_3 );

    layout1_2_3_2 = new QHBoxLayout( 0, 0, 6, "layout1_2_3_2"); 

    m_edDIFFreq = new QLineEdit( centralWidget(), "m_edDIFFreq" );
    m_edDIFFreq->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edDIFFreq->sizePolicy().hasHeightForWidth() ) );
    m_edDIFFreq->setMaximumSize( QSize( 80, 32767 ) );
    layout1_2_3_2->addWidget( m_edDIFFreq );

    textLabel2_2_3_2 = new QLabel( centralWidget(), "textLabel2_2_3_2" );
    textLabel2_2_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_2_3_2->sizePolicy().hasHeightForWidth() ) );
    layout1_2_3_2->addWidget( textLabel2_2_3_2 );
    layout23_2->addLayout( layout1_2_3_2 );

    FrmNMRPulseLayout->addLayout( layout23_2, 6, 0 );

    layout23_3 = new QVBoxLayout( 0, 0, 6, "layout23_3"); 

    m_ckbDNR = new QCheckBox( centralWidget(), "m_ckbDNR" );
    m_ckbDNR->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_ckbDNR->sizePolicy().hasHeightForWidth() ) );
    layout23_3->addWidget( m_ckbDNR );

    layout18 = new QHBoxLayout( 0, 0, 6, "layout18"); 

    layout2_3 = new QVBoxLayout( 0, 0, 6, "layout2_3"); 

    textLabel1_5 = new QLabel( centralWidget(), "textLabel1_5" );
    textLabel1_5->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_5->sizePolicy().hasHeightForWidth() ) );
    layout2_3->addWidget( textLabel1_5 );

    layout1_3 = new QHBoxLayout( 0, 0, 6, "layout1_3"); 

    m_edBGPos = new QLineEdit( centralWidget(), "m_edBGPos" );
    m_edBGPos->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edBGPos->sizePolicy().hasHeightForWidth() ) );
    m_edBGPos->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3->addWidget( m_edBGPos );

    textLabel2_3 = new QLabel( centralWidget(), "textLabel2_3" );
    textLabel2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3->sizePolicy().hasHeightForWidth() ) );
    layout1_3->addWidget( textLabel2_3 );
    layout2_3->addLayout( layout1_3 );
    layout18->addLayout( layout2_3 );

    layout2_2_2 = new QVBoxLayout( 0, 0, 6, "layout2_2_2"); 

    textLabel1_2_2 = new QLabel( centralWidget(), "textLabel1_2_2" );
    textLabel1_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_2_2->sizePolicy().hasHeightForWidth() ) );
    layout2_2_2->addWidget( textLabel1_2_2 );

    layout1_2_2 = new QHBoxLayout( 0, 0, 6, "layout1_2_2"); 

    m_edBGWidth = new QLineEdit( centralWidget(), "m_edBGWidth" );
    m_edBGWidth->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edBGWidth->sizePolicy().hasHeightForWidth() ) );
    m_edBGWidth->setMaximumSize( QSize( 80, 32767 ) );
    layout1_2_2->addWidget( m_edBGWidth );

    textLabel2_2_2 = new QLabel( centralWidget(), "textLabel2_2_2" );
    textLabel2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout1_2_2->addWidget( textLabel2_2_2 );
    layout2_2_2->addLayout( layout1_2_2 );
    layout18->addLayout( layout2_2_2 );
    layout23_3->addLayout( layout18 );

    FrmNMRPulseLayout->addLayout( layout23_3, 3, 0 );

    m_btnFFT = new QPushButton( centralWidget(), "m_btnFFT" );
    m_btnFFT->setAutoDefault( FALSE );

    FrmNMRPulseLayout->addWidget( m_btnFFT, 7, 0 );

    layout93 = new QHBoxLayout( 0, 0, 6, "layout93"); 

    textLabel1_3 = new QLabel( centralWidget(), "textLabel1_3" );
    textLabel1_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_3->sizePolicy().hasHeightForWidth() ) );
    layout93->addWidget( textLabel1_3 );

    m_cmbWindowFunc = new QComboBox( FALSE, centralWidget(), "m_cmbWindowFunc" );
    layout93->addWidget( m_cmbWindowFunc );

    FrmNMRPulseLayout->addLayout( layout93, 5, 0 );

    layout9 = new QHBoxLayout( 0, 0, 6, "layout9"); 

    textLabel4 = new QLabel( centralWidget(), "textLabel4" );
    textLabel4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel4->sizePolicy().hasHeightForWidth() ) );
    layout9->addWidget( textLabel4 );

    m_numPhaseAdv = new KDoubleNumInput( centralWidget(), "m_numPhaseAdv" );
    m_numPhaseAdv->setMinValue( 0 );
    m_numPhaseAdv->setMaxValue( 999900 );
    m_numPhaseAdv->setPrecision( 1 );
    layout9->addWidget( m_numPhaseAdv );

    FrmNMRPulseLayout->addLayout( layout9, 2, 0 );

    layout20 = new QHBoxLayout( 0, 0, 6, "layout20"); 

    layout2 = new QVBoxLayout( 0, 0, 6, "layout2"); 

    textLabel1 = new QLabel( centralWidget(), "textLabel1" );
    textLabel1->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1->sizePolicy().hasHeightForWidth() ) );
    layout2->addWidget( textLabel1 );

    layout1 = new QHBoxLayout( 0, 0, 6, "layout1"); 

    m_edPos = new QLineEdit( centralWidget(), "m_edPos" );
    m_edPos->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edPos->sizePolicy().hasHeightForWidth() ) );
    m_edPos->setMaximumSize( QSize( 80, 32767 ) );
    layout1->addWidget( m_edPos );

    textLabel2 = new QLabel( centralWidget(), "textLabel2" );
    textLabel2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2->sizePolicy().hasHeightForWidth() ) );
    layout1->addWidget( textLabel2 );
    layout2->addLayout( layout1 );
    layout20->addLayout( layout2 );

    layout2_2 = new QVBoxLayout( 0, 0, 6, "layout2_2"); 

    textLabel1_2 = new QLabel( centralWidget(), "textLabel1_2" );
    textLabel1_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_2->sizePolicy().hasHeightForWidth() ) );
    layout2_2->addWidget( textLabel1_2 );

    layout1_2 = new QHBoxLayout( 0, 0, 6, "layout1_2"); 

    m_edWidth = new QLineEdit( centralWidget(), "m_edWidth" );
    m_edWidth->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edWidth->sizePolicy().hasHeightForWidth() ) );
    m_edWidth->setMaximumSize( QSize( 80, 32767 ) );
    layout1_2->addWidget( m_edWidth );

    textLabel2_2 = new QLabel( centralWidget(), "textLabel2_2" );
    textLabel2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_2->sizePolicy().hasHeightForWidth() ) );
    layout1_2->addWidget( textLabel2_2 );
    layout2_2->addLayout( layout1_2 );
    layout20->addLayout( layout2_2 );

    FrmNMRPulseLayout->addLayout( layout20, 1, 0 );

    layout26 = new QHBoxLayout( 0, 0, 6, "layout26"); 

    textLabel1_2_3 = new QLabel( centralWidget(), "textLabel1_2_3" );
    textLabel1_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, textLabel1_2_3->sizePolicy().hasHeightForWidth() ) );
    layout26->addWidget( textLabel1_2_3 );

    m_cmbDSO = new QComboBox( FALSE, centralWidget(), "m_cmbDSO" );
    layout26->addWidget( m_cmbDSO );

    FrmNMRPulseLayout->addLayout( layout26, 0, 0 );

    layout30 = new QVBoxLayout( 0, 0, 6, "layout30"); 

    layout15 = new QHBoxLayout( 0, 0, 6, "layout15"); 

    m_urlDump = new KURLRequester( centralWidget(), "m_urlDump" );
    m_urlDump->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)7, (QSizePolicy::SizeType)5, 0, 0, m_urlDump->sizePolicy().hasHeightForWidth() ) );
    m_urlDump->setCursor( QCursor( 0 ) );
    m_urlDump->setFocusPolicy( KURLRequester::StrongFocus );
    layout15->addWidget( m_urlDump );

    m_btnDump = new QPushButton( centralWidget(), "m_btnDump" );
    m_btnDump->setAutoDefault( FALSE );
    layout15->addWidget( m_btnDump );
    layout30->addLayout( layout15 );

    layout29 = new QVBoxLayout( 0, 0, 6, "layout29"); 

    m_graph = new XQGraph( centralWidget(), "m_graph" );
    m_graph->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)7, (QSizePolicy::SizeType)7, 0, 0, m_graph->sizePolicy().hasHeightForWidth() ) );
    layout29->addWidget( m_graph );

    groupBox1 = new QGroupBox( centralWidget(), "groupBox1" );
    groupBox1->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, groupBox1->sizePolicy().hasHeightForWidth() ) );
    groupBox1->setColumnLayout(0, Qt::Vertical );
    groupBox1->layout()->setSpacing( 6 );
    groupBox1->layout()->setMargin( 11 );
    groupBox1Layout = new QHBoxLayout( groupBox1->layout() );
    groupBox1Layout->setAlignment( Qt::AlignTop );

    m_ckbIncrAvg = new QCheckBox( groupBox1, "m_ckbIncrAvg" );
    m_ckbIncrAvg->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckbIncrAvg->sizePolicy().hasHeightForWidth() ) );
    groupBox1Layout->addWidget( m_ckbIncrAvg );

    m_numExtraAvg = new QSpinBox( groupBox1, "m_numExtraAvg" );
    m_numExtraAvg->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)1, (QSizePolicy::SizeType)0, 0, 0, m_numExtraAvg->sizePolicy().hasHeightForWidth() ) );
    groupBox1Layout->addWidget( m_numExtraAvg );

    m_btnAvgClear = new QPushButton( groupBox1, "m_btnAvgClear" );
    m_btnAvgClear->setAutoDefault( FALSE );
    groupBox1Layout->addWidget( m_btnAvgClear );
    layout29->addWidget( groupBox1 );
    layout30->addLayout( layout29 );

    FrmNMRPulseLayout->addMultiCellLayout( layout30, 0, 8, 1, 1 );

    // toolbars

    languageChange();
    resize( QSize(634, 479).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // tab order
    setTabOrder( m_cmbDSO, m_edPos );
    setTabOrder( m_edPos, m_edWidth );
    setTabOrder( m_edWidth, m_numPhaseAdv );
    setTabOrder( m_numPhaseAdv, m_ckbDNR );
    setTabOrder( m_ckbDNR, m_edBGPos );
    setTabOrder( m_edBGPos, m_edBGWidth );
    setTabOrder( m_edBGWidth, m_edFFTPos );
    setTabOrder( m_edFFTPos, m_edFFTLen );
    setTabOrder( m_edFFTLen, m_cmbWindowFunc );
    setTabOrder( m_cmbWindowFunc, m_edDIFFreq );
    setTabOrder( m_edDIFFreq, m_btnFFT );
    setTabOrder( m_btnFFT, m_numEcho );
    setTabOrder( m_numEcho, m_edEchoPeriod );
    setTabOrder( m_edEchoPeriod, m_ckbIncrAvg );
    setTabOrder( m_ckbIncrAvg, m_numExtraAvg );
    setTabOrder( m_numExtraAvg, m_btnAvgClear );
    setTabOrder( m_btnAvgClear, m_urlDump );
    setTabOrder( m_urlDump, m_btnDump );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmNMRPulse::~FrmNMRPulse()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmNMRPulse::languageChange()
{
    setCaption( tr2i18n( "NMR Pulse Settings" ) );
    groupBox1_2->setTitle( tr2i18n( "Multiple Echoes" ) );
    textLabel1_5_2_2->setText( tr2i18n( "Period (=2Tau)" ) );
    textLabel2_2_3->setText( tr2i18n( "ms" ) );
    textLabel1_5_2_2_2->setText( tr2i18n( "# of Echoes" ) );
    textLabel1_5_2->setText( tr2i18n( "FFT Origin" ) );
    textLabel2_3_2->setText( tr2i18n( "ms" ) );
    textLabel1_4->setText( tr2i18n( "FFT Length" ) );
    textLabel2_4->setText( tr2i18n( "pts" ) );
    textLabel1_5_2_2_3->setText( tr2i18n( "Digital IF Freq" ) );
    textLabel2_2_3_2->setText( tr2i18n( "kHz" ) );
    m_ckbDNR->setText( tr2i18n( "Dynamic Noise Reduction (DNR)" ) );
    textLabel1_5->setText( tr2i18n( "Pos for Sub." ) );
    textLabel2_3->setText( tr2i18n( "ms" ) );
    textLabel1_2_2->setText( tr2i18n( "Width" ) );
    textLabel2_2_2->setText( tr2i18n( "ms" ) );
    m_btnFFT->setText( tr2i18n( "SHOW FFT" ) );
    textLabel1_3->setText( tr2i18n( "Window Func." ) );
    textLabel4->setText( tr2i18n( "Phase" ) );
    m_numPhaseAdv->setSuffix( tr2i18n( "deg" ) );
    textLabel1->setText( tr2i18n( "Origin from Trig" ) );
    textLabel2->setText( tr2i18n( "ms" ) );
    textLabel1_2->setText( tr2i18n( "Width" ) );
    textLabel2_2->setText( tr2i18n( "ms" ) );
    textLabel1_2_3->setText( tr2i18n( "DSO" ) );
    m_btnDump->setText( tr2i18n( "DUMP" ) );
    groupBox1->setTitle( tr2i18n( "Incremental/Moving Average" ) );
    m_ckbIncrAvg->setText( tr2i18n( "Incremental" ) );
    m_btnAvgClear->setText( tr2i18n( "CLEAR" ) );
}

#include "nmrpulseform.moc"
