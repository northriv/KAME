#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../../../kame/users/nmr/forms/nmrrelaxform.ui'
**
** Created: 木  3 2 16:40:59 2006
**      by: The User Interface Compiler ($Id: nmrrelaxform.cpp,v 1.1.2.1 2006/03/02 09:19:16 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "nmrrelaxform.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <kurlrequester.h>
#include <qgroupbox.h>
#include <qcheckbox.h>
#include <qlabel.h>
#include <qlineedit.h>
#include <qcombobox.h>
#include <qspinbox.h>
#include <knuminput.h>
#include <qtextbrowser.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include <qaction.h>
#include <qmenubar.h>
#include <qpopupmenu.h>
#include <qtoolbar.h>
#include "../../graph/graphwidget.h"

/*
 *  Constructs a FrmNMRT1 as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
FrmNMRT1::FrmNMRT1( QWidget* parent, const char* name, WFlags fl )
    : QMainWindow( parent, name, fl )
{
    (void)statusBar();
    if ( !name )
	setName( "FrmNMRT1" );
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );
    FrmNMRT1Layout = new QGridLayout( centralWidget(), 1, 1, 2, 6, "FrmNMRT1Layout"); 

    layout19 = new QVBoxLayout( 0, 0, 6, "layout19"); 

    m_graph = new XQGraph( centralWidget(), "m_graph" );
    m_graph->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)7, (QSizePolicy::SizeType)7, 0, 0, m_graph->sizePolicy().hasHeightForWidth() ) );
    layout19->addWidget( m_graph );

    layout15 = new QHBoxLayout( 0, 0, 6, "layout15"); 

    m_urlDump = new KURLRequester( centralWidget(), "m_urlDump" );
    m_urlDump->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)7, (QSizePolicy::SizeType)5, 0, 0, m_urlDump->sizePolicy().hasHeightForWidth() ) );
    m_urlDump->setCursor( QCursor( 0 ) );
    m_urlDump->setFocusPolicy( KURLRequester::StrongFocus );
    layout15->addWidget( m_urlDump );

    m_btnDump = new QPushButton( centralWidget(), "m_btnDump" );
    m_btnDump->setAutoDefault( FALSE );
    layout15->addWidget( m_btnDump );
    layout19->addLayout( layout15 );

    FrmNMRT1Layout->addMultiCellLayout( layout19, 1, 2, 1, 1 );

    groupBox4 = new QGroupBox( centralWidget(), "groupBox4" );
    groupBox4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)5, 0, 0, groupBox4->sizePolicy().hasHeightForWidth() ) );
    groupBox4->setMargin( 0 );
    groupBox4->setMidLineWidth( 0 );
    groupBox4->setAlignment( int( QGroupBox::AlignTop ) );
    groupBox4->setColumnLayout(0, Qt::Vertical );
    groupBox4->layout()->setSpacing( 1 );
    groupBox4->layout()->setMargin( 4 );
    groupBox4Layout = new QGridLayout( groupBox4->layout() );
    groupBox4Layout->setAlignment( Qt::AlignTop );

    m_ckbActive = new QCheckBox( groupBox4, "m_ckbActive" );
    m_ckbActive->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckbActive->sizePolicy().hasHeightForWidth() ) );

    groupBox4Layout->addWidget( m_ckbActive, 0, 0 );

    layout28 = new QVBoxLayout( 0, 0, 6, "layout28"); 

    textLabel1_2 = new QLabel( groupBox4, "textLabel1_2" );
    textLabel1_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)5, 0, 0, textLabel1_2->sizePolicy().hasHeightForWidth() ) );
    textLabel1_2->setAlignment( int( QLabel::WordBreak | QLabel::AlignCenter ) );
    layout28->addWidget( textLabel1_2 );

    layout13 = new QHBoxLayout( 0, 0, 6, "layout13"); 

    layout9 = new QHBoxLayout( 0, 0, 6, "layout9"); 

    textLabel2_2_3 = new QLabel( groupBox4, "textLabel2_2_3" );
    textLabel2_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_2_3->sizePolicy().hasHeightForWidth() ) );
    layout9->addWidget( textLabel2_2_3 );

    m_edP1Min = new QLineEdit( groupBox4, "m_edP1Min" );
    m_edP1Min->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edP1Min->sizePolicy().hasHeightForWidth() ) );
    m_edP1Min->setMinimumSize( QSize( 50, 0 ) );
    m_edP1Min->setMaximumSize( QSize( 80, 32767 ) );
    layout9->addWidget( m_edP1Min );
    layout13->addLayout( layout9 );

    layout10 = new QHBoxLayout( 0, 0, 6, "layout10"); 

    textLabel2_2_2 = new QLabel( groupBox4, "textLabel2_2_2" );
    textLabel2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout10->addWidget( textLabel2_2_2 );

    m_edP1Max = new QLineEdit( groupBox4, "m_edP1Max" );
    m_edP1Max->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edP1Max->sizePolicy().hasHeightForWidth() ) );
    m_edP1Max->setMinimumSize( QSize( 50, 0 ) );
    m_edP1Max->setMaximumSize( QSize( 80, 32767 ) );
    layout10->addWidget( m_edP1Max );
    layout13->addLayout( layout10 );
    layout28->addLayout( layout13 );

    groupBox4Layout->addMultiCellLayout( layout28, 0, 1, 1, 1 );

    m_ckbT2Mode = new QCheckBox( groupBox4, "m_ckbT2Mode" );
    m_ckbT2Mode->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckbT2Mode->sizePolicy().hasHeightForWidth() ) );

    groupBox4Layout->addWidget( m_ckbT2Mode, 0, 2 );

    layout15_2 = new QHBoxLayout( 0, 0, 6, "layout15_2"); 

    textLabel1 = new QLabel( groupBox4, "textLabel1" );
    textLabel1->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1->sizePolicy().hasHeightForWidth() ) );
    layout15_2->addWidget( textLabel1 );

    m_cmbP1Dist = new QComboBox( FALSE, groupBox4, "m_cmbP1Dist" );
    m_cmbP1Dist->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)1, (QSizePolicy::SizeType)0, 0, 0, m_cmbP1Dist->sizePolicy().hasHeightForWidth() ) );
    layout15_2->addWidget( m_cmbP1Dist );

    groupBox4Layout->addLayout( layout15_2, 1, 2 );

    layout21 = new QHBoxLayout( 0, 0, 6, "layout21"); 

    textLabel5_2_3_3_2 = new QLabel( groupBox4, "textLabel5_2_3_3_2" );
    textLabel5_2_3_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_3_2->sizePolicy().hasHeightForWidth() ) );
    textLabel5_2_3_3_2->setAlignment( int( QLabel::AlignCenter ) );
    layout21->addWidget( textLabel5_2_3_3_2 );

    m_numExtraAvg = new QSpinBox( groupBox4, "m_numExtraAvg" );
    m_numExtraAvg->setMaxValue( 256 );
    m_numExtraAvg->setMinValue( 1 );
    layout21->addWidget( m_numExtraAvg );

    groupBox4Layout->addLayout( layout21, 0, 3 );

    layout22 = new QHBoxLayout( 0, 0, 6, "layout22"); 

    textLabel5_2_3_3_2_2 = new QLabel( groupBox4, "textLabel5_2_3_3_2_2" );
    textLabel5_2_3_3_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_3_2_2->sizePolicy().hasHeightForWidth() ) );
    textLabel5_2_3_3_2_2->setAlignment( int( QLabel::AlignCenter ) );
    layout22->addWidget( textLabel5_2_3_3_2_2 );

    m_numIgnore = new QSpinBox( groupBox4, "m_numIgnore" );
    layout22->addWidget( m_numIgnore );

    groupBox4Layout->addLayout( layout22, 1, 3 );

    FrmNMRT1Layout->addMultiCellWidget( groupBox4, 0, 0, 0, 1 );

    layout23 = new QVBoxLayout( 0, 0, 6, "layout23"); 

    layout18_2_2 = new QHBoxLayout( 0, 0, 6, "layout18_2_2"); 

    textLabel1_3_2_2 = new QLabel( centralWidget(), "textLabel1_3_2_2" );
    textLabel1_3_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_3_2_2->sizePolicy().hasHeightForWidth() ) );
    layout18_2_2->addWidget( textLabel1_3_2_2 );

    m_cmbPulse1 = new QComboBox( FALSE, centralWidget(), "m_cmbPulse1" );
    m_cmbPulse1->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)1, (QSizePolicy::SizeType)0, 0, 0, m_cmbPulse1->sizePolicy().hasHeightForWidth() ) );
    layout18_2_2->addWidget( m_cmbPulse1 );
    layout23->addLayout( layout18_2_2 );

    layout18_2 = new QHBoxLayout( 0, 0, 6, "layout18_2"); 

    textLabel1_3_2 = new QLabel( centralWidget(), "textLabel1_3_2" );
    textLabel1_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_3_2->sizePolicy().hasHeightForWidth() ) );
    layout18_2->addWidget( textLabel1_3_2 );

    m_cmbPulse2 = new QComboBox( FALSE, centralWidget(), "m_cmbPulse2" );
    m_cmbPulse2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)1, (QSizePolicy::SizeType)0, 0, 0, m_cmbPulse2->sizePolicy().hasHeightForWidth() ) );
    layout18_2->addWidget( m_cmbPulse2 );
    layout23->addLayout( layout18_2 );

    layout18 = new QHBoxLayout( 0, 0, 6, "layout18"); 

    textLabel1_3 = new QLabel( centralWidget(), "textLabel1_3" );
    textLabel1_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_3->sizePolicy().hasHeightForWidth() ) );
    layout18->addWidget( textLabel1_3 );

    m_cmbPulser = new QComboBox( FALSE, centralWidget(), "m_cmbPulser" );
    m_cmbPulser->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)1, (QSizePolicy::SizeType)0, 0, 0, m_cmbPulser->sizePolicy().hasHeightForWidth() ) );
    layout18->addWidget( m_cmbPulser );
    layout23->addLayout( layout18 );

    FrmNMRT1Layout->addLayout( layout23, 1, 0 );

    groupBox5 = new QGroupBox( centralWidget(), "groupBox5" );
    groupBox5->setFrameShape( QGroupBox::GroupBoxPanel );
    groupBox5->setMargin( 0 );
    groupBox5->setColumnLayout(0, Qt::Vertical );
    groupBox5->layout()->setSpacing( 1 );
    groupBox5->layout()->setMargin( 4 );
    groupBox5Layout = new QGridLayout( groupBox5->layout() );
    groupBox5Layout->setAlignment( Qt::AlignTop );

    m_ckbAutoPhase = new QCheckBox( groupBox5, "m_ckbAutoPhase" );
    m_ckbAutoPhase->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckbAutoPhase->sizePolicy().hasHeightForWidth() ) );

    groupBox5Layout->addWidget( m_ckbAutoPhase, 0, 0 );

    m_ckbAbsFit = new QCheckBox( groupBox5, "m_ckbAbsFit" );
    m_ckbAbsFit->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckbAbsFit->sizePolicy().hasHeightForWidth() ) );

    groupBox5Layout->addWidget( m_ckbAbsFit, 0, 1 );

    textLabel4_2 = new QLabel( groupBox5, "textLabel4_2" );
    textLabel4_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel4_2->sizePolicy().hasHeightForWidth() ) );

    groupBox5Layout->addWidget( textLabel4_2, 6, 0 );

    layout25 = new QHBoxLayout( 0, 0, 6, "layout25"); 

    textLabel5 = new QLabel( groupBox5, "textLabel5" );
    textLabel5->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel5->sizePolicy().hasHeightForWidth() ) );
    layout25->addWidget( textLabel5 );

    m_cmbFunction = new QComboBox( FALSE, groupBox5, "m_cmbFunction" );
    layout25->addWidget( m_cmbFunction );

    groupBox5Layout->addMultiCellLayout( layout25, 5, 5, 0, 1 );

    layout61 = new QHBoxLayout( 0, 0, 6, "layout61"); 

    textLabel3_2 = new QLabel( groupBox5, "textLabel3_2" );
    textLabel3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel3_2->sizePolicy().hasHeightForWidth() ) );
    layout61->addWidget( textLabel3_2 );

    m_edSmoothSamples = new QLineEdit( groupBox5, "m_edSmoothSamples" );
    m_edSmoothSamples->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edSmoothSamples->sizePolicy().hasHeightForWidth() ) );
    m_edSmoothSamples->setMaximumSize( QSize( 80, 32767 ) );
    layout61->addWidget( m_edSmoothSamples );

    groupBox5Layout->addMultiCellLayout( layout61, 4, 4, 0, 1 );

    layout56_2 = new QHBoxLayout( 0, 0, 6, "layout56_2"); 

    textLabel3_3 = new QLabel( groupBox5, "textLabel3_3" );
    textLabel3_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel3_3->sizePolicy().hasHeightForWidth() ) );
    layout56_2->addWidget( textLabel3_3 );

    layout1_3_2 = new QHBoxLayout( 0, 0, 6, "layout1_3_2"); 

    m_edBW = new QLineEdit( groupBox5, "m_edBW" );
    m_edBW->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edBW->sizePolicy().hasHeightForWidth() ) );
    m_edBW->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2->addWidget( m_edBW );

    textLabel2_3_2_2 = new QLabel( groupBox5, "textLabel2_3_2_2" );
    textLabel2_3_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2->addWidget( textLabel2_3_2_2 );
    layout56_2->addLayout( layout1_3_2 );

    groupBox5Layout->addMultiCellLayout( layout56_2, 3, 3, 0, 1 );

    layout9_2 = new QHBoxLayout( 0, 0, 6, "layout9_2"); 

    textLabel4 = new QLabel( groupBox5, "textLabel4" );
    textLabel4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel4->sizePolicy().hasHeightForWidth() ) );
    layout9_2->addWidget( textLabel4 );

    m_numPhase = new KDoubleNumInput( groupBox5, "m_numPhase" );
    m_numPhase->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)1, (QSizePolicy::SizeType)0, 0, 0, m_numPhase->sizePolicy().hasHeightForWidth() ) );
    m_numPhase->setMinValue( -360 );
    m_numPhase->setMaxValue( 360 );
    layout9_2->addWidget( m_numPhase );

    groupBox5Layout->addMultiCellLayout( layout9_2, 1, 1, 0, 1 );

    layout56 = new QHBoxLayout( 0, 0, 6, "layout56"); 

    textLabel3 = new QLabel( groupBox5, "textLabel3" );
    textLabel3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel3->sizePolicy().hasHeightForWidth() ) );
    layout56->addWidget( textLabel3 );

    layout1_3 = new QHBoxLayout( 0, 0, 6, "layout1_3"); 

    m_edFreq = new QLineEdit( groupBox5, "m_edFreq" );
    m_edFreq->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edFreq->sizePolicy().hasHeightForWidth() ) );
    m_edFreq->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3->addWidget( m_edFreq );

    textLabel2_3_2 = new QLabel( groupBox5, "textLabel2_3_2" );
    textLabel2_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3->addWidget( textLabel2_3_2 );
    layout56->addLayout( layout1_3 );

    groupBox5Layout->addMultiCellLayout( layout56, 2, 2, 0, 1 );

    m_txtFitStatus = new QTextBrowser( groupBox5, "m_txtFitStatus" );
    m_txtFitStatus->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)7, (QSizePolicy::SizeType)3, 0, 0, m_txtFitStatus->sizePolicy().hasHeightForWidth() ) );
    m_txtFitStatus->setFrameShape( QTextBrowser::Box );
    m_txtFitStatus->setFrameShadow( QTextBrowser::Raised );
    m_txtFitStatus->setTextFormat( QTextBrowser::PlainText );

    groupBox5Layout->addMultiCellWidget( m_txtFitStatus, 7, 7, 0, 1 );

    m_btnClear = new QPushButton( groupBox5, "m_btnClear" );
    m_btnClear->setAutoDefault( FALSE );

    groupBox5Layout->addMultiCellWidget( m_btnClear, 9, 9, 0, 1 );

    m_btnResetFit = new QPushButton( groupBox5, "m_btnResetFit" );

    groupBox5Layout->addMultiCellWidget( m_btnResetFit, 8, 8, 0, 1 );

    FrmNMRT1Layout->addWidget( groupBox5, 2, 0 );

    // toolbars

    languageChange();
    resize( QSize(613, 604).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // tab order
    setTabOrder( m_ckbActive, m_edP1Min );
    setTabOrder( m_edP1Min, m_edP1Max );
    setTabOrder( m_edP1Max, m_ckbT2Mode );
    setTabOrder( m_ckbT2Mode, m_cmbP1Dist );
    setTabOrder( m_cmbP1Dist, m_numExtraAvg );
    setTabOrder( m_numExtraAvg, m_numIgnore );
    setTabOrder( m_numIgnore, m_cmbPulse1 );
    setTabOrder( m_cmbPulse1, m_cmbPulse2 );
    setTabOrder( m_cmbPulse2, m_cmbPulser );
    setTabOrder( m_cmbPulser, m_ckbAutoPhase );
    setTabOrder( m_ckbAutoPhase, m_ckbAbsFit );
    setTabOrder( m_ckbAbsFit, m_numPhase );
    setTabOrder( m_numPhase, m_edFreq );
    setTabOrder( m_edFreq, m_edBW );
    setTabOrder( m_edBW, m_edSmoothSamples );
    setTabOrder( m_edSmoothSamples, m_cmbFunction );
    setTabOrder( m_cmbFunction, m_txtFitStatus );
    setTabOrder( m_txtFitStatus, m_btnResetFit );
    setTabOrder( m_btnResetFit, m_btnClear );
    setTabOrder( m_btnClear, m_urlDump );
    setTabOrder( m_urlDump, m_btnDump );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmNMRT1::~FrmNMRT1()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmNMRT1::languageChange()
{
    setCaption( tr2i18n( "NMR/NQR Relax(T1) Measurement" ) );
    m_btnDump->setText( tr2i18n( "DUMP" ) );
    groupBox4->setTitle( tr2i18n( "Acquision Control" ) );
    m_ckbActive->setText( tr2i18n( "Control Pulser" ) );
    textLabel1_2->setText( tr2i18n( "<i>P</i><font size=\"-1\">1</font>(ms) / 2Tau (us)" ) );
    textLabel2_2_3->setText( tr2i18n( "Min" ) );
    textLabel2_2_2->setText( tr2i18n( "Max" ) );
    m_ckbT2Mode->setText( tr2i18n( "T2 Measurement" ) );
    textLabel1->setText( tr2i18n( "<i>P</i><font size=\"-1\">1</font> Dist" ) );
    textLabel5_2_3_3_2->setText( tr2i18n( "Extra Average" ) );
    textLabel5_2_3_3_2_2->setText( tr2i18n( "Ignore Cnt." ) );
    textLabel1_3_2_2->setText( tr2i18n( "PulseAnalyzer 1" ) );
    textLabel1_3_2->setText( tr2i18n( "PulseAnalyzer 2" ) );
    textLabel1_3->setText( tr2i18n( "Pulser" ) );
    groupBox5->setTitle( tr2i18n( "Data Processing" ) );
    m_ckbAutoPhase->setText( tr2i18n( "Auto Phase" ) );
    m_ckbAbsFit->setText( tr2i18n( "Abs Value Fit" ) );
    textLabel4_2->setText( tr2i18n( "Fitting Status" ) );
    textLabel5->setText( tr2i18n( "Relax Func" ) );
    textLabel3_2->setText( tr2i18n( "Smooth Samples" ) );
    textLabel3_3->setText( tr2i18n( "Band Width" ) );
    textLabel2_3_2_2->setText( tr2i18n( "kHz" ) );
    textLabel4->setText( tr2i18n( "Phase" ) );
    m_numPhase->setSuffix( tr2i18n( " deg" ) );
    textLabel3->setText( tr2i18n( "Frequency" ) );
    textLabel2_3_2->setText( tr2i18n( "kHz" ) );
    m_btnClear->setText( tr2i18n( "CLEAR CURVE" ) );
    m_btnResetFit->setText( tr2i18n( "RESET FITTING" ) );
}

#include "nmrrelaxform.moc"
