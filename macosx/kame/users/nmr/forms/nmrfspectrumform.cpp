#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../../../kame/users/nmr/forms/nmrfspectrumform.ui'
**
** Created: 木  3 2 16:40:40 2006
**      by: The User Interface Compiler ($Id: nmrfspectrumform.cpp,v 1.1.2.1 2006/03/02 09:19:11 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "nmrfspectrumform.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qcheckbox.h>
#include <qlabel.h>
#include <qlineedit.h>
#include <qcombobox.h>
#include <qgroupbox.h>
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
 *  Constructs a FrmNMRFSpectrum as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
FrmNMRFSpectrum::FrmNMRFSpectrum( QWidget* parent, const char* name, WFlags fl )
    : QMainWindow( parent, name, fl )
{
    (void)statusBar();
    if ( !name )
	setName( "FrmNMRFSpectrum" );
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );
    FrmNMRFSpectrumLayout = new QGridLayout( centralWidget(), 1, 1, 2, 6, "FrmNMRFSpectrumLayout"); 

    layout23 = new QVBoxLayout( 0, 0, 6, "layout23"); 

    m_ckbActive = new QCheckBox( centralWidget(), "m_ckbActive" );
    m_ckbActive->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckbActive->sizePolicy().hasHeightForWidth() ) );
    layout23->addWidget( m_ckbActive );

    layout2 = new QVBoxLayout( 0, 0, 6, "layout2"); 

    textLabel1 = new QLabel( centralWidget(), "textLabel1" );
    textLabel1->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1->sizePolicy().hasHeightForWidth() ) );
    layout2->addWidget( textLabel1 );

    layout1 = new QHBoxLayout( 0, 0, 6, "layout1"); 

    m_edCenterFreq = new QLineEdit( centralWidget(), "m_edCenterFreq" );
    m_edCenterFreq->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edCenterFreq->sizePolicy().hasHeightForWidth() ) );
    m_edCenterFreq->setMaximumSize( QSize( 80, 32767 ) );
    layout1->addWidget( m_edCenterFreq );

    textLabel2 = new QLabel( centralWidget(), "textLabel2" );
    textLabel2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2->sizePolicy().hasHeightForWidth() ) );
    layout1->addWidget( textLabel2 );
    layout2->addLayout( layout1 );
    layout23->addLayout( layout2 );

    layout2_3 = new QVBoxLayout( 0, 0, 6, "layout2_3"); 

    textLabel1_3 = new QLabel( centralWidget(), "textLabel1_3" );
    textLabel1_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_3->sizePolicy().hasHeightForWidth() ) );
    layout2_3->addWidget( textLabel1_3 );

    layout1_3 = new QHBoxLayout( 0, 0, 6, "layout1_3"); 

    m_edFreqStep = new QLineEdit( centralWidget(), "m_edFreqStep" );
    m_edFreqStep->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edFreqStep->sizePolicy().hasHeightForWidth() ) );
    m_edFreqStep->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3->addWidget( m_edFreqStep );

    textLabel2_3 = new QLabel( centralWidget(), "textLabel2_3" );
    textLabel2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3->sizePolicy().hasHeightForWidth() ) );
    layout1_3->addWidget( textLabel2_3 );
    layout2_3->addLayout( layout1_3 );
    layout23->addLayout( layout2_3 );

    layout2_2_2 = new QVBoxLayout( 0, 0, 6, "layout2_2_2"); 

    textLabel1_2_2 = new QLabel( centralWidget(), "textLabel1_2_2" );
    textLabel1_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_2_2->sizePolicy().hasHeightForWidth() ) );
    layout2_2_2->addWidget( textLabel1_2_2 );

    layout1_2_2 = new QHBoxLayout( 0, 0, 6, "layout1_2_2"); 

    m_edFreqSpan = new QLineEdit( centralWidget(), "m_edFreqSpan" );
    m_edFreqSpan->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edFreqSpan->sizePolicy().hasHeightForWidth() ) );
    m_edFreqSpan->setMaximumSize( QSize( 80, 32767 ) );
    layout1_2_2->addWidget( m_edFreqSpan );

    textLabel2_2_2 = new QLabel( centralWidget(), "textLabel2_2_2" );
    textLabel2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout1_2_2->addWidget( textLabel2_2_2 );
    layout2_2_2->addLayout( layout1_2_2 );
    layout23->addLayout( layout2_2_2 );

    layout2_2 = new QVBoxLayout( 0, 0, 6, "layout2_2"); 

    textLabel1_2 = new QLabel( centralWidget(), "textLabel1_2" );
    textLabel1_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_2->sizePolicy().hasHeightForWidth() ) );
    layout2_2->addWidget( textLabel1_2 );

    layout1_2 = new QHBoxLayout( 0, 0, 6, "layout1_2"); 

    m_edBW = new QLineEdit( centralWidget(), "m_edBW" );
    m_edBW->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edBW->sizePolicy().hasHeightForWidth() ) );
    m_edBW->setMaximumSize( QSize( 80, 32767 ) );
    layout1_2->addWidget( m_edBW );

    textLabel2_2 = new QLabel( centralWidget(), "textLabel2_2" );
    textLabel2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_2->sizePolicy().hasHeightForWidth() ) );
    layout1_2->addWidget( textLabel2_2 );
    layout2_2->addLayout( layout1_2 );
    layout23->addLayout( layout2_2 );

    m_btnClear = new QPushButton( centralWidget(), "m_btnClear" );
    m_btnClear->setAutoDefault( FALSE );
    layout23->addWidget( m_btnClear );

    FrmNMRFSpectrumLayout->addMultiCellLayout( layout23, 0, 1, 0, 0 );

    layout22 = new QVBoxLayout( 0, 0, 6, "layout22"); 

    layout12 = new QVBoxLayout( 0, 0, 6, "layout12"); 

    textLabel1_4 = new QLabel( centralWidget(), "textLabel1_4" );
    textLabel1_4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_4->sizePolicy().hasHeightForWidth() ) );
    layout12->addWidget( textLabel1_4 );

    m_cmbPulse = new QComboBox( FALSE, centralWidget(), "m_cmbPulse" );
    layout12->addWidget( m_cmbPulse );
    layout22->addLayout( layout12 );

    groupBox1 = new QGroupBox( centralWidget(), "groupBox1" );
    groupBox1->setColumnLayout(0, Qt::Vertical );
    groupBox1->layout()->setSpacing( 6 );
    groupBox1->layout()->setMargin( 11 );
    groupBox1Layout = new QGridLayout( groupBox1->layout() );
    groupBox1Layout->setAlignment( Qt::AlignTop );

    layout1_4 = new QHBoxLayout( 0, 0, 6, "layout1_4"); 

    m_edSG1FreqOffset = new QLineEdit( groupBox1, "m_edSG1FreqOffset" );
    m_edSG1FreqOffset->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edSG1FreqOffset->sizePolicy().hasHeightForWidth() ) );
    m_edSG1FreqOffset->setMaximumSize( QSize( 80, 32767 ) );
    layout1_4->addWidget( m_edSG1FreqOffset );

    textLabel2_4 = new QLabel( groupBox1, "textLabel2_4" );
    textLabel2_4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_4->sizePolicy().hasHeightForWidth() ) );
    layout1_4->addWidget( textLabel2_4 );

    groupBox1Layout->addLayout( layout1_4, 2, 0 );

    m_cmbSG1 = new QComboBox( FALSE, groupBox1, "m_cmbSG1" );

    groupBox1Layout->addWidget( m_cmbSG1, 0, 0 );

    textLabel1_5 = new QLabel( groupBox1, "textLabel1_5" );
    textLabel1_5->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_5->sizePolicy().hasHeightForWidth() ) );

    groupBox1Layout->addWidget( textLabel1_5, 1, 0 );
    layout22->addWidget( groupBox1 );

    groupBox1_2 = new QGroupBox( centralWidget(), "groupBox1_2" );
    groupBox1_2->setColumnLayout(0, Qt::Vertical );
    groupBox1_2->layout()->setSpacing( 6 );
    groupBox1_2->layout()->setMargin( 11 );
    groupBox1_2Layout = new QGridLayout( groupBox1_2->layout() );
    groupBox1_2Layout->setAlignment( Qt::AlignTop );

    layout1_4_2 = new QHBoxLayout( 0, 0, 6, "layout1_4_2"); 

    m_edSG2FreqOffset = new QLineEdit( groupBox1_2, "m_edSG2FreqOffset" );
    m_edSG2FreqOffset->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edSG2FreqOffset->sizePolicy().hasHeightForWidth() ) );
    m_edSG2FreqOffset->setMaximumSize( QSize( 80, 32767 ) );
    layout1_4_2->addWidget( m_edSG2FreqOffset );

    textLabel2_4_2 = new QLabel( groupBox1_2, "textLabel2_4_2" );
    textLabel2_4_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_4_2->sizePolicy().hasHeightForWidth() ) );
    layout1_4_2->addWidget( textLabel2_4_2 );

    groupBox1_2Layout->addLayout( layout1_4_2, 2, 0 );

    m_cmbSG2 = new QComboBox( FALSE, groupBox1_2, "m_cmbSG2" );

    groupBox1_2Layout->addWidget( m_cmbSG2, 0, 0 );

    textLabel1_5_2 = new QLabel( groupBox1_2, "textLabel1_5_2" );
    textLabel1_5_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_5_2->sizePolicy().hasHeightForWidth() ) );

    groupBox1_2Layout->addWidget( textLabel1_5_2, 1, 0 );
    layout22->addWidget( groupBox1_2 );

    FrmNMRFSpectrumLayout->addMultiCellLayout( layout22, 0, 1, 2, 2 );

    layout15 = new QHBoxLayout( 0, 0, 6, "layout15"); 

    m_urlDump = new KURLRequester( centralWidget(), "m_urlDump" );
    m_urlDump->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)7, (QSizePolicy::SizeType)5, 0, 0, m_urlDump->sizePolicy().hasHeightForWidth() ) );
    m_urlDump->setCursor( QCursor( 0 ) );
    m_urlDump->setFocusPolicy( KURLRequester::StrongFocus );
    layout15->addWidget( m_urlDump );

    m_btnDump = new QPushButton( centralWidget(), "m_btnDump" );
    m_btnDump->setAutoDefault( FALSE );
    layout15->addWidget( m_btnDump );

    FrmNMRFSpectrumLayout->addLayout( layout15, 1, 1 );

    m_graph = new XQGraph( centralWidget(), "m_graph" );
    m_graph->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)7, (QSizePolicy::SizeType)7, 0, 0, m_graph->sizePolicy().hasHeightForWidth() ) );

    FrmNMRFSpectrumLayout->addWidget( m_graph, 0, 1 );

    // toolbars

    languageChange();
    resize( QSize(672, 287).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // tab order
    setTabOrder( m_edCenterFreq, m_edBW );
    setTabOrder( m_edBW, m_edFreqStep );
    setTabOrder( m_edFreqStep, m_edFreqSpan );
    setTabOrder( m_edFreqSpan, m_btnClear );
    setTabOrder( m_btnClear, m_urlDump );
    setTabOrder( m_urlDump, m_btnDump );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmNMRFSpectrum::~FrmNMRFSpectrum()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmNMRFSpectrum::languageChange()
{
    setCaption( tr2i18n( "NMR Spectrum Averager (Frequency Sweep)" ) );
    m_ckbActive->setText( tr2i18n( "Activate Freq. Sweep" ) );
    textLabel1->setText( tr2i18n( "Center Freq" ) );
    textLabel2->setText( tr2i18n( "MHz" ) );
    textLabel1_3->setText( tr2i18n( "Step" ) );
    textLabel2_3->setText( tr2i18n( "kHz" ) );
    textLabel1_2_2->setText( tr2i18n( "Span" ) );
    textLabel2_2_2->setText( tr2i18n( "kHz" ) );
    textLabel1_2->setText( tr2i18n( "Band Width" ) );
    textLabel2_2->setText( tr2i18n( "kHz" ) );
    m_btnClear->setText( tr2i18n( "CLEAR SPECTRUM" ) );
    textLabel1_4->setText( tr2i18n( "Pulse Analyzer" ) );
    groupBox1->setTitle( tr2i18n( "Signal Generator 1" ) );
    textLabel2_4->setText( tr2i18n( "MHz" ) );
    textLabel1_5->setText( tr2i18n( "Offset Freq." ) );
    groupBox1_2->setTitle( tr2i18n( "Signal Generator 2" ) );
    textLabel2_4_2->setText( tr2i18n( "MHz" ) );
    textLabel1_5_2->setText( tr2i18n( "Offset Freq." ) );
    m_btnDump->setText( tr2i18n( "DUMP" ) );
}

#include "nmrfspectrumform.moc"
