#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../../../kame/users/dso/forms/dsoform.ui'
**
** Created: 木  3 2 16:36:29 2006
**      by: The User Interface Compiler ($Id: dsoform.cpp,v 1.1.2.1 2006/03/02 09:19:33 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "dsoform.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qcombobox.h>
#include <qlineedit.h>
#include <qcheckbox.h>
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
 *  Constructs a FrmDSO as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
FrmDSO::FrmDSO( QWidget* parent, const char* name, WFlags fl )
    : QMainWindow( parent, name, fl )
{
    (void)statusBar();
    if ( !name )
	setName( "FrmDSO" );
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );
    FrmDSOLayout = new QGridLayout( centralWidget(), 1, 1, 2, 6, "FrmDSOLayout"); 

    layout2_2_2 = new QHBoxLayout( 0, 0, 6, "layout2_2_2"); 

    textLabel2_2_2 = new QLabel( centralWidget(), "textLabel2_2_2" );
    textLabel2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout2_2_2->addWidget( textLabel2_2_2 );

    m_cmbRecordLength = new QComboBox( FALSE, centralWidget(), "m_cmbRecordLength" );
    m_cmbRecordLength->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_cmbRecordLength->sizePolicy().hasHeightForWidth() ) );
    layout2_2_2->addWidget( m_cmbRecordLength );

    FrmDSOLayout->addLayout( layout2_2_2, 6, 0 );

    layout5_4_2 = new QHBoxLayout( 0, 0, 6, "layout5_4_2"); 

    textLabel1_2_5_2 = new QLabel( centralWidget(), "textLabel1_2_5_2" );
    textLabel1_2_5_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_2_5_2->sizePolicy().hasHeightForWidth() ) );
    layout5_4_2->addWidget( textLabel1_2_5_2 );

    m_edTrigPos = new QLineEdit( centralWidget(), "m_edTrigPos" );
    m_edTrigPos->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edTrigPos->sizePolicy().hasHeightForWidth() ) );
    layout5_4_2->addWidget( m_edTrigPos );

    textLabel1_2_2_4_2 = new QLabel( centralWidget(), "textLabel1_2_2_4_2" );
    textLabel1_2_2_4_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_2_2_4_2->sizePolicy().hasHeightForWidth() ) );
    layout5_4_2->addWidget( textLabel1_2_2_4_2 );

    FrmDSOLayout->addLayout( layout5_4_2, 5, 0 );

    layout5_4 = new QHBoxLayout( 0, 0, 6, "layout5_4"); 

    textLabel1_2_5 = new QLabel( centralWidget(), "textLabel1_2_5" );
    textLabel1_2_5->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_2_5->sizePolicy().hasHeightForWidth() ) );
    layout5_4->addWidget( textLabel1_2_5 );

    m_edTimeWidth = new QLineEdit( centralWidget(), "m_edTimeWidth" );
    m_edTimeWidth->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)4, (QSizePolicy::SizeType)0, 0, 0, m_edTimeWidth->sizePolicy().hasHeightForWidth() ) );
    layout5_4->addWidget( m_edTimeWidth );

    textLabel1_2_2_4 = new QLabel( centralWidget(), "textLabel1_2_2_4" );
    textLabel1_2_2_4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_2_2_4->sizePolicy().hasHeightForWidth() ) );
    layout5_4->addWidget( textLabel1_2_2_4 );

    FrmDSOLayout->addLayout( layout5_4, 4, 0 );

    layout4 = new QHBoxLayout( 0, 0, 6, "layout4"); 

    textLabel1 = new QLabel( centralWidget(), "textLabel1" );
    textLabel1->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1->sizePolicy().hasHeightForWidth() ) );
    layout4->addWidget( textLabel1 );

    m_edAverage = new QLineEdit( centralWidget(), "m_edAverage" );
    m_edAverage->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edAverage->sizePolicy().hasHeightForWidth() ) );
    m_edAverage->setMinimumSize( QSize( 33, 0 ) );
    layout4->addWidget( m_edAverage );

    FrmDSOLayout->addLayout( layout4, 3, 0 );

    m_ckbFetch = new QCheckBox( centralWidget(), "m_ckbFetch" );
    m_ckbFetch->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckbFetch->sizePolicy().hasHeightForWidth() ) );

    FrmDSOLayout->addWidget( m_ckbFetch, 0, 0 );

    m_btnForceTrigger = new QPushButton( centralWidget(), "m_btnForceTrigger" );
    m_btnForceTrigger->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)1, (QSizePolicy::SizeType)0, 0, 0, m_btnForceTrigger->sizePolicy().hasHeightForWidth() ) );

    FrmDSOLayout->addWidget( m_btnForceTrigger, 2, 0 );

    groupBox4 = new QGroupBox( centralWidget(), "groupBox4" );
    groupBox4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)1, (QSizePolicy::SizeType)5, 0, 0, groupBox4->sizePolicy().hasHeightForWidth() ) );
    groupBox4->setColumnLayout(0, Qt::Vertical );
    groupBox4->layout()->setSpacing( 6 );
    groupBox4->layout()->setMargin( 2 );
    groupBox4Layout = new QGridLayout( groupBox4->layout() );
    groupBox4Layout->setAlignment( Qt::AlignTop );

    m_cmbTrace1 = new QComboBox( FALSE, groupBox4, "m_cmbTrace1" );
    m_cmbTrace1->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_cmbTrace1->sizePolicy().hasHeightForWidth() ) );

    groupBox4Layout->addWidget( m_cmbTrace1, 0, 0 );

    layout7 = new QVBoxLayout( 0, 0, 6, "layout7"); 

    layout5 = new QHBoxLayout( 0, 0, 6, "layout5"); 

    textLabel1_2 = new QLabel( groupBox4, "textLabel1_2" );
    textLabel1_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_2->sizePolicy().hasHeightForWidth() ) );
    layout5->addWidget( textLabel1_2 );

    m_edVFullScale1 = new QLineEdit( groupBox4, "m_edVFullScale1" );
    m_edVFullScale1->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edVFullScale1->sizePolicy().hasHeightForWidth() ) );
    layout5->addWidget( m_edVFullScale1 );

    textLabel1_2_2 = new QLabel( groupBox4, "textLabel1_2_2" );
    textLabel1_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_2_2->sizePolicy().hasHeightForWidth() ) );
    layout5->addWidget( textLabel1_2_2 );
    layout7->addLayout( layout5 );

    layout5_2 = new QHBoxLayout( 0, 0, 6, "layout5_2"); 

    textLabel1_2_3 = new QLabel( groupBox4, "textLabel1_2_3" );
    textLabel1_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_2_3->sizePolicy().hasHeightForWidth() ) );
    layout5_2->addWidget( textLabel1_2_3 );

    m_edVOffset1 = new QLineEdit( groupBox4, "m_edVOffset1" );
    m_edVOffset1->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)4, (QSizePolicy::SizeType)0, 0, 0, m_edVOffset1->sizePolicy().hasHeightForWidth() ) );
    layout5_2->addWidget( m_edVOffset1 );

    textLabel1_2_2_2 = new QLabel( groupBox4, "textLabel1_2_2_2" );
    textLabel1_2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout5_2->addWidget( textLabel1_2_2_2 );
    layout7->addLayout( layout5_2 );

    groupBox4Layout->addLayout( layout7, 1, 0 );

    FrmDSOLayout->addWidget( groupBox4, 7, 0 );

    groupBox5 = new QGroupBox( centralWidget(), "groupBox5" );
    groupBox5->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)1, (QSizePolicy::SizeType)5, 0, 0, groupBox5->sizePolicy().hasHeightForWidth() ) );
    groupBox5->setColumnLayout(0, Qt::Vertical );
    groupBox5->layout()->setSpacing( 6 );
    groupBox5->layout()->setMargin( 2 );
    groupBox5Layout = new QGridLayout( groupBox5->layout() );
    groupBox5Layout->setAlignment( Qt::AlignTop );

    m_cmbTrace2 = new QComboBox( FALSE, groupBox5, "m_cmbTrace2" );
    m_cmbTrace2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_cmbTrace2->sizePolicy().hasHeightForWidth() ) );

    groupBox5Layout->addWidget( m_cmbTrace2, 0, 0 );

    layout7_2 = new QVBoxLayout( 0, 0, 6, "layout7_2"); 

    layout5_3 = new QHBoxLayout( 0, 0, 6, "layout5_3"); 

    textLabel1_2_4 = new QLabel( groupBox5, "textLabel1_2_4" );
    textLabel1_2_4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_2_4->sizePolicy().hasHeightForWidth() ) );
    layout5_3->addWidget( textLabel1_2_4 );

    m_edVFullScale2 = new QLineEdit( groupBox5, "m_edVFullScale2" );
    m_edVFullScale2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edVFullScale2->sizePolicy().hasHeightForWidth() ) );
    layout5_3->addWidget( m_edVFullScale2 );

    textLabel1_2_2_3 = new QLabel( groupBox5, "textLabel1_2_2_3" );
    textLabel1_2_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_2_2_3->sizePolicy().hasHeightForWidth() ) );
    layout5_3->addWidget( textLabel1_2_2_3 );
    layout7_2->addLayout( layout5_3 );

    layout5_2_2 = new QHBoxLayout( 0, 0, 6, "layout5_2_2"); 

    textLabel1_2_3_2 = new QLabel( groupBox5, "textLabel1_2_3_2" );
    textLabel1_2_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_2_3_2->sizePolicy().hasHeightForWidth() ) );
    layout5_2_2->addWidget( textLabel1_2_3_2 );

    m_edVOffset2 = new QLineEdit( groupBox5, "m_edVOffset2" );
    m_edVOffset2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edVOffset2->sizePolicy().hasHeightForWidth() ) );
    layout5_2_2->addWidget( m_edVOffset2 );

    textLabel1_2_2_2_2 = new QLabel( groupBox5, "textLabel1_2_2_2_2" );
    textLabel1_2_2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_2_2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout5_2_2->addWidget( textLabel1_2_2_2_2 );
    layout7_2->addLayout( layout5_2_2 );

    groupBox5Layout->addLayout( layout7_2, 1, 0 );

    FrmDSOLayout->addWidget( groupBox5, 8, 0 );

    m_ckbSingleSeq = new QCheckBox( centralWidget(), "m_ckbSingleSeq" );
    m_ckbSingleSeq->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckbSingleSeq->sizePolicy().hasHeightForWidth() ) );

    FrmDSOLayout->addWidget( m_ckbSingleSeq, 1, 0 );

    groupBox4_2 = new QGroupBox( centralWidget(), "groupBox4_2" );
    groupBox4_2->setColumnLayout(0, Qt::Vertical );
    groupBox4_2->layout()->setSpacing( 6 );
    groupBox4_2->layout()->setMargin( 2 );
    groupBox4_2Layout = new QGridLayout( groupBox4_2->layout() );
    groupBox4_2Layout->setAlignment( Qt::AlignTop );

    m_cmbPulser = new QComboBox( FALSE, groupBox4_2, "m_cmbPulser" );
    m_cmbPulser->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_cmbPulser->sizePolicy().hasHeightForWidth() ) );

    groupBox4_2Layout->addMultiCellWidget( m_cmbPulser, 1, 1, 1, 2 );

    textLabel1_3 = new QLabel( groupBox4_2, "textLabel1_3" );
    textLabel1_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_3->sizePolicy().hasHeightForWidth() ) );

    groupBox4_2Layout->addWidget( textLabel1_3, 1, 0 );

    m_ckb4x = new QCheckBox( groupBox4_2, "m_ckb4x" );
    m_ckb4x->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckb4x->sizePolicy().hasHeightForWidth() ) );

    groupBox4_2Layout->addWidget( m_ckb4x, 0, 2 );

    m_ckbEnable = new QCheckBox( groupBox4_2, "m_ckbEnable" );
    m_ckbEnable->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckbEnable->sizePolicy().hasHeightForWidth() ) );

    groupBox4_2Layout->addMultiCellWidget( m_ckbEnable, 0, 0, 0, 1 );

    FrmDSOLayout->addWidget( groupBox4_2, 9, 0 );

    layout15 = new QVBoxLayout( 0, 0, 6, "layout15"); 

    layout15_2 = new QHBoxLayout( 0, 0, 6, "layout15_2"); 

    m_urlDump = new KURLRequester( centralWidget(), "m_urlDump" );
    m_urlDump->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)7, (QSizePolicy::SizeType)5, 0, 0, m_urlDump->sizePolicy().hasHeightForWidth() ) );
    m_urlDump->setCursor( QCursor( 0 ) );
    m_urlDump->setFocusPolicy( KURLRequester::StrongFocus );
    layout15_2->addWidget( m_urlDump );

    m_btnDump = new QPushButton( centralWidget(), "m_btnDump" );
    m_btnDump->setAutoDefault( FALSE );
    layout15_2->addWidget( m_btnDump );
    layout15->addLayout( layout15_2 );

    m_graphwidget = new XQGraph( centralWidget(), "m_graphwidget" );
    m_graphwidget->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)7, (QSizePolicy::SizeType)7, 0, 0, m_graphwidget->sizePolicy().hasHeightForWidth() ) );
    layout15->addWidget( m_graphwidget );

    groupBox1 = new QGroupBox( centralWidget(), "groupBox1" );
    groupBox1->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)1, (QSizePolicy::SizeType)5, 0, 0, groupBox1->sizePolicy().hasHeightForWidth() ) );
    groupBox1->setColumnLayout(0, Qt::Vertical );
    groupBox1->layout()->setSpacing( 6 );
    groupBox1->layout()->setMargin( 2 );
    groupBox1Layout = new QGridLayout( groupBox1->layout() );
    groupBox1Layout->setAlignment( Qt::AlignTop );

    m_ckbFIREnabled = new QCheckBox( groupBox1, "m_ckbFIREnabled" );
    m_ckbFIREnabled->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckbFIREnabled->sizePolicy().hasHeightForWidth() ) );

    groupBox1Layout->addWidget( m_ckbFIREnabled, 0, 0 );

    layout5_4_3_2 = new QHBoxLayout( 0, 0, 6, "layout5_4_3_2"); 

    textLabel1_2_5_3_2 = new QLabel( groupBox1, "textLabel1_2_5_3_2" );
    textLabel1_2_5_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_2_5_3_2->sizePolicy().hasHeightForWidth() ) );
    layout5_4_3_2->addWidget( textLabel1_2_5_3_2 );

    m_edFIRSharpness = new QLineEdit( groupBox1, "m_edFIRSharpness" );
    m_edFIRSharpness->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edFIRSharpness->sizePolicy().hasHeightForWidth() ) );
    m_edFIRSharpness->setMinimumSize( QSize( 23, 0 ) );
    layout5_4_3_2->addWidget( m_edFIRSharpness );

    groupBox1Layout->addLayout( layout5_4_3_2, 0, 1 );

    layout5_4_3 = new QHBoxLayout( 0, 0, 6, "layout5_4_3"); 

    textLabel1_2_5_3 = new QLabel( groupBox1, "textLabel1_2_5_3" );
    textLabel1_2_5_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_2_5_3->sizePolicy().hasHeightForWidth() ) );
    layout5_4_3->addWidget( textLabel1_2_5_3 );

    m_edFIRBandWidth = new QLineEdit( groupBox1, "m_edFIRBandWidth" );
    m_edFIRBandWidth->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edFIRBandWidth->sizePolicy().hasHeightForWidth() ) );
    m_edFIRBandWidth->setMinimumSize( QSize( 69, 0 ) );
    layout5_4_3->addWidget( m_edFIRBandWidth );

    textLabel1_2_2_4_3 = new QLabel( groupBox1, "textLabel1_2_2_4_3" );
    textLabel1_2_2_4_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_2_2_4_3->sizePolicy().hasHeightForWidth() ) );
    layout5_4_3->addWidget( textLabel1_2_2_4_3 );

    groupBox1Layout->addLayout( layout5_4_3, 1, 0 );

    layout5_4_3_3 = new QHBoxLayout( 0, 0, 6, "layout5_4_3_3"); 

    textLabel1_2_5_3_3 = new QLabel( groupBox1, "textLabel1_2_5_3_3" );
    textLabel1_2_5_3_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_2_5_3_3->sizePolicy().hasHeightForWidth() ) );
    layout5_4_3_3->addWidget( textLabel1_2_5_3_3 );

    m_edFIRCenterFreq = new QLineEdit( groupBox1, "m_edFIRCenterFreq" );
    m_edFIRCenterFreq->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edFIRCenterFreq->sizePolicy().hasHeightForWidth() ) );
    m_edFIRCenterFreq->setMinimumSize( QSize( 69, 0 ) );
    layout5_4_3_3->addWidget( m_edFIRCenterFreq );

    textLabel1_2_2_4_3_2 = new QLabel( groupBox1, "textLabel1_2_2_4_3_2" );
    textLabel1_2_2_4_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel1_2_2_4_3_2->sizePolicy().hasHeightForWidth() ) );
    layout5_4_3_3->addWidget( textLabel1_2_2_4_3_2 );

    groupBox1Layout->addLayout( layout5_4_3_3, 1, 1 );
    layout15->addWidget( groupBox1 );

    FrmDSOLayout->addMultiCellLayout( layout15, 0, 9, 1, 1 );

    // toolbars

    languageChange();
    resize( QSize(477, 501).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // tab order
    setTabOrder( m_ckbFetch, m_ckbSingleSeq );
    setTabOrder( m_ckbSingleSeq, m_btnForceTrigger );
    setTabOrder( m_btnForceTrigger, m_edAverage );
    setTabOrder( m_edAverage, m_edTimeWidth );
    setTabOrder( m_edTimeWidth, m_edTrigPos );
    setTabOrder( m_edTrigPos, m_cmbRecordLength );
    setTabOrder( m_cmbRecordLength, m_cmbTrace1 );
    setTabOrder( m_cmbTrace1, m_edVFullScale1 );
    setTabOrder( m_edVFullScale1, m_edVOffset1 );
    setTabOrder( m_edVOffset1, m_cmbTrace2 );
    setTabOrder( m_cmbTrace2, m_edVFullScale2 );
    setTabOrder( m_edVFullScale2, m_edVOffset2 );
    setTabOrder( m_edVOffset2, m_ckbFIREnabled );
    setTabOrder( m_ckbFIREnabled, m_edFIRSharpness );
    setTabOrder( m_edFIRSharpness, m_edFIRBandWidth );
    setTabOrder( m_edFIRBandWidth, m_edFIRCenterFreq );
    setTabOrder( m_edFIRCenterFreq, m_urlDump );
    setTabOrder( m_urlDump, m_btnDump );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmDSO::~FrmDSO()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmDSO::languageChange()
{
    setCaption( tr2i18n( "DSO Control" ) );
    textLabel2_2_2->setText( tr2i18n( "Record Length" ) );
    textLabel1_2_5_2->setText( tr2i18n( "Trigger Position" ) );
    textLabel1_2_2_4_2->setText( tr2i18n( "%" ) );
    textLabel1_2_5->setText( tr2i18n( "Time Width" ) );
    textLabel1_2_2_4->setText( tr2i18n( "sec." ) );
    textLabel1->setText( tr2i18n( "Average" ) );
    m_ckbFetch->setText( tr2i18n( "Fetch Continueous" ) );
    m_btnForceTrigger->setText( tr2i18n( "Force Trigger" ) );
    groupBox4->setTitle( tr2i18n( "Trace1" ) );
    textLabel1_2->setText( tr2i18n( "Full Scale" ) );
    textLabel1_2_2->setText( tr2i18n( "Vp-p" ) );
    textLabel1_2_3->setText( tr2i18n( "Offset" ) );
    textLabel1_2_2_2->setText( tr2i18n( "V" ) );
    groupBox5->setTitle( tr2i18n( "Trace2" ) );
    textLabel1_2_4->setText( tr2i18n( "Full Scale" ) );
    textLabel1_2_2_3->setText( tr2i18n( "Vp-p" ) );
    textLabel1_2_3_2->setText( tr2i18n( "Offset" ) );
    textLabel1_2_2_2_2->setText( tr2i18n( "V" ) );
    m_ckbSingleSeq->setText( tr2i18n( "Single Sequence" ) );
    groupBox4_2->setTitle( tr2i18n( "PhaseCycle Flipping" ) );
    textLabel1_3->setText( tr2i18n( "Pulser" ) );
    m_ckb4x->setText( tr2i18n( "4x" ) );
    m_ckbEnable->setText( tr2i18n( "Enable" ) );
    m_btnDump->setText( tr2i18n( "DUMP" ) );
    groupBox1->setTitle( tr2i18n( "Digital Filter" ) );
    m_ckbFIREnabled->setText( tr2i18n( "FIR filter" ) );
    textLabel1_2_5_3_2->setText( tr2i18n( "Sharpness" ) );
    textLabel1_2_5_3->setText( tr2i18n( "Band Width" ) );
    textLabel1_2_2_4_3->setText( tr2i18n( "kHz" ) );
    textLabel1_2_5_3_3->setText( tr2i18n( "Center Freq" ) );
    textLabel1_2_2_4_3_2->setText( tr2i18n( "kHz" ) );
}

#include "dsoform.moc"
