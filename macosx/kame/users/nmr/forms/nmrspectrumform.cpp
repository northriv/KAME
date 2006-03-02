#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../../../kame/users/nmr/forms/nmrspectrumform.ui'
**
** Created: 木  3 2 16:41:10 2006
**      by: The User Interface Compiler ($Id: nmrspectrumform.cpp,v 1.1.2.1 2006/03/02 09:19:12 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "nmrspectrumform.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <kurlrequester.h>
#include <qlabel.h>
#include <qlineedit.h>
#include <qcombobox.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include <qaction.h>
#include <qmenubar.h>
#include <qpopupmenu.h>
#include <qtoolbar.h>
#include "../../graph/graphwidget.h"

/*
 *  Constructs a FrmNMRSpectrum as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
FrmNMRSpectrum::FrmNMRSpectrum( QWidget* parent, const char* name, WFlags fl )
    : QMainWindow( parent, name, fl )
{
    (void)statusBar();
    if ( !name )
	setName( "FrmNMRSpectrum" );
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );
    FrmNMRSpectrumLayout = new QGridLayout( centralWidget(), 1, 1, 2, 6, "FrmNMRSpectrumLayout"); 

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

    FrmNMRSpectrumLayout->addMultiCellLayout( layout19, 0, 6, 1, 1 );

    layout12_2_2 = new QHBoxLayout( 0, 0, 6, "layout12_2_2"); 

    layout2_3_2 = new QVBoxLayout( 0, 0, 6, "layout2_3_2"); 

    textLabel1_3_2 = new QLabel( centralWidget(), "textLabel1_3_2" );
    textLabel1_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_3_2->sizePolicy().hasHeightForWidth() ) );
    layout2_3_2->addWidget( textLabel1_3_2 );

    layout1_3_2 = new QHBoxLayout( 0, 0, 6, "layout1_3_2"); 

    m_edFieldFactor = new QLineEdit( centralWidget(), "m_edFieldFactor" );
    m_edFieldFactor->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edFieldFactor->sizePolicy().hasHeightForWidth() ) );
    m_edFieldFactor->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2->addWidget( m_edFieldFactor );

    textLabel2_3_2 = new QLabel( centralWidget(), "textLabel2_3_2" );
    textLabel2_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2->addWidget( textLabel2_3_2 );
    layout2_3_2->addLayout( layout1_3_2 );
    layout12_2_2->addLayout( layout2_3_2 );

    layout2_2_2_2 = new QVBoxLayout( 0, 0, 6, "layout2_2_2_2"); 

    textLabel1_2_2_2 = new QLabel( centralWidget(), "textLabel1_2_2_2" );
    textLabel1_2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout2_2_2_2->addWidget( textLabel1_2_2_2 );

    layout1_2_2_2 = new QHBoxLayout( 0, 0, 6, "layout1_2_2_2"); 

    m_edResidual = new QLineEdit( centralWidget(), "m_edResidual" );
    m_edResidual->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edResidual->sizePolicy().hasHeightForWidth() ) );
    m_edResidual->setMaximumSize( QSize( 80, 32767 ) );
    layout1_2_2_2->addWidget( m_edResidual );

    textLabel2_2_2_2 = new QLabel( centralWidget(), "textLabel2_2_2_2" );
    textLabel2_2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout1_2_2_2->addWidget( textLabel2_2_2_2 );
    layout2_2_2_2->addLayout( layout1_2_2_2 );
    layout12_2_2->addLayout( layout2_2_2_2 );

    FrmNMRSpectrumLayout->addLayout( layout12_2_2, 5, 0 );

    m_btnClear = new QPushButton( centralWidget(), "m_btnClear" );
    m_btnClear->setAutoDefault( FALSE );

    FrmNMRSpectrumLayout->addWidget( m_btnClear, 6, 0 );

    layout12_2 = new QHBoxLayout( 0, 0, 6, "layout12_2"); 

    layout2_3 = new QVBoxLayout( 0, 0, 6, "layout2_3"); 

    textLabel1_3 = new QLabel( centralWidget(), "textLabel1_3" );
    textLabel1_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_3->sizePolicy().hasHeightForWidth() ) );
    layout2_3->addWidget( textLabel1_3 );

    layout1_3 = new QHBoxLayout( 0, 0, 6, "layout1_3"); 

    m_edHMin = new QLineEdit( centralWidget(), "m_edHMin" );
    m_edHMin->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edHMin->sizePolicy().hasHeightForWidth() ) );
    m_edHMin->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3->addWidget( m_edHMin );

    textLabel2_3 = new QLabel( centralWidget(), "textLabel2_3" );
    textLabel2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3->sizePolicy().hasHeightForWidth() ) );
    layout1_3->addWidget( textLabel2_3 );
    layout2_3->addLayout( layout1_3 );
    layout12_2->addLayout( layout2_3 );

    layout2_2_2 = new QVBoxLayout( 0, 0, 6, "layout2_2_2"); 

    textLabel1_2_2 = new QLabel( centralWidget(), "textLabel1_2_2" );
    textLabel1_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_2_2->sizePolicy().hasHeightForWidth() ) );
    layout2_2_2->addWidget( textLabel1_2_2 );

    layout1_2_2 = new QHBoxLayout( 0, 0, 6, "layout1_2_2"); 

    m_edHMax = new QLineEdit( centralWidget(), "m_edHMax" );
    m_edHMax->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edHMax->sizePolicy().hasHeightForWidth() ) );
    m_edHMax->setMaximumSize( QSize( 80, 32767 ) );
    layout1_2_2->addWidget( m_edHMax );

    textLabel2_2_2 = new QLabel( centralWidget(), "textLabel2_2_2" );
    textLabel2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout1_2_2->addWidget( textLabel2_2_2 );
    layout2_2_2->addLayout( layout1_2_2 );
    layout12_2->addLayout( layout2_2_2 );

    FrmNMRSpectrumLayout->addLayout( layout12_2, 3, 0 );

    layout12 = new QHBoxLayout( 0, 0, 6, "layout12"); 

    layout2 = new QVBoxLayout( 0, 0, 6, "layout2"); 

    textLabel1 = new QLabel( centralWidget(), "textLabel1" );
    textLabel1->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1->sizePolicy().hasHeightForWidth() ) );
    layout2->addWidget( textLabel1 );

    layout1 = new QHBoxLayout( 0, 0, 6, "layout1"); 

    m_edFreq = new QLineEdit( centralWidget(), "m_edFreq" );
    m_edFreq->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edFreq->sizePolicy().hasHeightForWidth() ) );
    m_edFreq->setMaximumSize( QSize( 80, 32767 ) );
    layout1->addWidget( m_edFreq );

    textLabel2 = new QLabel( centralWidget(), "textLabel2" );
    textLabel2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2->sizePolicy().hasHeightForWidth() ) );
    layout1->addWidget( textLabel2 );
    layout2->addLayout( layout1 );
    layout12->addLayout( layout2 );

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
    layout12->addLayout( layout2_2 );

    FrmNMRSpectrumLayout->addLayout( layout12, 2, 0 );

    layout2_3_3 = new QVBoxLayout( 0, 0, 6, "layout2_3_3"); 

    textLabel1_3_3 = new QLabel( centralWidget(), "textLabel1_3_3" );
    textLabel1_3_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_3_3->sizePolicy().hasHeightForWidth() ) );
    layout2_3_3->addWidget( textLabel1_3_3 );

    layout1_3_3 = new QHBoxLayout( 0, 0, 6, "layout1_3_3"); 

    m_edResolution = new QLineEdit( centralWidget(), "m_edResolution" );
    m_edResolution->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edResolution->sizePolicy().hasHeightForWidth() ) );
    m_edResolution->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_3->addWidget( m_edResolution );

    textLabel2_3_3 = new QLabel( centralWidget(), "textLabel2_3_3" );
    textLabel2_3_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_3->sizePolicy().hasHeightForWidth() ) );
    layout1_3_3->addWidget( textLabel2_3_3 );
    layout2_3_3->addLayout( layout1_3_3 );

    FrmNMRSpectrumLayout->addLayout( layout2_3_3, 4, 0 );

    layout49 = new QHBoxLayout( 0, 0, 6, "layout49"); 

    textLabel1_4 = new QLabel( centralWidget(), "textLabel1_4" );
    layout49->addWidget( textLabel1_4 );

    m_cmbFieldEntry = new QComboBox( FALSE, centralWidget(), "m_cmbFieldEntry" );
    layout49->addWidget( m_cmbFieldEntry );

    FrmNMRSpectrumLayout->addLayout( layout49, 0, 0 );

    layout49_2 = new QHBoxLayout( 0, 0, 6, "layout49_2"); 

    textLabel1_4_2 = new QLabel( centralWidget(), "textLabel1_4_2" );
    layout49_2->addWidget( textLabel1_4_2 );

    m_cmbPulse = new QComboBox( FALSE, centralWidget(), "m_cmbPulse" );
    layout49_2->addWidget( m_cmbPulse );

    FrmNMRSpectrumLayout->addLayout( layout49_2, 1, 0 );

    // toolbars

    languageChange();
    resize( QSize(693, 348).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // tab order
    setTabOrder( m_cmbFieldEntry, m_cmbPulse );
    setTabOrder( m_cmbPulse, m_edFreq );
    setTabOrder( m_edFreq, m_edBW );
    setTabOrder( m_edBW, m_edHMin );
    setTabOrder( m_edHMin, m_edHMax );
    setTabOrder( m_edHMax, m_edResolution );
    setTabOrder( m_edResolution, m_edFieldFactor );
    setTabOrder( m_edFieldFactor, m_edResidual );
    setTabOrder( m_edResidual, m_btnClear );
    setTabOrder( m_btnClear, m_urlDump );
    setTabOrder( m_urlDump, m_btnDump );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmNMRSpectrum::~FrmNMRSpectrum()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmNMRSpectrum::languageChange()
{
    setCaption( tr2i18n( "NMR Spectrum Averager" ) );
    m_btnDump->setText( tr2i18n( "DUMP" ) );
    textLabel1_3_2->setText( tr2i18n( "Field Factor" ) );
    textLabel2_3_2->setText( tr2i18n( "T/?" ) );
    textLabel1_2_2_2->setText( tr2i18n( "Residual Field" ) );
    textLabel2_2_2_2->setText( tr2i18n( "T" ) );
    m_btnClear->setText( tr2i18n( "CLEAR SPECTRUM" ) );
    textLabel1_3->setText( tr2i18n( "Field Min" ) );
    textLabel2_3->setText( tr2i18n( "T" ) );
    textLabel1_2_2->setText( tr2i18n( "Field Max" ) );
    textLabel2_2_2->setText( tr2i18n( "T" ) );
    textLabel1->setText( tr2i18n( "Center Freq" ) );
    textLabel2->setText( tr2i18n( "MHz" ) );
    textLabel1_2->setText( tr2i18n( "Band Width" ) );
    textLabel2_2->setText( tr2i18n( "kHz" ) );
    textLabel1_3_3->setText( tr2i18n( "Resolution" ) );
    textLabel2_3_3->setText( tr2i18n( "T" ) );
    textLabel1_4->setText( tr2i18n( "Magnet PS" ) );
    textLabel1_4_2->setText( tr2i18n( "NMR Pulse Analyzer" ) );
}

#include "nmrspectrumform.moc"
