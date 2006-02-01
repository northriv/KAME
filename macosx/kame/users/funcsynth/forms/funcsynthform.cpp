#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../../../kame/users/funcsynth/forms/funcsynthform.ui'
**
** Created: 水  2 1 03:49:54 2006
**      by: The User Interface Compiler ($Id: funcsynthform.cpp,v 1.1 2006/02/01 18:45:30 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "funcsynthform.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qcheckbox.h>
#include <qlabel.h>
#include <qcombobox.h>
#include <qlineedit.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include <qaction.h>
#include <qmenubar.h>
#include <qpopupmenu.h>
#include <qtoolbar.h>

/*
 *  Constructs a FrmFuncSynth as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
FrmFuncSynth::FrmFuncSynth( QWidget* parent, const char* name, WFlags fl )
    : QMainWindow( parent, name, fl )
{
    (void)statusBar();
    if ( !name )
	setName( "FrmFuncSynth" );
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );
    FrmFuncSynthLayout = new QGridLayout( centralWidget(), 1, 1, 2, 6, "FrmFuncSynthLayout"); 

    m_ckbOutput = new QCheckBox( centralWidget(), "m_ckbOutput" );
    m_ckbOutput->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckbOutput->sizePolicy().hasHeightForWidth() ) );

    FrmFuncSynthLayout->addWidget( m_ckbOutput, 0, 0 );

    m_btnTrig = new QPushButton( centralWidget(), "m_btnTrig" );
    m_btnTrig->setAutoDefault( FALSE );

    FrmFuncSynthLayout->addWidget( m_btnTrig, 9, 0 );

    textLabel1 = new QLabel( centralWidget(), "textLabel1" );

    FrmFuncSynthLayout->addWidget( textLabel1, 1, 0 );

    m_cmbMode = new QComboBox( FALSE, centralWidget(), "m_cmbMode" );

    FrmFuncSynthLayout->addWidget( m_cmbMode, 2, 0 );

    textLabel2 = new QLabel( centralWidget(), "textLabel2" );

    FrmFuncSynthLayout->addWidget( textLabel2, 3, 0 );

    m_cmbFunc = new QComboBox( FALSE, centralWidget(), "m_cmbFunc" );

    FrmFuncSynthLayout->addWidget( m_cmbFunc, 4, 0 );

    layout23 = new QHBoxLayout( 0, 0, 6, "layout23"); 

    textLabel1_5_2_2 = new QLabel( centralWidget(), "textLabel1_5_2_2" );
    textLabel1_5_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_5_2_2->sizePolicy().hasHeightForWidth() ) );
    layout23->addWidget( textLabel1_5_2_2 );

    layout1_2_3 = new QHBoxLayout( 0, 0, 6, "layout1_2_3"); 

    m_edFreq = new QLineEdit( centralWidget(), "m_edFreq" );
    m_edFreq->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edFreq->sizePolicy().hasHeightForWidth() ) );
    m_edFreq->setMaximumSize( QSize( 80, 32767 ) );
    layout1_2_3->addWidget( m_edFreq );

    textLabel2_2_3 = new QLabel( centralWidget(), "textLabel2_2_3" );
    textLabel2_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_2_3->sizePolicy().hasHeightForWidth() ) );
    layout1_2_3->addWidget( textLabel2_2_3 );
    layout23->addLayout( layout1_2_3 );

    FrmFuncSynthLayout->addLayout( layout23, 5, 0 );

    layout23_3 = new QHBoxLayout( 0, 0, 6, "layout23_3"); 

    textLabel1_5_2_2_3 = new QLabel( centralWidget(), "textLabel1_5_2_2_3" );
    textLabel1_5_2_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_5_2_2_3->sizePolicy().hasHeightForWidth() ) );
    layout23_3->addWidget( textLabel1_5_2_2_3 );

    layout1_2_3_3 = new QHBoxLayout( 0, 0, 6, "layout1_2_3_3"); 

    m_edAmp = new QLineEdit( centralWidget(), "m_edAmp" );
    m_edAmp->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edAmp->sizePolicy().hasHeightForWidth() ) );
    m_edAmp->setMaximumSize( QSize( 80, 32767 ) );
    layout1_2_3_3->addWidget( m_edAmp );

    textLabel2_2_3_3 = new QLabel( centralWidget(), "textLabel2_2_3_3" );
    textLabel2_2_3_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_2_3_3->sizePolicy().hasHeightForWidth() ) );
    layout1_2_3_3->addWidget( textLabel2_2_3_3 );
    layout23_3->addLayout( layout1_2_3_3 );

    FrmFuncSynthLayout->addLayout( layout23_3, 6, 0 );

    layout23_2 = new QHBoxLayout( 0, 0, 6, "layout23_2"); 

    textLabel1_5_2_2_2 = new QLabel( centralWidget(), "textLabel1_5_2_2_2" );
    textLabel1_5_2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_5_2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout23_2->addWidget( textLabel1_5_2_2_2 );

    layout1_2_3_2 = new QHBoxLayout( 0, 0, 6, "layout1_2_3_2"); 

    m_edPhase = new QLineEdit( centralWidget(), "m_edPhase" );
    m_edPhase->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edPhase->sizePolicy().hasHeightForWidth() ) );
    m_edPhase->setMaximumSize( QSize( 80, 32767 ) );
    layout1_2_3_2->addWidget( m_edPhase );

    la = new QLabel( centralWidget(), "la" );
    la->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, la->sizePolicy().hasHeightForWidth() ) );
    layout1_2_3_2->addWidget( la );
    layout23_2->addLayout( layout1_2_3_2 );

    FrmFuncSynthLayout->addLayout( layout23_2, 7, 0 );

    layout23_2_2 = new QHBoxLayout( 0, 0, 6, "layout23_2_2"); 

    textLabel1_5_2_2_2_2 = new QLabel( centralWidget(), "textLabel1_5_2_2_2_2" );
    textLabel1_5_2_2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel1_5_2_2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout23_2_2->addWidget( textLabel1_5_2_2_2_2 );

    layout1_2_3_2_2 = new QHBoxLayout( 0, 0, 6, "layout1_2_3_2_2"); 

    m_edOffset = new QLineEdit( centralWidget(), "m_edOffset" );
    m_edOffset->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edOffset->sizePolicy().hasHeightForWidth() ) );
    m_edOffset->setMaximumSize( QSize( 80, 32767 ) );
    layout1_2_3_2_2->addWidget( m_edOffset );

    la_2 = new QLabel( centralWidget(), "la_2" );
    la_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, la_2->sizePolicy().hasHeightForWidth() ) );
    layout1_2_3_2_2->addWidget( la_2 );
    layout23_2_2->addLayout( layout1_2_3_2_2 );

    FrmFuncSynthLayout->addLayout( layout23_2_2, 8, 0 );

    // toolbars

    languageChange();
    resize( QSize(177, 288).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmFuncSynth::~FrmFuncSynth()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmFuncSynth::languageChange()
{
    setCaption( tr2i18n( "Function Synthesizer Control" ) );
    m_ckbOutput->setText( tr2i18n( "Output" ) );
    m_btnTrig->setText( tr2i18n( "Trigger Burst" ) );
    textLabel1->setText( tr2i18n( "Modulation" ) );
    textLabel2->setText( tr2i18n( "Function" ) );
    textLabel1_5_2_2->setText( tr2i18n( "Freq" ) );
    textLabel2_2_3->setText( tr2i18n( "Hz" ) );
    textLabel1_5_2_2_3->setText( tr2i18n( "Amp" ) );
    textLabel2_2_3_3->setText( tr2i18n( "Vp-p" ) );
    textLabel1_5_2_2_2->setText( tr2i18n( "Init. Phase" ) );
    la->setText( tr2i18n( "deg." ) );
    textLabel1_5_2_2_2_2->setText( tr2i18n( "Offset" ) );
    la_2->setText( tr2i18n( "V" ) );
}

#include "funcsynthform.moc"
