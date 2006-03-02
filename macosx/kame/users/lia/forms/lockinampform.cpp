#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../../../kame/users/lia/forms/lockinampform.ui'
**
** Created: 木  3 2 16:39:32 2006
**      by: The User Interface Compiler ($Id: lockinampform.cpp,v 1.1.2.1 2006/03/02 09:20:46 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "lockinampform.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qcheckbox.h>
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

/*
 *  Constructs a FrmLIA as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
FrmLIA::FrmLIA( QWidget* parent, const char* name, WFlags fl )
    : QMainWindow( parent, name, fl )
{
    (void)statusBar();
    if ( !name )
	setName( "FrmLIA" );
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );
    FrmLIALayout = new QGridLayout( centralWidget(), 1, 1, 2, 6, "FrmLIALayout"); 

    m_ckbAutoScaleY = new QCheckBox( centralWidget(), "m_ckbAutoScaleY" );
    m_ckbAutoScaleY->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckbAutoScaleY->sizePolicy().hasHeightForWidth() ) );

    FrmLIALayout->addWidget( m_ckbAutoScaleY, 6, 0 );

    layout3_2 = new QHBoxLayout( 0, 0, 6, "layout3_2"); 

    textLabel2_2 = new QLabel( centralWidget(), "textLabel2_2" );
    layout3_2->addWidget( textLabel2_2 );

    m_edFetchFreq = new QLineEdit( centralWidget(), "m_edFetchFreq" );
    layout3_2->addWidget( m_edFetchFreq );

    FrmLIALayout->addLayout( layout3_2, 4, 0 );

    m_ckbAutoScaleX = new QCheckBox( centralWidget(), "m_ckbAutoScaleX" );
    m_ckbAutoScaleX->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckbAutoScaleX->sizePolicy().hasHeightForWidth() ) );

    FrmLIALayout->addWidget( m_ckbAutoScaleX, 5, 0 );

    layout2 = new QHBoxLayout( 0, 0, 6, "layout2"); 

    textLabel1 = new QLabel( centralWidget(), "textLabel1" );
    layout2->addWidget( textLabel1 );

    m_cmbSens = new QComboBox( FALSE, centralWidget(), "m_cmbSens" );
    layout2->addWidget( m_cmbSens );

    FrmLIALayout->addLayout( layout2, 2, 0 );

    layout1 = new QHBoxLayout( 0, 0, 6, "layout1"); 

    textLabel4 = new QLabel( centralWidget(), "textLabel4" );
    layout1->addWidget( textLabel4 );

    m_cmbTimeConst = new QComboBox( FALSE, centralWidget(), "m_cmbTimeConst" );
    layout1->addWidget( m_cmbTimeConst );

    FrmLIALayout->addLayout( layout1, 3, 0 );

    layout3 = new QHBoxLayout( 0, 0, 6, "layout3"); 

    textLabel2 = new QLabel( centralWidget(), "textLabel2" );
    layout3->addWidget( textLabel2 );

    m_edOutput = new QLineEdit( centralWidget(), "m_edOutput" );
    layout3->addWidget( m_edOutput );

    textLabel3 = new QLabel( centralWidget(), "textLabel3" );
    layout3->addWidget( textLabel3 );

    FrmLIALayout->addLayout( layout3, 0, 0 );

    layout3_3 = new QHBoxLayout( 0, 0, 6, "layout3_3"); 

    textLabel2_3 = new QLabel( centralWidget(), "textLabel2_3" );
    layout3_3->addWidget( textLabel2_3 );

    m_edFreq = new QLineEdit( centralWidget(), "m_edFreq" );
    layout3_3->addWidget( m_edFreq );

    textLabel3_2 = new QLabel( centralWidget(), "textLabel3_2" );
    layout3_3->addWidget( textLabel3_2 );

    FrmLIALayout->addLayout( layout3_3, 1, 0 );

    // toolbars

    languageChange();
    resize( QSize(174, 212).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmLIA::~FrmLIA()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmLIA::languageChange()
{
    setCaption( tr2i18n( "LIA Control" ) );
    m_ckbAutoScaleY->setText( tr2i18n( "Autoscale for Y" ) );
    textLabel2_2->setText( tr2i18n( "Fetch Freq." ) );
    m_ckbAutoScaleX->setText( tr2i18n( "Autoscale for X" ) );
    textLabel1->setText( tr2i18n( "Sensitibity" ) );
    textLabel4->setText( tr2i18n( "Time Const." ) );
    textLabel2->setText( tr2i18n( "Output" ) );
    textLabel3->setText( tr2i18n( "V" ) );
    textLabel2_3->setText( tr2i18n( "Frequency" ) );
    textLabel3_2->setText( tr2i18n( "Hz" ) );
}

#include "lockinampform.moc"
