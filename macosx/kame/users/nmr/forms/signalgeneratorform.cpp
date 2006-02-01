#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../../../kame/users/nmr/forms/signalgeneratorform.ui'
**
** Created: 水  2 1 03:51:40 2006
**      by: The User Interface Compiler ($Id: signalgeneratorform.cpp,v 1.1 2006/02/01 18:43:56 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "signalgeneratorform.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qcheckbox.h>
#include <qlabel.h>
#include <qlineedit.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include <qaction.h>
#include <qmenubar.h>
#include <qpopupmenu.h>
#include <qtoolbar.h>

/*
 *  Constructs a FrmSG as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
FrmSG::FrmSG( QWidget* parent, const char* name, WFlags fl )
    : QMainWindow( parent, name, fl )
{
    (void)statusBar();
    if ( !name )
	setName( "FrmSG" );
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );
    FrmSGLayout = new QGridLayout( centralWidget(), 1, 1, 2, 6, "FrmSGLayout"); 

    layout41 = new QGridLayout( 0, 1, 1, 0, 6, "layout41"); 

    m_ckbAMON = new QCheckBox( centralWidget(), "m_ckbAMON" );
    m_ckbAMON->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckbAMON->sizePolicy().hasHeightForWidth() ) );

    layout41->addWidget( m_ckbAMON, 2, 0 );

    layout59_3 = new QHBoxLayout( 0, 0, 6, "layout59_3"); 

    textLabel5_3 = new QLabel( centralWidget(), "textLabel5_3" );
    textLabel5_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_3->sizePolicy().hasHeightForWidth() ) );
    layout59_3->addWidget( textLabel5_3 );

    layout1_3_2_3 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_3"); 

    m_edOLevel = new QLineEdit( centralWidget(), "m_edOLevel" );
    m_edOLevel->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edOLevel->sizePolicy().hasHeightForWidth() ) );
    m_edOLevel->setMinimumSize( QSize( 44, 0 ) );
    m_edOLevel->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_3->addWidget( m_edOLevel );

    textLabel2_3_2_3 = new QLabel( centralWidget(), "textLabel2_3_2_3" );
    textLabel2_3_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_3->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_3->addWidget( textLabel2_3_2_3 );
    layout59_3->addLayout( layout1_3_2_3 );

    layout41->addLayout( layout59_3, 0, 0 );

    m_ckbFMON = new QCheckBox( centralWidget(), "m_ckbFMON" );
    m_ckbFMON->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckbFMON->sizePolicy().hasHeightForWidth() ) );

    layout41->addWidget( m_ckbFMON, 3, 0 );

    layout59_3_2 = new QHBoxLayout( 0, 0, 6, "layout59_3_2"); 

    textLabel5_3_2 = new QLabel( centralWidget(), "textLabel5_3_2" );
    textLabel5_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_3_2->sizePolicy().hasHeightForWidth() ) );
    layout59_3_2->addWidget( textLabel5_3_2 );

    layout1_3_2_3_2 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_3_2"); 

    m_edFreq = new QLineEdit( centralWidget(), "m_edFreq" );
    m_edFreq->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edFreq->sizePolicy().hasHeightForWidth() ) );
    m_edFreq->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_3_2->addWidget( m_edFreq );

    textLabel2_3_2_3_2 = new QLabel( centralWidget(), "textLabel2_3_2_3_2" );
    textLabel2_3_2_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_3_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_3_2->addWidget( textLabel2_3_2_3_2 );
    layout59_3_2->addLayout( layout1_3_2_3_2 );

    layout41->addLayout( layout59_3_2, 1, 0 );

    FrmSGLayout->addLayout( layout41, 0, 0 );
    spacer20 = new QSpacerItem( 141, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    FrmSGLayout->addItem( spacer20, 0, 1 );
    spacer21 = new QSpacerItem( 20, 161, QSizePolicy::Minimum, QSizePolicy::Expanding );
    FrmSGLayout->addItem( spacer21, 1, 0 );

    // toolbars

    languageChange();
    resize( QSize(160, 132).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmSG::~FrmSG()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmSG::languageChange()
{
    setCaption( tr2i18n( "Signal Generator Control" ) );
    m_ckbAMON->setText( tr2i18n( "AM ON" ) );
    textLabel5_3->setText( tr2i18n( "Output Level" ) );
    textLabel2_3_2_3->setText( tr2i18n( "dBm" ) );
    m_ckbFMON->setText( tr2i18n( "FM ON" ) );
    textLabel5_3_2->setText( tr2i18n( "Frequency" ) );
    textLabel2_3_2_3_2->setText( tr2i18n( "MHz" ) );
}

#include "signalgeneratorform.moc"
