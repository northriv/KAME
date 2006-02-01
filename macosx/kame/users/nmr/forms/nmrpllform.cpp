#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../../../kame/users/nmr/forms/nmrpllform.ui'
**
** Created: 土  1 7 03:31:49 2006
**      by: The User Interface Compiler ($Id: nmrpllform.cpp,v 1.1 2006/02/01 18:43:58 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "nmrpllform.h"

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
 *  Constructs a frmNMRPLL as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
frmNMRPLL::frmNMRPLL( QWidget* parent, const char* name, WFlags fl )
    : QMainWindow( parent, name, fl )
{
    (void)statusBar();
    if ( !name )
	setName( "frmNMRPLL" );
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );
    frmNMRPLLLayout = new QGridLayout( centralWidget(), 1, 1, 11, 6, "frmNMRPLLLayout"); 

    ckbControl = new QCheckBox( centralWidget(), "ckbControl" );
    ckbControl->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, ckbControl->sizePolicy().hasHeightForWidth() ) );

    frmNMRPLLLayout->addWidget( ckbControl, 0, 0 );

    layout59_3_2_4 = new QHBoxLayout( 0, 0, 6, "layout59_3_2_4"); 

    textLabel5_3_2_4 = new QLabel( centralWidget(), "textLabel5_3_2_4" );
    textLabel5_3_2_4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_3_2_4->sizePolicy().hasHeightForWidth() ) );
    layout59_3_2_4->addWidget( textLabel5_3_2_4 );

    layout1_3_2_3_2_4 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_3_2_4"); 

    edWait4SG = new QLineEdit( centralWidget(), "edWait4SG" );
    edWait4SG->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, edWait4SG->sizePolicy().hasHeightForWidth() ) );
    edWait4SG->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_3_2_4->addWidget( edWait4SG );

    textLabel2_3_2_3_2_3 = new QLabel( centralWidget(), "textLabel2_3_2_3_2_3" );
    textLabel2_3_2_3_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_3_2_3->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_3_2_4->addWidget( textLabel2_3_2_3_2_3 );
    layout59_3_2_4->addLayout( layout1_3_2_3_2_4 );

    frmNMRPLLLayout->addLayout( layout59_3_2_4, 2, 0 );

    layout59_3_2_4_2 = new QHBoxLayout( 0, 0, 6, "layout59_3_2_4_2"); 

    textLabel5_3_2_4_2 = new QLabel( centralWidget(), "textLabel5_3_2_4_2" );
    textLabel5_3_2_4_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_3_2_4_2->sizePolicy().hasHeightForWidth() ) );
    layout59_3_2_4_2->addWidget( textLabel5_3_2_4_2 );

    layout1_3_2_3_2_4_2 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_3_2_4_2"); 

    eddphidf = new QLineEdit( centralWidget(), "eddphidf" );
    eddphidf->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, eddphidf->sizePolicy().hasHeightForWidth() ) );
    eddphidf->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_3_2_4_2->addWidget( eddphidf );

    textLabel2_3_2_3_2_3_2 = new QLabel( centralWidget(), "textLabel2_3_2_3_2_3_2" );
    textLabel2_3_2_3_2_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_3_2_3_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_3_2_4_2->addWidget( textLabel2_3_2_3_2_3_2 );
    layout59_3_2_4_2->addLayout( layout1_3_2_3_2_4_2 );

    frmNMRPLLLayout->addLayout( layout59_3_2_4_2, 1, 0 );

    // toolbars

    languageChange();
    resize( QSize(218, 116).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );
}

/*
 *  Destroys the object and frees any allocated resources
 */
frmNMRPLL::~frmNMRPLL()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void frmNMRPLL::languageChange()
{
    setCaption( tr2i18n( "NMR PLL (Phase Locked Loop)" ) );
    ckbControl->setText( tr2i18n( "Control" ) );
    textLabel5_3_2_4->setText( tr2i18n( "Wait for SG" ) );
    textLabel2_3_2_3_2_3->setText( tr2i18n( "ms" ) );
    textLabel5_3_2_4_2->setText( tr2i18n( "d Phi / d f" ) );
    textLabel2_3_2_3_2_3_2->setText( tr2i18n( "deg./MHz" ) );
}

#include "nmrpllform.moc"
