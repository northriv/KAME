#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../../../kame/users/nmr/forms/nmrsgcontrolform.ui'
**
** Created: Fri Jan 6 01:08:09 2006
**      by: The User Interface Compiler ($Id: nmrsgcontrolform.cpp,v 1.1 2006/02/01 18:43:58 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "nmrsgcontrolform.h"

#include <qvariant.h>
#include <qpushbutton.h>
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
 *  Constructs a frmNMRSGControl as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
frmNMRSGControl::frmNMRSGControl( QWidget* parent, const char* name, WFlags fl )
    : QMainWindow( parent, name, fl )
{
    (void)statusBar();
    if ( !name )
	setName( "frmNMRSGControl" );
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );
    frmNMRSGControlLayout = new QGridLayout( centralWidget(), 1, 1, 11, 6, "frmNMRSGControlLayout"); 

    layout59_3_2_2 = new QHBoxLayout( 0, 0, 6, "layout59_3_2_2"); 

    textLabel5_3_2_2 = new QLabel( centralWidget(), "textLabel5_3_2_2" );
    textLabel5_3_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_3_2_2->sizePolicy().hasHeightForWidth() ) );
    layout59_3_2_2->addWidget( textLabel5_3_2_2 );

    layout1_3_2_3_2_2 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_3_2_2"); 

    edSG1FreqOffset = new QLineEdit( centralWidget(), "edSG1FreqOffset" );
    edSG1FreqOffset->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, edSG1FreqOffset->sizePolicy().hasHeightForWidth() ) );
    edSG1FreqOffset->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_3_2_2->addWidget( edSG1FreqOffset );

    textLabel2_3_2_3_2_2 = new QLabel( centralWidget(), "textLabel2_3_2_3_2_2" );
    textLabel2_3_2_3_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_3_2_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_3_2_2->addWidget( textLabel2_3_2_3_2_2 );
    layout59_3_2_2->addLayout( layout1_3_2_3_2_2 );

    frmNMRSGControlLayout->addLayout( layout59_3_2_2, 1, 0 );

    layout59_3_2_3 = new QHBoxLayout( 0, 0, 6, "layout59_3_2_3"); 

    textLabel5_3_2_3 = new QLabel( centralWidget(), "textLabel5_3_2_3" );
    textLabel5_3_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_3_2_3->sizePolicy().hasHeightForWidth() ) );
    layout59_3_2_3->addWidget( textLabel5_3_2_3 );

    layout1_3_2_3_2_3 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_3_2_3"); 

    edSG2FreqOffset = new QLineEdit( centralWidget(), "edSG2FreqOffset" );
    edSG2FreqOffset->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, edSG2FreqOffset->sizePolicy().hasHeightForWidth() ) );
    edSG2FreqOffset->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_3_2_3->addWidget( edSG2FreqOffset );

    textLabel2_3_2_3_2_3 = new QLabel( centralWidget(), "textLabel2_3_2_3_2_3" );
    textLabel2_3_2_3_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_3_2_3->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_3_2_3->addWidget( textLabel2_3_2_3_2_3 );
    layout59_3_2_3->addLayout( layout1_3_2_3_2_3 );

    frmNMRSGControlLayout->addLayout( layout59_3_2_3, 2, 0 );

    layout59_3_2 = new QHBoxLayout( 0, 0, 6, "layout59_3_2"); 

    textLabel5_3_2 = new QLabel( centralWidget(), "textLabel5_3_2" );
    textLabel5_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_3_2->sizePolicy().hasHeightForWidth() ) );
    layout59_3_2->addWidget( textLabel5_3_2 );

    layout1_3_2_3_2 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_3_2"); 

    edFreq = new QLineEdit( centralWidget(), "edFreq" );
    edFreq->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, edFreq->sizePolicy().hasHeightForWidth() ) );
    edFreq->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_3_2->addWidget( edFreq );

    textLabel2_3_2_3_2 = new QLabel( centralWidget(), "textLabel2_3_2_3_2" );
    textLabel2_3_2_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_3_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_3_2->addWidget( textLabel2_3_2_3_2 );
    layout59_3_2->addLayout( layout1_3_2_3_2 );

    frmNMRSGControlLayout->addLayout( layout59_3_2, 0, 0 );

    // toolbars

    languageChange();
    resize( QSize(202, 173).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );
}

/*
 *  Destroys the object and frees any allocated resources
 */
frmNMRSGControl::~frmNMRSGControl()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void frmNMRSGControl::languageChange()
{
    setCaption( tr2i18n( "NMR Signal Generator Control" ) );
    textLabel5_3_2_2->setText( tr2i18n( "SG1 Freq. Offset" ) );
    textLabel2_3_2_3_2_2->setText( tr2i18n( "MHz" ) );
    textLabel5_3_2_3->setText( tr2i18n( "SG2 Freq. Offset" ) );
    textLabel2_3_2_3_2_3->setText( tr2i18n( "MHz" ) );
    textLabel5_3_2->setText( tr2i18n( "Rx Frequency" ) );
    textLabel2_3_2_3_2->setText( tr2i18n( "MHz" ) );
}

#include "nmrsgcontrolform.moc"
