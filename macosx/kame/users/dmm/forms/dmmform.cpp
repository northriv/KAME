#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../../../kame/users/dmm/forms/dmmform.ui'
**
** Created: 水  2 1 03:46:03 2006
**      by: The User Interface Compiler ($Id: dmmform.cpp,v 1.1 2006/02/01 18:45:42 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "dmmform.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qcombobox.h>
#include <qspinbox.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include <qaction.h>
#include <qmenubar.h>
#include <qpopupmenu.h>
#include <qtoolbar.h>

/*
 *  Constructs a FrmDMM as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
FrmDMM::FrmDMM( QWidget* parent, const char* name, WFlags fl )
    : QMainWindow( parent, name, fl )
{
    (void)statusBar();
    if ( !name )
	setName( "FrmDMM" );
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );
    FrmDMMLayout = new QGridLayout( centralWidget(), 1, 1, 2, 6, "FrmDMMLayout"); 

    layout4 = new QVBoxLayout( 0, 0, 6, "layout4"); 

    textLabel4 = new QLabel( centralWidget(), "textLabel4" );
    layout4->addWidget( textLabel4 );

    m_cmbFunction = new QComboBox( FALSE, centralWidget(), "m_cmbFunction" );
    layout4->addWidget( m_cmbFunction );

    FrmDMMLayout->addMultiCellLayout( layout4, 0, 0, 0, 1 );
    spacer2 = new QSpacerItem( 20, 16, QSizePolicy::Minimum, QSizePolicy::Expanding );
    FrmDMMLayout->addItem( spacer2, 1, 0 );

    layout23 = new QVBoxLayout( 0, 0, 6, "layout23"); 

    textLabel3 = new QLabel( centralWidget(), "textLabel3" );
    layout23->addWidget( textLabel3 );

    m_numWait = new QSpinBox( centralWidget(), "m_numWait" );
    m_numWait->setMaxValue( 99999 );
    layout23->addWidget( m_numWait );

    FrmDMMLayout->addMultiCellLayout( layout23, 2, 2, 0, 1 );
    spacer3 = new QSpacerItem( 20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding );
    FrmDMMLayout->addItem( spacer3, 3, 1 );

    // toolbars

    languageChange();
    resize( QSize(141, 145).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmDMM::~FrmDMM()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmDMM::languageChange()
{
    setCaption( tr2i18n( "DMM Settings" ) );
    textLabel4->setText( tr2i18n( "Function" ) );
    textLabel3->setText( tr2i18n( "Wait" ) );
    m_numWait->setSuffix( tr2i18n( " ms" ) );
}

#include "dmmform.moc"
