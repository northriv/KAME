#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../../kame/thermometer/forms/caltableform.ui'
**
** Created: 水  2 1 03:42:18 2006
**      by: The User Interface Compiler ($Id: caltableform.cpp,v 1.1 2006/02/01 18:45:19 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "caltableform.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <kcombobox.h>
#include <qlineedit.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>

/*
 *  Constructs a FrmCalTable as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 */
FrmCalTable::FrmCalTable( QWidget* parent, const char* name, WFlags fl )
    : QWidget( parent, name, fl )
{
    if ( !name )
	setName( "FrmCalTable" );
    FrmCalTableLayout = new QGridLayout( this, 1, 1, 2, 6, "FrmCalTableLayout"); 

    layout8 = new QGridLayout( 0, 1, 1, 0, 6, "layout8"); 

    btnDump = new QPushButton( this, "btnDump" );
    btnDump->setAcceptDrops( FALSE );
    btnDump->setAutoDefault( FALSE );

    layout8->addWidget( btnDump, 2, 0 );

    layout3 = new QVBoxLayout( 0, 0, 6, "layout3"); 

    textLabel1 = new QLabel( this, "textLabel1" );
    layout3->addWidget( textLabel1 );

    cmbThermometer = new KComboBox( FALSE, this, "cmbThermometer" );
    cmbThermometer->setAcceptDrops( FALSE );
    layout3->addWidget( cmbThermometer );

    layout8->addMultiCellLayout( layout3, 0, 0, 0, 2 );

    layout8_2 = new QVBoxLayout( 0, 0, 6, "layout8_2"); 

    textLabel3 = new QLabel( this, "textLabel3" );
    layout8_2->addWidget( textLabel3 );

    edValue = new QLineEdit( this, "edValue" );
    layout8_2->addWidget( edValue );

    layout8->addLayout( layout8_2, 1, 2 );
    spacer3 = new QSpacerItem( 16, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    layout8->addItem( spacer3, 1, 1 );
    spacer5 = new QSpacerItem( 16, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    layout8->addItem( spacer5, 2, 1 );

    layout10 = new QVBoxLayout( 0, 0, 6, "layout10"); 

    textLabel2 = new QLabel( this, "textLabel2" );
    layout10->addWidget( textLabel2 );

    layout9 = new QHBoxLayout( 0, 0, 6, "layout9"); 

    edTemp = new QLineEdit( this, "edTemp" );
    layout9->addWidget( edTemp );

    textLabel1_2 = new QLabel( this, "textLabel1_2" );
    layout9->addWidget( textLabel1_2 );
    layout10->addLayout( layout9 );

    layout8->addLayout( layout10, 1, 0 );

    FrmCalTableLayout->addLayout( layout8, 0, 0 );
    spacer5_2 = new QSpacerItem( 61, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    FrmCalTableLayout->addItem( spacer5_2, 0, 1 );
    spacer6 = new QSpacerItem( 20, 41, QSizePolicy::Minimum, QSizePolicy::Expanding );
    FrmCalTableLayout->addItem( spacer6, 1, 0 );
    languageChange();
    resize( QSize(238, 157).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // tab order
    setTabOrder( cmbThermometer, edTemp );
    setTabOrder( edTemp, edValue );
    setTabOrder( edValue, btnDump );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmCalTable::~FrmCalTable()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmCalTable::languageChange()
{
    setCaption( tr2i18n( "Calibration Table" ) );
    btnDump->setText( tr2i18n( "Dump Table" ) );
    textLabel1->setText( tr2i18n( "Thermometer" ) );
    textLabel3->setText( tr2i18n( "Value" ) );
    textLabel2->setText( tr2i18n( "Temp." ) );
    textLabel1_2->setText( tr2i18n( "K" ) );
}

void FrmCalTable::btnOK_clicked()
{
    qWarning( "FrmCalTable::btnOK_clicked(): Not implemented yet" );
}

#include "caltableform.moc"
