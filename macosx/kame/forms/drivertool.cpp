#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../kame/forms/drivertool.ui'
**
** Created: 水  2 1 03:36:30 2006
**      by: The User Interface Compiler ($Id: drivertool.cpp,v 1.1 2006/02/01 18:45:13 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "drivertool.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qgroupbox.h>
#include <kurlrequester.h>
#include <qcheckbox.h>
#include <qtable.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>

/*
 *  Constructs a FrmDriver as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 */
FrmDriver::FrmDriver( QWidget* parent, const char* name, WFlags fl )
    : QWidget( parent, name, fl )
{
    if ( !name )
	setName( "FrmDriver" );
    FrmDriverLayout = new QGridLayout( this, 1, 1, 2, 6, "FrmDriverLayout"); 

    groupBox1 = new QGroupBox( this, "groupBox1" );
    groupBox1->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, groupBox1->sizePolicy().hasHeightForWidth() ) );
    groupBox1->setFrameShape( QGroupBox::GroupBoxPanel );
    groupBox1->setFrameShadow( QGroupBox::Sunken );
    groupBox1->setColumnLayout(0, Qt::Vertical );
    groupBox1->layout()->setSpacing( 6 );
    groupBox1->layout()->setMargin( 11 );
    groupBox1Layout = new QGridLayout( groupBox1->layout() );
    groupBox1Layout->setAlignment( Qt::AlignTop );

    m_urlBinRec = new KURLRequester( groupBox1, "m_urlBinRec" );
    m_urlBinRec->setEnabled( FALSE );

    groupBox1Layout->addWidget( m_urlBinRec, 0, 0 );

    m_ckbBinRecWrite = new QCheckBox( groupBox1, "m_ckbBinRecWrite" );
    m_ckbBinRecWrite->setEnabled( FALSE );

    groupBox1Layout->addWidget( m_ckbBinRecWrite, 0, 2 );
    spacer4 = new QSpacerItem( 16, 20, QSizePolicy::Fixed, QSizePolicy::Minimum );
    groupBox1Layout->addItem( spacer4, 0, 1 );

    FrmDriverLayout->addMultiCellWidget( groupBox1, 0, 0, 0, 2 );

    m_tblDrivers = new QTable( this, "m_tblDrivers" );
    m_tblDrivers->setEnabled( FALSE );
    m_tblDrivers->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)5, 0, 0, m_tblDrivers->sizePolicy().hasHeightForWidth() ) );
    m_tblDrivers->setMinimumSize( QSize( 300, 300 ) );
    m_tblDrivers->setNumRows( 0 );
    m_tblDrivers->setNumCols( 0 );
    m_tblDrivers->setSorting( FALSE );
    m_tblDrivers->setSelectionMode( QTable::Single );

    FrmDriverLayout->addMultiCellWidget( m_tblDrivers, 1, 1, 0, 2 );
    spacer2 = new QSpacerItem( 60, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    FrmDriverLayout->addItem( spacer2, 2, 0 );

    m_btnNew = new QPushButton( this, "m_btnNew" );

    FrmDriverLayout->addWidget( m_btnNew, 2, 1 );

    m_btnDelete = new QPushButton( this, "m_btnDelete" );
    m_btnDelete->setEnabled( FALSE );

    FrmDriverLayout->addWidget( m_btnDelete, 2, 2 );
    languageChange();
    resize( QSize(304, 402).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmDriver::~FrmDriver()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmDriver::languageChange()
{
    setCaption( tr2i18n( "Drivers" ) );
    groupBox1->setTitle( tr2i18n( "Raw Stream Recorder" ) );
    m_ckbBinRecWrite->setText( tr2i18n( "Write" ) );
    m_btnNew->setText( tr2i18n( "New Driver" ) );
    m_btnDelete->setText( tr2i18n( "Delete Driver" ) );
}

#include "drivertool.moc"
