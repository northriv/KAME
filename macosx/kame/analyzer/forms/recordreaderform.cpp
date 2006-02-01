#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../../kame/analyzer/forms/recordreaderform.ui'
**
** Created: 水  2 1 04:00:08 2006
**      by: The User Interface Compiler ($Id: recordreaderform.cpp,v 1.1 2006/02/01 18:45:07 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "recordreaderform.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <kurlrequester.h>
#include <qlineedit.h>
#include <qcombobox.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>

/*
 *  Constructs a FrmRecordReader as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 */
FrmRecordReader::FrmRecordReader( QWidget* parent, const char* name, WFlags fl )
    : QWidget( parent, name, fl )
{
    if ( !name )
	setName( "FrmRecordReader" );
    setEnabled( TRUE );
    setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)5, 0, 0, sizePolicy().hasHeightForWidth() ) );
    FrmRecordReaderLayout = new QGridLayout( this, 1, 1, 2, 6, "FrmRecordReaderLayout"); 
    spacer6 = new QSpacerItem( 20, 16, QSizePolicy::Minimum, QSizePolicy::Expanding );
    FrmRecordReaderLayout->addItem( spacer6, 1, 0 );
    spacer4 = new QSpacerItem( 16, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    FrmRecordReaderLayout->addItem( spacer4, 0, 1 );

    layout12 = new QGridLayout( 0, 1, 1, 0, 6, "layout12"); 
    spacer2 = new QSpacerItem( 16, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    layout12->addItem( spacer2, 1, 0 );

    layout1 = new QVBoxLayout( 0, 0, 6, "layout1"); 

    textLabel1_ = new QLabel( this, "textLabel1_" );
    layout1->addWidget( textLabel1_ );

    urlBinRec = new KURLRequester( this, "urlBinRec" );
    urlBinRec->setEnabled( FALSE );
    layout1->addWidget( urlBinRec );

    layout12->addMultiCellLayout( layout1, 0, 0, 0, 2 );

    layout6 = new QVBoxLayout( 0, 0, 6, "layout6"); 

    layout5 = new QHBoxLayout( 0, 0, 6, "layout5"); 

    textLabel3 = new QLabel( this, "textLabel3" );
    layout5->addWidget( textLabel3 );

    edTime = new QLineEdit( this, "edTime" );
    edTime->setEnabled( FALSE );
    layout5->addWidget( edTime );
    layout6->addLayout( layout5 );

    layout12->addMultiCellLayout( layout6, 2, 2, 0, 2 );
    spacer3 = new QSpacerItem( 88, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    layout12->addItem( spacer3, 1, 2 );

    layout11 = new QVBoxLayout( 0, 0, 6, "layout11"); 

    layout8 = new QVBoxLayout( 0, 0, 6, "layout8"); 

    layout2 = new QHBoxLayout( 0, 0, 6, "layout2"); 

    btnRW = new QPushButton( this, "btnRW" );
    btnRW->setEnabled( FALSE );
    btnRW->setToggleButton( TRUE );
    btnRW->setAutoDefault( FALSE );
    layout2->addWidget( btnRW );

    btnStop = new QPushButton( this, "btnStop" );
    btnStop->setEnabled( FALSE );
    btnStop->setAutoDefault( FALSE );
    layout2->addWidget( btnStop );

    btnFF = new QPushButton( this, "btnFF" );
    btnFF->setEnabled( FALSE );
    btnFF->setToggleButton( TRUE );
    btnFF->setAutoDefault( FALSE );
    layout2->addWidget( btnFF );
    layout8->addLayout( layout2 );

    layout3 = new QHBoxLayout( 0, 0, 6, "layout3"); 

    btnFirst = new QPushButton( this, "btnFirst" );
    btnFirst->setEnabled( FALSE );
    btnFirst->setAutoDefault( FALSE );
    layout3->addWidget( btnFirst );

    btnBack = new QPushButton( this, "btnBack" );
    btnBack->setEnabled( FALSE );
    btnBack->setAutoDefault( FALSE );
    layout3->addWidget( btnBack );

    btnNext = new QPushButton( this, "btnNext" );
    btnNext->setEnabled( FALSE );
    btnNext->setAutoDefault( FALSE );
    layout3->addWidget( btnNext );
    layout8->addLayout( layout3 );
    layout11->addLayout( layout8 );

    layout10 = new QHBoxLayout( 0, 0, 6, "layout10"); 

    textLabel2 = new QLabel( this, "textLabel2" );
    textLabel2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2->sizePolicy().hasHeightForWidth() ) );
    layout10->addWidget( textLabel2 );

    cmbSpeed = new QComboBox( FALSE, this, "cmbSpeed" );
    cmbSpeed->setEnabled( FALSE );
    cmbSpeed->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, cmbSpeed->sizePolicy().hasHeightForWidth() ) );
    layout10->addWidget( cmbSpeed );
    layout11->addLayout( layout10 );

    layout12->addLayout( layout11, 1, 1 );

    FrmRecordReaderLayout->addLayout( layout12, 0, 0 );
    languageChange();
    resize( QSize(278, 217).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // tab order
    setTabOrder( urlBinRec, btnRW );
    setTabOrder( btnRW, btnStop );
    setTabOrder( btnStop, btnFF );
    setTabOrder( btnFF, btnFirst );
    setTabOrder( btnFirst, btnBack );
    setTabOrder( btnBack, btnNext );
    setTabOrder( btnNext, cmbSpeed );
    setTabOrder( cmbSpeed, edTime );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmRecordReader::~FrmRecordReader()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmRecordReader::languageChange()
{
    setCaption( tr2i18n( "Raw Stream Reader" ) );
    textLabel1_->setText( tr2i18n( "Raw Stream Binary" ) );
    textLabel3->setText( tr2i18n( "Position" ) );
    btnRW->setText( tr2i18n( "RW" ) );
    btnStop->setText( tr2i18n( "STOP" ) );
    btnFF->setText( tr2i18n( "FF" ) );
    btnFirst->setText( tr2i18n( "FIRST" ) );
    btnBack->setText( tr2i18n( "BACK" ) );
    btnNext->setText( tr2i18n( "NEXT" ) );
    textLabel2->setText( tr2i18n( "Speed" ) );
    cmbSpeed->clear();
    cmbSpeed->insertItem( tr2i18n( "Fastest" ) );
    cmbSpeed->insertItem( tr2i18n( "Fast" ) );
    cmbSpeed->insertItem( tr2i18n( "Normal" ) );
    cmbSpeed->insertItem( tr2i18n( "Slow" ) );
    cmbSpeed->insertItem( tr2i18n( "Slowest" ) );
    cmbSpeed->setCurrentItem( 2 );
}

#include "recordreaderform.moc"
