#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../../../kame/users/nmr/forms/pulserdrivermoreform.ui'
**
** Created: 水  2 1 03:51:44 2006
**      by: The User Interface Compiler ($Id: pulserdrivermoreform.cpp,v 1.1 2006/02/01 18:43:54 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "pulserdrivermoreform.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qcombobox.h>
#include <qspinbox.h>
#include <qlineedit.h>
#include <qcheckbox.h>
#include <qgroupbox.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include <qaction.h>
#include <qmenubar.h>
#include <qpopupmenu.h>
#include <qtoolbar.h>

/*
 *  Constructs a FrmPulserMore as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
FrmPulserMore::FrmPulserMore( QWidget* parent, const char* name, WFlags fl )
    : QMainWindow( parent, name, fl )
{
    (void)statusBar();
    if ( !name )
	setName( "FrmPulserMore" );
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );
    FrmPulserMoreLayout = new QGridLayout( centralWidget(), 1, 1, 2, 6, "FrmPulserMoreLayout"); 

    layout85_3_2 = new QHBoxLayout( 0, 0, 6, "layout85_3_2"); 

    textLabel2_3_3 = new QLabel( centralWidget(), "textLabel2_3_3" );
    layout85_3_2->addWidget( textLabel2_3_3 );

    m_cmbPhaseCycle = new QComboBox( FALSE, centralWidget(), "m_cmbPhaseCycle" );
    layout85_3_2->addWidget( m_cmbPhaseCycle );

    FrmPulserMoreLayout->addLayout( layout85_3_2, 0, 0 );

    layout71_2 = new QHBoxLayout( 0, 0, 6, "layout71_2"); 

    textLabel5_2_3_3_2 = new QLabel( centralWidget(), "textLabel5_2_3_3_2" );
    textLabel5_2_3_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_3_2->sizePolicy().hasHeightForWidth() ) );
    layout71_2->addWidget( textLabel5_2_3_3_2 );

    m_numEcho = new QSpinBox( centralWidget(), "m_numEcho" );
    m_numEcho->setMinValue( 1 );
    layout71_2->addWidget( m_numEcho );

    FrmPulserMoreLayout->addLayout( layout71_2, 1, 0 );

    layout59_3_2 = new QHBoxLayout( 0, 0, 6, "layout59_3_2"); 

    textLabel5_3_2 = new QLabel( centralWidget(), "textLabel5_3_2" );
    textLabel5_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_3_2->sizePolicy().hasHeightForWidth() ) );
    layout59_3_2->addWidget( textLabel5_3_2 );

    layout1_3_2_3_2 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_3_2"); 

    m_edG2Setup = new QLineEdit( centralWidget(), "m_edG2Setup" );
    m_edG2Setup->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edG2Setup->sizePolicy().hasHeightForWidth() ) );
    m_edG2Setup->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_3_2->addWidget( m_edG2Setup );

    textLabel2_3_2_3_2 = new QLabel( centralWidget(), "textLabel2_3_2_3_2" );
    textLabel2_3_2_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_3_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_3_2->addWidget( textLabel2_3_2_3_2 );
    layout59_3_2->addLayout( layout1_3_2_3_2 );

    FrmPulserMoreLayout->addLayout( layout59_3_2, 4, 0 );

    m_ckbDrivenEquilibrium = new QCheckBox( centralWidget(), "m_ckbDrivenEquilibrium" );
    m_ckbDrivenEquilibrium->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckbDrivenEquilibrium->sizePolicy().hasHeightForWidth() ) );

    FrmPulserMoreLayout->addMultiCellWidget( m_ckbDrivenEquilibrium, 2, 3, 0, 0 );

    groupBox9 = new QGroupBox( centralWidget(), "groupBox9" );
    groupBox9->setColumnLayout(0, Qt::Vertical );
    groupBox9->layout()->setSpacing( 6 );
    groupBox9->layout()->setMargin( 11 );
    groupBox9Layout = new QGridLayout( groupBox9->layout() );
    groupBox9Layout->setAlignment( Qt::AlignTop );

    layout59_2_3_2_4_2 = new QHBoxLayout( 0, 0, 6, "layout59_2_3_2_4_2"); 

    textLabel5_2_3_2_4_2 = new QLabel( groupBox9, "textLabel5_2_3_2_4_2" );
    textLabel5_2_3_2_4_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_2_4_2->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3_2_4_2->addWidget( textLabel5_2_3_2_4_2 );

    layout1_3_2_2_3_2_4_2 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3_2_4_2"); 

    m_edPortLevel8 = new QLineEdit( groupBox9, "m_edPortLevel8" );
    m_edPortLevel8->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edPortLevel8->sizePolicy().hasHeightForWidth() ) );
    m_edPortLevel8->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3_2_4_2->addWidget( m_edPortLevel8 );

    textLabel2_3_2_2_3_2_4_2 = new QLabel( groupBox9, "textLabel2_3_2_2_3_2_4_2" );
    textLabel2_3_2_2_3_2_4_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2_3_2_4_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2_3_2_4_2->addWidget( textLabel2_3_2_2_3_2_4_2 );
    layout59_2_3_2_4_2->addLayout( layout1_3_2_2_3_2_4_2 );

    groupBox9Layout->addLayout( layout59_2_3_2_4_2, 0, 0 );

    layout59_2_3_2_4_2_2_2 = new QHBoxLayout( 0, 0, 6, "layout59_2_3_2_4_2_2_2"); 

    textLabel5_2_3_2_4_2_2_2 = new QLabel( groupBox9, "textLabel5_2_3_2_4_2_2_2" );
    textLabel5_2_3_2_4_2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_2_4_2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3_2_4_2_2_2->addWidget( textLabel5_2_3_2_4_2_2_2 );

    layout1_3_2_2_3_2_4_2_2_2 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3_2_4_2_2_2"); 

    m_edPortLevel10 = new QLineEdit( groupBox9, "m_edPortLevel10" );
    m_edPortLevel10->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edPortLevel10->sizePolicy().hasHeightForWidth() ) );
    m_edPortLevel10->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3_2_4_2_2_2->addWidget( m_edPortLevel10 );

    textLabel2_3_2_2_3_2_4_2_2_2 = new QLabel( groupBox9, "textLabel2_3_2_2_3_2_4_2_2_2" );
    textLabel2_3_2_2_3_2_4_2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2_3_2_4_2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2_3_2_4_2_2_2->addWidget( textLabel2_3_2_2_3_2_4_2_2_2 );
    layout59_2_3_2_4_2_2_2->addLayout( layout1_3_2_2_3_2_4_2_2_2 );

    groupBox9Layout->addLayout( layout59_2_3_2_4_2_2_2, 2, 0 );

    layout59_2_3_2_4_2_2 = new QHBoxLayout( 0, 0, 6, "layout59_2_3_2_4_2_2"); 

    textLabel5_2_3_2_4_2_2 = new QLabel( groupBox9, "textLabel5_2_3_2_4_2_2" );
    textLabel5_2_3_2_4_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_2_4_2_2->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3_2_4_2_2->addWidget( textLabel5_2_3_2_4_2_2 );

    layout1_3_2_2_3_2_4_2_2 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3_2_4_2_2"); 

    m_edPortLevel9 = new QLineEdit( groupBox9, "m_edPortLevel9" );
    m_edPortLevel9->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edPortLevel9->sizePolicy().hasHeightForWidth() ) );
    m_edPortLevel9->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3_2_4_2_2->addWidget( m_edPortLevel9 );

    textLabel2_3_2_2_3_2_4_2_2 = new QLabel( groupBox9, "textLabel2_3_2_2_3_2_4_2_2" );
    textLabel2_3_2_2_3_2_4_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2_3_2_4_2_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2_3_2_4_2_2->addWidget( textLabel2_3_2_2_3_2_4_2_2 );
    layout59_2_3_2_4_2_2->addLayout( layout1_3_2_2_3_2_4_2_2 );

    groupBox9Layout->addLayout( layout59_2_3_2_4_2_2, 1, 0 );

    layout59_2_3_2_4_2_2_3 = new QHBoxLayout( 0, 0, 6, "layout59_2_3_2_4_2_2_3"); 

    textLabel5_2_3_2_4_2_2_3 = new QLabel( groupBox9, "textLabel5_2_3_2_4_2_2_3" );
    textLabel5_2_3_2_4_2_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_2_4_2_2_3->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3_2_4_2_2_3->addWidget( textLabel5_2_3_2_4_2_2_3 );

    layout1_3_2_2_3_2_4_2_2_3 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3_2_4_2_2_3"); 

    m_edPortLevel11 = new QLineEdit( groupBox9, "m_edPortLevel11" );
    m_edPortLevel11->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edPortLevel11->sizePolicy().hasHeightForWidth() ) );
    m_edPortLevel11->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3_2_4_2_2_3->addWidget( m_edPortLevel11 );

    textLabel2_3_2_2_3_2_4_2_2_3 = new QLabel( groupBox9, "textLabel2_3_2_2_3_2_4_2_2_3" );
    textLabel2_3_2_2_3_2_4_2_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2_3_2_4_2_2_3->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2_3_2_4_2_2_3->addWidget( textLabel2_3_2_2_3_2_4_2_2_3 );
    layout59_2_3_2_4_2_2_3->addLayout( layout1_3_2_2_3_2_4_2_2_3 );

    groupBox9Layout->addLayout( layout59_2_3_2_4_2_2_3, 3, 0 );

    layout59_2_3_2_4_2_2_4 = new QHBoxLayout( 0, 0, 6, "layout59_2_3_2_4_2_2_4"); 

    textLabel5_2_3_2_4_2_2_4 = new QLabel( groupBox9, "textLabel5_2_3_2_4_2_2_4" );
    textLabel5_2_3_2_4_2_2_4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_2_4_2_2_4->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3_2_4_2_2_4->addWidget( textLabel5_2_3_2_4_2_2_4 );

    layout1_3_2_2_3_2_4_2_2_4 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3_2_4_2_2_4"); 

    m_edPortLevel12 = new QLineEdit( groupBox9, "m_edPortLevel12" );
    m_edPortLevel12->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edPortLevel12->sizePolicy().hasHeightForWidth() ) );
    m_edPortLevel12->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3_2_4_2_2_4->addWidget( m_edPortLevel12 );

    textLabel2_3_2_2_3_2_4_2_2_4 = new QLabel( groupBox9, "textLabel2_3_2_2_3_2_4_2_2_4" );
    textLabel2_3_2_2_3_2_4_2_2_4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2_3_2_4_2_2_4->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2_3_2_4_2_2_4->addWidget( textLabel2_3_2_2_3_2_4_2_2_4 );
    layout59_2_3_2_4_2_2_4->addLayout( layout1_3_2_2_3_2_4_2_2_4 );

    groupBox9Layout->addLayout( layout59_2_3_2_4_2_2_4, 4, 0 );

    layout59_2_3_2_4_2_2_5 = new QHBoxLayout( 0, 0, 6, "layout59_2_3_2_4_2_2_5"); 

    textLabel5_2_3_2_4_2_2_5 = new QLabel( groupBox9, "textLabel5_2_3_2_4_2_2_5" );
    textLabel5_2_3_2_4_2_2_5->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_2_4_2_2_5->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3_2_4_2_2_5->addWidget( textLabel5_2_3_2_4_2_2_5 );

    layout1_3_2_2_3_2_4_2_2_5 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3_2_4_2_2_5"); 

    m_edPortLevel13 = new QLineEdit( groupBox9, "m_edPortLevel13" );
    m_edPortLevel13->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edPortLevel13->sizePolicy().hasHeightForWidth() ) );
    m_edPortLevel13->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3_2_4_2_2_5->addWidget( m_edPortLevel13 );

    textLabel2_3_2_2_3_2_4_2_2_5 = new QLabel( groupBox9, "textLabel2_3_2_2_3_2_4_2_2_5" );
    textLabel2_3_2_2_3_2_4_2_2_5->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2_3_2_4_2_2_5->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2_3_2_4_2_2_5->addWidget( textLabel2_3_2_2_3_2_4_2_2_5 );
    layout59_2_3_2_4_2_2_5->addLayout( layout1_3_2_2_3_2_4_2_2_5 );

    groupBox9Layout->addLayout( layout59_2_3_2_4_2_2_5, 5, 0 );

    layout59_2_3_2_4_2_2_2_2 = new QHBoxLayout( 0, 0, 6, "layout59_2_3_2_4_2_2_2_2"); 

    textLabel5_2_3_2_4_2_2_2_2 = new QLabel( groupBox9, "textLabel5_2_3_2_4_2_2_2_2" );
    textLabel5_2_3_2_4_2_2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_2_4_2_2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3_2_4_2_2_2_2->addWidget( textLabel5_2_3_2_4_2_2_2_2 );

    layout1_3_2_2_3_2_4_2_2_2_2 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3_2_4_2_2_2_2"); 

    m_edPortLevel14 = new QLineEdit( groupBox9, "m_edPortLevel14" );
    m_edPortLevel14->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edPortLevel14->sizePolicy().hasHeightForWidth() ) );
    m_edPortLevel14->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3_2_4_2_2_2_2->addWidget( m_edPortLevel14 );

    textLabel2_3_2_2_3_2_4_2_2_2_2 = new QLabel( groupBox9, "textLabel2_3_2_2_3_2_4_2_2_2_2" );
    textLabel2_3_2_2_3_2_4_2_2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2_3_2_4_2_2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2_3_2_4_2_2_2_2->addWidget( textLabel2_3_2_2_3_2_4_2_2_2_2 );
    layout59_2_3_2_4_2_2_2_2->addLayout( layout1_3_2_2_3_2_4_2_2_2_2 );

    groupBox9Layout->addLayout( layout59_2_3_2_4_2_2_2_2, 6, 0 );

    FrmPulserMoreLayout->addWidget( groupBox9, 7, 1 );

    groupBox3_2_3 = new QGroupBox( centralWidget(), "groupBox3_2_3" );
    groupBox3_2_3->setColumnLayout(0, Qt::Vertical );
    groupBox3_2_3->layout()->setSpacing( 6 );
    groupBox3_2_3->layout()->setMargin( 11 );
    groupBox3_2_3Layout = new QGridLayout( groupBox3_2_3->layout() );
    groupBox3_2_3Layout->setAlignment( Qt::AlignTop );

    layout59_2_3_2_4_2_2_2_2_2_3 = new QHBoxLayout( 0, 0, 6, "layout59_2_3_2_4_2_2_2_2_2_3"); 

    textLabel5_2_3_2_4_2_2_2_2_2_3 = new QLabel( groupBox3_2_3, "textLabel5_2_3_2_4_2_2_2_2_2_3" );
    textLabel5_2_3_2_4_2_2_2_2_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_2_4_2_2_2_2_2_3->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3_2_4_2_2_2_2_2_3->addWidget( textLabel5_2_3_2_4_2_2_2_2_2_3 );

    layout1_3_2_2_3_2_4_2_2_2_2_2_3 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3_2_4_2_2_2_2_2_3"); 

    m_edQAMLevel1 = new QLineEdit( groupBox3_2_3, "m_edQAMLevel1" );
    m_edQAMLevel1->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edQAMLevel1->sizePolicy().hasHeightForWidth() ) );
    m_edQAMLevel1->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3_2_4_2_2_2_2_2_3->addWidget( m_edQAMLevel1 );
    layout59_2_3_2_4_2_2_2_2_2_3->addLayout( layout1_3_2_2_3_2_4_2_2_2_2_2_3 );

    groupBox3_2_3Layout->addLayout( layout59_2_3_2_4_2_2_2_2_2_3, 0, 0 );

    layout59_2_3_2_4_2_2_2_2_2_2_3 = new QHBoxLayout( 0, 0, 6, "layout59_2_3_2_4_2_2_2_2_2_2_3"); 

    textLabel5_2_3_2_4_2_2_2_2_2_2_3 = new QLabel( groupBox3_2_3, "textLabel5_2_3_2_4_2_2_2_2_2_2_3" );
    textLabel5_2_3_2_4_2_2_2_2_2_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_2_4_2_2_2_2_2_2_3->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3_2_4_2_2_2_2_2_2_3->addWidget( textLabel5_2_3_2_4_2_2_2_2_2_2_3 );

    layout1_3_2_2_3_2_4_2_2_2_2_2_2_3 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3_2_4_2_2_2_2_2_2_3"); 

    m_edQAMLevel2 = new QLineEdit( groupBox3_2_3, "m_edQAMLevel2" );
    m_edQAMLevel2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edQAMLevel2->sizePolicy().hasHeightForWidth() ) );
    m_edQAMLevel2->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3_2_4_2_2_2_2_2_2_3->addWidget( m_edQAMLevel2 );
    layout59_2_3_2_4_2_2_2_2_2_2_3->addLayout( layout1_3_2_2_3_2_4_2_2_2_2_2_2_3 );

    groupBox3_2_3Layout->addLayout( layout59_2_3_2_4_2_2_2_2_2_2_3, 1, 0 );

    FrmPulserMoreLayout->addMultiCellWidget( groupBox3_2_3, 3, 5, 1, 1 );

    groupBox3_2 = new QGroupBox( centralWidget(), "groupBox3_2" );
    groupBox3_2->setColumnLayout(0, Qt::Vertical );
    groupBox3_2->layout()->setSpacing( 6 );
    groupBox3_2->layout()->setMargin( 11 );
    groupBox3_2Layout = new QGridLayout( groupBox3_2->layout() );
    groupBox3_2Layout->setAlignment( Qt::AlignTop );

    layout59_2_3_2_4_2_2_2_2_2 = new QHBoxLayout( 0, 0, 6, "layout59_2_3_2_4_2_2_2_2_2"); 

    textLabel5_2_3_2_4_2_2_2_2_2 = new QLabel( groupBox3_2, "textLabel5_2_3_2_4_2_2_2_2_2" );
    textLabel5_2_3_2_4_2_2_2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_2_4_2_2_2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3_2_4_2_2_2_2_2->addWidget( textLabel5_2_3_2_4_2_2_2_2_2 );

    layout1_3_2_2_3_2_4_2_2_2_2_2 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3_2_4_2_2_2_2_2"); 

    m_edQAMOffset1 = new QLineEdit( groupBox3_2, "m_edQAMOffset1" );
    m_edQAMOffset1->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edQAMOffset1->sizePolicy().hasHeightForWidth() ) );
    m_edQAMOffset1->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3_2_4_2_2_2_2_2->addWidget( m_edQAMOffset1 );

    textLabel2_3_2_2_3_2_4_2_2_2_2_2 = new QLabel( groupBox3_2, "textLabel2_3_2_2_3_2_4_2_2_2_2_2" );
    textLabel2_3_2_2_3_2_4_2_2_2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2_3_2_4_2_2_2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2_3_2_4_2_2_2_2_2->addWidget( textLabel2_3_2_2_3_2_4_2_2_2_2_2 );
    layout59_2_3_2_4_2_2_2_2_2->addLayout( layout1_3_2_2_3_2_4_2_2_2_2_2 );

    groupBox3_2Layout->addLayout( layout59_2_3_2_4_2_2_2_2_2, 0, 0 );

    layout59_2_3_2_4_2_2_2_2_2_2 = new QHBoxLayout( 0, 0, 6, "layout59_2_3_2_4_2_2_2_2_2_2"); 

    textLabel5_2_3_2_4_2_2_2_2_2_2 = new QLabel( groupBox3_2, "textLabel5_2_3_2_4_2_2_2_2_2_2" );
    textLabel5_2_3_2_4_2_2_2_2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_2_4_2_2_2_2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3_2_4_2_2_2_2_2_2->addWidget( textLabel5_2_3_2_4_2_2_2_2_2_2 );

    layout1_3_2_2_3_2_4_2_2_2_2_2_2 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3_2_4_2_2_2_2_2_2"); 

    m_edQAMOffset2 = new QLineEdit( groupBox3_2, "m_edQAMOffset2" );
    m_edQAMOffset2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edQAMOffset2->sizePolicy().hasHeightForWidth() ) );
    m_edQAMOffset2->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3_2_4_2_2_2_2_2_2->addWidget( m_edQAMOffset2 );

    textLabel2_3_2_2_3_2_4_2_2_2_2_2_2 = new QLabel( groupBox3_2, "textLabel2_3_2_2_3_2_4_2_2_2_2_2_2" );
    textLabel2_3_2_2_3_2_4_2_2_2_2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2_3_2_4_2_2_2_2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2_3_2_4_2_2_2_2_2_2->addWidget( textLabel2_3_2_2_3_2_4_2_2_2_2_2_2 );
    layout59_2_3_2_4_2_2_2_2_2_2->addLayout( layout1_3_2_2_3_2_4_2_2_2_2_2_2 );

    groupBox3_2Layout->addLayout( layout59_2_3_2_4_2_2_2_2_2_2, 1, 0 );

    FrmPulserMoreLayout->addMultiCellWidget( groupBox3_2, 0, 2, 1, 1 );

    groupBox3_2_4 = new QGroupBox( centralWidget(), "groupBox3_2_4" );
    groupBox3_2_4->setColumnLayout(0, Qt::Vertical );
    groupBox3_2_4->layout()->setSpacing( 6 );
    groupBox3_2_4->layout()->setMargin( 11 );
    groupBox3_2_4Layout = new QGridLayout( groupBox3_2_4->layout() );
    groupBox3_2_4Layout->setAlignment( Qt::AlignTop );

    layout59_2_3_2_4_2_2_2_2_2_4 = new QHBoxLayout( 0, 0, 6, "layout59_2_3_2_4_2_2_2_2_2_4"); 

    textLabel5_2_3_2_4_2_2_2_2_2_4 = new QLabel( groupBox3_2_4, "textLabel5_2_3_2_4_2_2_2_2_2_4" );
    textLabel5_2_3_2_4_2_2_2_2_2_4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_2_4_2_2_2_2_2_4->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3_2_4_2_2_2_2_2_4->addWidget( textLabel5_2_3_2_4_2_2_2_2_2_4 );

    layout1_3_2_2_3_2_4_2_2_2_2_2_4 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3_2_4_2_2_2_2_2_4"); 

    m_edQAMDelay1 = new QLineEdit( groupBox3_2_4, "m_edQAMDelay1" );
    m_edQAMDelay1->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edQAMDelay1->sizePolicy().hasHeightForWidth() ) );
    m_edQAMDelay1->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3_2_4_2_2_2_2_2_4->addWidget( m_edQAMDelay1 );

    textLabel2_3_2_2_3_2_4_2_2_2_2_2_3 = new QLabel( groupBox3_2_4, "textLabel2_3_2_2_3_2_4_2_2_2_2_2_3" );
    textLabel2_3_2_2_3_2_4_2_2_2_2_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2_3_2_4_2_2_2_2_2_3->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2_3_2_4_2_2_2_2_2_4->addWidget( textLabel2_3_2_2_3_2_4_2_2_2_2_2_3 );
    layout59_2_3_2_4_2_2_2_2_2_4->addLayout( layout1_3_2_2_3_2_4_2_2_2_2_2_4 );

    groupBox3_2_4Layout->addLayout( layout59_2_3_2_4_2_2_2_2_2_4, 0, 0 );

    layout59_2_3_2_4_2_2_2_2_2_2_4 = new QHBoxLayout( 0, 0, 6, "layout59_2_3_2_4_2_2_2_2_2_2_4"); 

    textLabel5_2_3_2_4_2_2_2_2_2_2_4 = new QLabel( groupBox3_2_4, "textLabel5_2_3_2_4_2_2_2_2_2_2_4" );
    textLabel5_2_3_2_4_2_2_2_2_2_2_4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_2_4_2_2_2_2_2_2_4->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3_2_4_2_2_2_2_2_2_4->addWidget( textLabel5_2_3_2_4_2_2_2_2_2_2_4 );

    layout1_3_2_2_3_2_4_2_2_2_2_2_2_4 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3_2_4_2_2_2_2_2_2_4"); 

    m_edQAMDelay2 = new QLineEdit( groupBox3_2_4, "m_edQAMDelay2" );
    m_edQAMDelay2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edQAMDelay2->sizePolicy().hasHeightForWidth() ) );
    m_edQAMDelay2->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3_2_4_2_2_2_2_2_2_4->addWidget( m_edQAMDelay2 );

    textLabel2_3_2_2_3_2_4_2_2_2_2_2_2_3 = new QLabel( groupBox3_2_4, "textLabel2_3_2_2_3_2_4_2_2_2_2_2_2_3" );
    textLabel2_3_2_2_3_2_4_2_2_2_2_2_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2_3_2_4_2_2_2_2_2_2_3->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2_3_2_4_2_2_2_2_2_2_4->addWidget( textLabel2_3_2_2_3_2_4_2_2_2_2_2_2_3 );
    layout59_2_3_2_4_2_2_2_2_2_2_4->addLayout( layout1_3_2_2_3_2_4_2_2_2_2_2_2_4 );

    groupBox3_2_4Layout->addLayout( layout59_2_3_2_4_2_2_2_2_2_2_4, 1, 0 );

    FrmPulserMoreLayout->addWidget( groupBox3_2_4, 6, 1 );

    groupBox3 = new QGroupBox( centralWidget(), "groupBox3" );
    groupBox3->setColumnLayout(0, Qt::Vertical );
    groupBox3->layout()->setSpacing( 6 );
    groupBox3->layout()->setMargin( 11 );
    groupBox3Layout = new QGridLayout( groupBox3->layout() );
    groupBox3Layout->setAlignment( Qt::AlignTop );

    layout59_2_3_2_4 = new QHBoxLayout( 0, 0, 6, "layout59_2_3_2_4"); 

    textLabel5_2_3_2_4 = new QLabel( groupBox3, "textLabel5_2_3_2_4" );
    textLabel5_2_3_2_4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_2_4->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3_2_4->addWidget( textLabel5_2_3_2_4 );

    layout1_3_2_2_3_2_4 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3_2_4"); 

    m_edASWHold = new QLineEdit( groupBox3, "m_edASWHold" );
    m_edASWHold->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edASWHold->sizePolicy().hasHeightForWidth() ) );
    m_edASWHold->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3_2_4->addWidget( m_edASWHold );

    textLabel2_3_2_2_3_2_4 = new QLabel( groupBox3, "textLabel2_3_2_2_3_2_4" );
    textLabel2_3_2_2_3_2_4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2_3_2_4->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2_3_2_4->addWidget( textLabel2_3_2_2_3_2_4 );
    layout59_2_3_2_4->addLayout( layout1_3_2_2_3_2_4 );

    groupBox3Layout->addLayout( layout59_2_3_2_4, 0, 0 );

    layout59_2_3_2_5 = new QHBoxLayout( 0, 0, 6, "layout59_2_3_2_5"); 

    textLabel5_2_3_2_5 = new QLabel( groupBox3, "textLabel5_2_3_2_5" );
    textLabel5_2_3_2_5->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_2_5->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3_2_5->addWidget( textLabel5_2_3_2_5 );

    layout1_3_2_2_3_2_5 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3_2_5"); 

    m_edASWSetup = new QLineEdit( groupBox3, "m_edASWSetup" );
    m_edASWSetup->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edASWSetup->sizePolicy().hasHeightForWidth() ) );
    m_edASWSetup->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3_2_5->addWidget( m_edASWSetup );

    textLabel2_3_2_2_3_2_5 = new QLabel( groupBox3, "textLabel2_3_2_2_3_2_5" );
    textLabel2_3_2_2_3_2_5->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2_3_2_5->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2_3_2_5->addWidget( textLabel2_3_2_2_3_2_5 );
    layout59_2_3_2_5->addLayout( layout1_3_2_2_3_2_5 );

    groupBox3Layout->addLayout( layout59_2_3_2_5, 1, 0 );

    layout59_2_3_2_5_2 = new QHBoxLayout( 0, 0, 6, "layout59_2_3_2_5_2"); 

    textLabel5_2_3_2_5_2 = new QLabel( groupBox3, "textLabel5_2_3_2_5_2" );
    textLabel5_2_3_2_5_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_2_5_2->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3_2_5_2->addWidget( textLabel5_2_3_2_5_2 );

    layout1_3_2_2_3_2_5_2 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3_2_5_2"); 

    m_edALTSep = new QLineEdit( groupBox3, "m_edALTSep" );
    m_edALTSep->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edALTSep->sizePolicy().hasHeightForWidth() ) );
    m_edALTSep->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3_2_5_2->addWidget( m_edALTSep );

    textLabel2_3_2_2_3_2_5_2 = new QLabel( groupBox3, "textLabel2_3_2_2_3_2_5_2" );
    textLabel2_3_2_2_3_2_5_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2_3_2_5_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2_3_2_5_2->addWidget( textLabel2_3_2_2_3_2_5_2 );
    layout59_2_3_2_5_2->addLayout( layout1_3_2_2_3_2_5_2 );

    groupBox3Layout->addLayout( layout59_2_3_2_5_2, 2, 0 );

    layout85_2 = new QHBoxLayout( 0, 0, 6, "layout85_2"); 

    textLabel2_2 = new QLabel( groupBox3, "textLabel2_2" );
    layout85_2->addWidget( textLabel2_2 );

    m_cmbASWFilter = new QComboBox( FALSE, groupBox3, "m_cmbASWFilter" );
    layout85_2->addWidget( m_cmbASWFilter );

    groupBox3Layout->addLayout( layout85_2, 3, 0 );

    FrmPulserMoreLayout->addWidget( groupBox3, 7, 0 );

    layout59_3_2_2 = new QHBoxLayout( 0, 0, 6, "layout59_3_2_2"); 

    textLabel5_3_2_2 = new QLabel( centralWidget(), "textLabel5_3_2_2" );
    textLabel5_3_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_3_2_2->sizePolicy().hasHeightForWidth() ) );
    layout59_3_2_2->addWidget( textLabel5_3_2_2 );

    layout1_3_2_3_2_2 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_3_2_2"); 

    m_edDIFFreq = new QLineEdit( centralWidget(), "m_edDIFFreq" );
    m_edDIFFreq->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edDIFFreq->sizePolicy().hasHeightForWidth() ) );
    m_edDIFFreq->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_3_2_2->addWidget( m_edDIFFreq );

    textLabel2_3_2_3_2_2 = new QLabel( centralWidget(), "textLabel2_3_2_3_2_2" );
    textLabel2_3_2_3_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_3_2_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_3_2_2->addWidget( textLabel2_3_2_3_2_2 );
    layout59_3_2_2->addLayout( layout1_3_2_3_2_2 );

    FrmPulserMoreLayout->addMultiCellLayout( layout59_3_2_2, 5, 6, 0, 0 );

    // toolbars

    languageChange();
    resize( QSize(377, 574).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // tab order
    setTabOrder( m_cmbPhaseCycle, m_numEcho );
    setTabOrder( m_numEcho, m_ckbDrivenEquilibrium );
    setTabOrder( m_ckbDrivenEquilibrium, m_edG2Setup );
    setTabOrder( m_edG2Setup, m_edDIFFreq );
    setTabOrder( m_edDIFFreq, m_edASWHold );
    setTabOrder( m_edASWHold, m_edASWSetup );
    setTabOrder( m_edASWSetup, m_edALTSep );
    setTabOrder( m_edALTSep, m_cmbASWFilter );
    setTabOrder( m_cmbASWFilter, m_edQAMOffset1 );
    setTabOrder( m_edQAMOffset1, m_edQAMOffset2 );
    setTabOrder( m_edQAMOffset2, m_edQAMLevel1 );
    setTabOrder( m_edQAMLevel1, m_edQAMLevel2 );
    setTabOrder( m_edQAMLevel2, m_edQAMDelay1 );
    setTabOrder( m_edQAMDelay1, m_edQAMDelay2 );
    setTabOrder( m_edQAMDelay2, m_edPortLevel8 );
    setTabOrder( m_edPortLevel8, m_edPortLevel9 );
    setTabOrder( m_edPortLevel9, m_edPortLevel10 );
    setTabOrder( m_edPortLevel10, m_edPortLevel11 );
    setTabOrder( m_edPortLevel11, m_edPortLevel12 );
    setTabOrder( m_edPortLevel12, m_edPortLevel13 );
    setTabOrder( m_edPortLevel13, m_edPortLevel14 );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmPulserMore::~FrmPulserMore()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmPulserMore::languageChange()
{
    setCaption( tr2i18n( "Pulser Control" ) );
    textLabel2_3_3->setText( tr2i18n( "# in Phase Cycle" ) );
    textLabel5_2_3_3_2->setText( tr2i18n( "Multiple Echoes" ) );
    textLabel5_3_2->setText( tr2i18n( "Pre-Gate Setup Time" ) );
    textLabel2_3_2_3_2->setText( tr2i18n( "us" ) );
    m_ckbDrivenEquilibrium->setText( tr2i18n( "Driven Equilibrium" ) );
    groupBox9->setTitle( tr2i18n( "Analog Pulse Level" ) );
    textLabel5_2_3_2_4_2->setText( tr2i18n( "Port8" ) );
    textLabel2_3_2_2_3_2_4_2->setText( tr2i18n( "V" ) );
    textLabel5_2_3_2_4_2_2_2->setText( tr2i18n( "Port10" ) );
    textLabel2_3_2_2_3_2_4_2_2_2->setText( tr2i18n( "V" ) );
    textLabel5_2_3_2_4_2_2->setText( tr2i18n( "Port9" ) );
    textLabel2_3_2_2_3_2_4_2_2->setText( tr2i18n( "V" ) );
    textLabel5_2_3_2_4_2_2_3->setText( tr2i18n( "Port11" ) );
    textLabel2_3_2_2_3_2_4_2_2_3->setText( tr2i18n( "V" ) );
    textLabel5_2_3_2_4_2_2_4->setText( tr2i18n( "Port12" ) );
    textLabel2_3_2_2_3_2_4_2_2_4->setText( tr2i18n( "V" ) );
    textLabel5_2_3_2_4_2_2_5->setText( tr2i18n( "Port13" ) );
    textLabel2_3_2_2_3_2_4_2_2_5->setText( tr2i18n( "V" ) );
    textLabel5_2_3_2_4_2_2_2_2->setText( tr2i18n( "Port14" ) );
    textLabel2_3_2_2_3_2_4_2_2_2_2->setText( tr2i18n( "V" ) );
    groupBox3_2_3->setTitle( tr2i18n( "QAM Level Tuning" ) );
    textLabel5_2_3_2_4_2_2_2_2_2_3->setText( tr2i18n( "Port Q" ) );
    textLabel5_2_3_2_4_2_2_2_2_2_2_3->setText( tr2i18n( "Port I" ) );
    groupBox3_2->setTitle( tr2i18n( "QAM DC Offset" ) );
    textLabel5_2_3_2_4_2_2_2_2_2->setText( tr2i18n( "Port Q" ) );
    textLabel2_3_2_2_3_2_4_2_2_2_2_2->setText( tr2i18n( "%<font size=\"-1\">F.S.</font>" ) );
    textLabel5_2_3_2_4_2_2_2_2_2_2->setText( tr2i18n( "Port I" ) );
    textLabel2_3_2_2_3_2_4_2_2_2_2_2_2->setText( tr2i18n( "%<font size=\"-1\">F.S.</font>" ) );
    groupBox3_2_4->setTitle( tr2i18n( "QAM Delay Tuning" ) );
    textLabel5_2_3_2_4_2_2_2_2_2_4->setText( tr2i18n( "Port Q" ) );
    textLabel2_3_2_2_3_2_4_2_2_2_2_2_3->setText( tr2i18n( "us" ) );
    textLabel5_2_3_2_4_2_2_2_2_2_2_4->setText( tr2i18n( "Port I" ) );
    textLabel2_3_2_2_3_2_4_2_2_2_2_2_2_3->setText( tr2i18n( "us" ) );
    groupBox3->setTitle( tr2i18n( "Analog Switch" ) );
    textLabel5_2_3_2_4->setText( tr2i18n( "ASW Hold" ) );
    textLabel2_3_2_2_3_2_4->setText( tr2i18n( "ms" ) );
    textLabel5_2_3_2_5->setText( tr2i18n( "ASW Setup" ) );
    textLabel2_3_2_2_3_2_5->setText( tr2i18n( "ms" ) );
    textLabel5_2_3_2_5_2->setText( tr2i18n( "ALT Separation" ) );
    textLabel2_3_2_2_3_2_5_2->setText( tr2i18n( "ms" ) );
    textLabel2_2->setText( tr2i18n( "Filter" ) );
    textLabel5_3_2_2->setText( tr2i18n( "Digital IF Freq." ) );
    textLabel2_3_2_3_2_2->setText( tr2i18n( "MHz" ) );
}

#include "pulserdrivermoreform.moc"
