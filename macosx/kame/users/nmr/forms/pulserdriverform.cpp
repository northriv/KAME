#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../../../kame/users/nmr/forms/pulserdriverform.ui'
**
** Created: 水  2 1 03:51:24 2006
**      by: The User Interface Compiler ($Id: pulserdriverform.cpp,v 1.1 2006/02/01 18:43:55 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "pulserdriverform.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qcheckbox.h>
#include <qlabel.h>
#include <qcombobox.h>
#include <qlineedit.h>
#include <qgroupbox.h>
#include <knuminput.h>
#include <qtable.h>
#include <qspinbox.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include <qaction.h>
#include <qmenubar.h>
#include <qpopupmenu.h>
#include <qtoolbar.h>
#include "../../graph/graphwidget.h"

/*
 *  Constructs a FrmPulser as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
FrmPulser::FrmPulser( QWidget* parent, const char* name, WFlags fl )
    : QMainWindow( parent, name, fl )
{
    (void)statusBar();
    if ( !name )
	setName( "FrmPulser" );
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );
    FrmPulserLayout = new QGridLayout( centralWidget(), 1, 1, 2, 6, "FrmPulserLayout"); 

    m_graph = new XQGraph( centralWidget(), "m_graph" );
    m_graph->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)7, (QSizePolicy::SizeType)5, 0, 0, m_graph->sizePolicy().hasHeightForWidth() ) );

    FrmPulserLayout->addMultiCellWidget( m_graph, 0, 2, 2, 2 );

    layout52 = new QVBoxLayout( 0, 4, 2, "layout52"); 

    m_ckbOutput = new QCheckBox( centralWidget(), "m_ckbOutput" );
    m_ckbOutput->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckbOutput->sizePolicy().hasHeightForWidth() ) );
    layout52->addWidget( m_ckbOutput );

    layout85_2 = new QHBoxLayout( 0, 0, 6, "layout85_2"); 

    textLabel2_2 = new QLabel( centralWidget(), "textLabel2_2" );
    layout85_2->addWidget( textLabel2_2 );

    m_cmbRTMode = new QComboBox( FALSE, centralWidget(), "m_cmbRTMode" );
    layout85_2->addWidget( m_cmbRTMode );
    layout52->addLayout( layout85_2 );

    layout59_3 = new QHBoxLayout( 0, 0, 6, "layout59_3"); 

    textLabel5_3 = new QLabel( centralWidget(), "textLabel5_3" );
    textLabel5_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_3->sizePolicy().hasHeightForWidth() ) );
    layout59_3->addWidget( textLabel5_3 );

    layout1_3_2_3 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_3"); 

    m_edRT = new QLineEdit( centralWidget(), "m_edRT" );
    m_edRT->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edRT->sizePolicy().hasHeightForWidth() ) );
    m_edRT->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_3->addWidget( m_edRT );

    textLabel2_3_2_3 = new QLabel( centralWidget(), "textLabel2_3_2_3" );
    textLabel2_3_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_3->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_3->addWidget( textLabel2_3_2_3 );
    layout59_3->addLayout( layout1_3_2_3 );
    layout52->addLayout( layout59_3 );

    layout59 = new QHBoxLayout( 0, 0, 6, "layout59"); 

    textLabel5 = new QLabel( centralWidget(), "textLabel5" );
    textLabel5->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5->sizePolicy().hasHeightForWidth() ) );
    layout59->addWidget( textLabel5 );

    layout1_3_2 = new QHBoxLayout( 0, 0, 6, "layout1_3_2"); 

    m_edTau = new QLineEdit( centralWidget(), "m_edTau" );
    m_edTau->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edTau->sizePolicy().hasHeightForWidth() ) );
    m_edTau->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2->addWidget( m_edTau );

    textLabel2_3_2 = new QLabel( centralWidget(), "textLabel2_3_2" );
    textLabel2_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2->addWidget( textLabel2_3_2 );
    layout59->addLayout( layout1_3_2 );
    layout52->addLayout( layout59 );

    groupBox3 = new QGroupBox( centralWidget(), "groupBox3" );
    groupBox3->setColumnLayout(0, Qt::Vertical );
    groupBox3->layout()->setSpacing( 2 );
    groupBox3->layout()->setMargin( 4 );
    groupBox3Layout = new QGridLayout( groupBox3->layout() );
    groupBox3Layout->setAlignment( Qt::AlignTop );

    layout59_2 = new QHBoxLayout( 0, 0, 6, "layout59_2"); 

    textLabel5_2 = new QLabel( groupBox3, "textLabel5_2" );
    textLabel5_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2->sizePolicy().hasHeightForWidth() ) );
    layout59_2->addWidget( textLabel5_2 );

    layout1_3_2_2 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2"); 

    m_edPW1 = new QLineEdit( groupBox3, "m_edPW1" );
    m_edPW1->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edPW1->sizePolicy().hasHeightForWidth() ) );
    m_edPW1->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2->addWidget( m_edPW1 );

    textLabel2_3_2_2 = new QLabel( groupBox3, "textLabel2_3_2_2" );
    textLabel2_3_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2->addWidget( textLabel2_3_2_2 );
    layout59_2->addLayout( layout1_3_2_2 );

    groupBox3Layout->addLayout( layout59_2, 0, 0 );

    layout85_3_3 = new QHBoxLayout( 0, 0, 6, "layout85_3_3"); 

    textLabel2_3_4 = new QLabel( groupBox3, "textLabel2_3_4" );
    layout85_3_3->addWidget( textLabel2_3_4 );

    m_cmbP1Func = new QComboBox( FALSE, groupBox3, "m_cmbP1Func" );
    layout85_3_3->addWidget( m_cmbP1Func );

    groupBox3Layout->addLayout( layout85_3_3, 2, 0 );

    layout47_2 = new QHBoxLayout( 0, 0, 6, "layout47_2"); 

    textLabel5_2_2_2_2_2 = new QLabel( groupBox3, "textLabel5_2_2_2_2_2" );
    textLabel5_2_2_2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_2_2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout47_2->addWidget( textLabel5_2_2_2_2_2 );

    m_dblP1Level = new KDoubleNumInput( groupBox3, "m_dblP1Level" );
    m_dblP1Level->setAcceptDrops( FALSE );
    m_dblP1Level->setValue( 0 );
    m_dblP1Level->setMinValue( 0 );
    m_dblP1Level->setMaxValue( 0 );
    m_dblP1Level->setPrecision( 1 );
    layout47_2->addWidget( m_dblP1Level );

    groupBox3Layout->addLayout( layout47_2, 1, 0 );
    layout52->addWidget( groupBox3 );

    groupBox3_3 = new QGroupBox( centralWidget(), "groupBox3_3" );
    groupBox3_3->setColumnLayout(0, Qt::Vertical );
    groupBox3_3->layout()->setSpacing( 2 );
    groupBox3_3->layout()->setMargin( 4 );
    groupBox3_3Layout = new QGridLayout( groupBox3_3->layout() );
    groupBox3_3Layout->setAlignment( Qt::AlignTop );

    layout59_2_4 = new QHBoxLayout( 0, 0, 6, "layout59_2_4"); 

    textLabel5_2_4 = new QLabel( groupBox3_3, "textLabel5_2_4" );
    textLabel5_2_4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_4->sizePolicy().hasHeightForWidth() ) );
    layout59_2_4->addWidget( textLabel5_2_4 );

    layout1_3_2_2_4 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_4"); 

    m_edPW2 = new QLineEdit( groupBox3_3, "m_edPW2" );
    m_edPW2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edPW2->sizePolicy().hasHeightForWidth() ) );
    m_edPW2->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_4->addWidget( m_edPW2 );

    textLabel2_3_2_2_4 = new QLabel( groupBox3_3, "textLabel2_3_2_2_4" );
    textLabel2_3_2_2_4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2_4->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2_4->addWidget( textLabel2_3_2_2_4 );
    layout59_2_4->addLayout( layout1_3_2_2_4 );

    groupBox3_3Layout->addLayout( layout59_2_4, 0, 0 );

    layout47_2_2 = new QHBoxLayout( 0, 0, 6, "layout47_2_2"); 

    textLabel5_2_2_2_2_2_2 = new QLabel( groupBox3_3, "textLabel5_2_2_2_2_2_2" );
    textLabel5_2_2_2_2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_2_2_2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout47_2_2->addWidget( textLabel5_2_2_2_2_2_2 );

    m_dblP2Level = new KDoubleNumInput( groupBox3_3, "m_dblP2Level" );
    m_dblP2Level->setAcceptDrops( FALSE );
    m_dblP2Level->setValue( 0 );
    m_dblP2Level->setMinValue( 0 );
    m_dblP2Level->setMaxValue( 0 );
    m_dblP2Level->setPrecision( 1 );
    layout47_2_2->addWidget( m_dblP2Level );

    groupBox3_3Layout->addLayout( layout47_2_2, 1, 0 );

    layout85_3_3_2 = new QHBoxLayout( 0, 0, 6, "layout85_3_3_2"); 

    textLabel2_3_4_2 = new QLabel( groupBox3_3, "textLabel2_3_4_2" );
    layout85_3_3_2->addWidget( textLabel2_3_4_2 );

    m_cmbP2Func = new QComboBox( FALSE, groupBox3_3, "m_cmbP2Func" );
    layout85_3_3_2->addWidget( m_cmbP2Func );

    groupBox3_3Layout->addLayout( layout85_3_3_2, 2, 0 );
    layout52->addWidget( groupBox3_3 );

    FrmPulserLayout->addLayout( layout52, 0, 0 );

    groupBox3_2 = new QGroupBox( centralWidget(), "groupBox3_2" );
    groupBox3_2->setColumnLayout(0, Qt::Vertical );
    groupBox3_2->layout()->setSpacing( 2 );
    groupBox3_2->layout()->setMargin( 4 );
    groupBox3_2Layout = new QGridLayout( groupBox3_2->layout() );
    groupBox3_2Layout->setAlignment( Qt::AlignTop );

    m_tblPulse = new QTable( groupBox3_2, "m_tblPulse" );
    m_tblPulse->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)7, 0, 0, m_tblPulse->sizePolicy().hasHeightForWidth() ) );
    m_tblPulse->setNumRows( 0 );
    m_tblPulse->setNumCols( 2 );
    m_tblPulse->setReadOnly( TRUE );
    m_tblPulse->setSelectionMode( QTable::MultiRow );

    groupBox3_2Layout->addWidget( m_tblPulse, 0, 0 );

    FrmPulserLayout->addMultiCellWidget( groupBox3_2, 2, 2, 0, 1 );

    m_btnMoreConfig = new QPushButton( centralWidget(), "m_btnMoreConfig" );

    FrmPulserLayout->addMultiCellWidget( m_btnMoreConfig, 1, 1, 0, 1 );

    layout50 = new QVBoxLayout( 0, 4, 2, "layout50"); 

    textLabel5_2_3_3_2_2 = new QLabel( centralWidget(), "textLabel5_2_3_3_2_2" );
    textLabel5_2_3_3_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_3_2_2->sizePolicy().hasHeightForWidth() ) );
    layout50->addWidget( textLabel5_2_3_3_2_2 );

    m_dblMasterLevel = new KDoubleNumInput( centralWidget(), "m_dblMasterLevel" );
    m_dblMasterLevel->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)1, (QSizePolicy::SizeType)0, 0, 0, m_dblMasterLevel->sizePolicy().hasHeightForWidth() ) );
    m_dblMasterLevel->setAcceptDrops( FALSE );
    m_dblMasterLevel->setValue( 0 );
    m_dblMasterLevel->setMinValue( 0 );
    m_dblMasterLevel->setMaxValue( 0 );
    m_dblMasterLevel->setPrecision( 1 );
    layout50->addWidget( m_dblMasterLevel );

    layout85_3 = new QHBoxLayout( 0, 0, 6, "layout85_3"); 

    textLabel2_3 = new QLabel( centralWidget(), "textLabel2_3" );
    layout85_3->addWidget( textLabel2_3 );

    m_cmbCombMode = new QComboBox( FALSE, centralWidget(), "m_cmbCombMode" );
    layout85_3->addWidget( m_cmbCombMode );
    layout50->addLayout( layout85_3 );

    groupBox4 = new QGroupBox( centralWidget(), "groupBox4" );
    groupBox4->setColumnLayout(0, Qt::Vertical );
    groupBox4->layout()->setSpacing( 2 );
    groupBox4->layout()->setMargin( 4 );
    groupBox4Layout = new QGridLayout( groupBox4->layout() );
    groupBox4Layout->setAlignment( Qt::AlignTop );

    layout59_2_3_2_3 = new QHBoxLayout( 0, 0, 6, "layout59_2_3_2_3"); 

    textLabel5_2_3_2_3 = new QLabel( groupBox4, "textLabel5_2_3_2_3" );
    textLabel5_2_3_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_2_3->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3_2_3->addWidget( textLabel5_2_3_2_3 );

    layout1_3_2_2_3_2_3 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3_2_3"); 

    m_edCombP1Alt = new QLineEdit( groupBox4, "m_edCombP1Alt" );
    m_edCombP1Alt->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edCombP1Alt->sizePolicy().hasHeightForWidth() ) );
    m_edCombP1Alt->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3_2_3->addWidget( m_edCombP1Alt );

    textLabel2_3_2_2_3_2_3 = new QLabel( groupBox4, "textLabel2_3_2_2_3_2_3" );
    textLabel2_3_2_2_3_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2_3_2_3->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2_3_2_3->addWidget( textLabel2_3_2_2_3_2_3 );
    layout59_2_3_2_3->addLayout( layout1_3_2_2_3_2_3 );

    groupBox4Layout->addLayout( layout59_2_3_2_3, 7, 0 );

    layout59_2_3_2_2 = new QHBoxLayout( 0, 0, 6, "layout59_2_3_2_2"); 

    textLabel5_2_3_2_2 = new QLabel( groupBox4, "textLabel5_2_3_2_2" );
    textLabel5_2_3_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_2_2->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3_2_2->addWidget( textLabel5_2_3_2_2 );

    layout1_3_2_2_3_2_2 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3_2_2"); 

    m_edCombP1 = new QLineEdit( groupBox4, "m_edCombP1" );
    m_edCombP1->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edCombP1->sizePolicy().hasHeightForWidth() ) );
    m_edCombP1->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3_2_2->addWidget( m_edCombP1 );

    textLabel2_3_2_2_3_2_2 = new QLabel( groupBox4, "textLabel2_3_2_2_3_2_2" );
    textLabel2_3_2_2_3_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2_3_2_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2_3_2_2->addWidget( textLabel2_3_2_2_3_2_2 );
    layout59_2_3_2_2->addLayout( layout1_3_2_2_3_2_2 );

    groupBox4Layout->addLayout( layout59_2_3_2_2, 6, 0 );

    layout59_2_3_2 = new QHBoxLayout( 0, 0, 6, "layout59_2_3_2"); 

    textLabel5_2_3_2 = new QLabel( groupBox4, "textLabel5_2_3_2" );
    textLabel5_2_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_2->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3_2->addWidget( textLabel5_2_3_2 );

    layout1_3_2_2_3_2 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3_2"); 

    m_edCombPT = new QLineEdit( groupBox4, "m_edCombPT" );
    m_edCombPT->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edCombPT->sizePolicy().hasHeightForWidth() ) );
    m_edCombPT->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3_2->addWidget( m_edCombPT );

    textLabel2_3_2_2_3_2 = new QLabel( groupBox4, "textLabel2_3_2_2_3_2" );
    textLabel2_3_2_2_3_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2_3_2->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2_3_2->addWidget( textLabel2_3_2_2_3_2 );
    layout59_2_3_2->addLayout( layout1_3_2_2_3_2 );

    groupBox4Layout->addLayout( layout59_2_3_2, 5, 0 );

    layout71 = new QHBoxLayout( 0, 0, 6, "layout71"); 

    textLabel5_2_3_3 = new QLabel( groupBox4, "textLabel5_2_3_3" );
    textLabel5_2_3_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_3->sizePolicy().hasHeightForWidth() ) );
    layout71->addWidget( textLabel5_2_3_3 );

    m_numCombNum = new QSpinBox( groupBox4, "m_numCombNum" );
    m_numCombNum->setMinValue( 0 );
    m_numCombNum->setValue( 0 );
    layout71->addWidget( m_numCombNum );

    groupBox4Layout->addLayout( layout71, 4, 0 );

    layout59_2_3_2_3_2 = new QHBoxLayout( 0, 0, 6, "layout59_2_3_2_3_2"); 

    textLabel5_2_3_2_3_4 = new QLabel( groupBox4, "textLabel5_2_3_2_3_4" );
    textLabel5_2_3_2_3_4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3_2_3_4->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3_2_3_2->addWidget( textLabel5_2_3_2_3_4 );

    layout1_3_2_2_3_2_3_4 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3_2_3_4"); 

    m_edCombOffRes = new QLineEdit( groupBox4, "m_edCombOffRes" );
    m_edCombOffRes->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edCombOffRes->sizePolicy().hasHeightForWidth() ) );
    m_edCombOffRes->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3_2_3_4->addWidget( m_edCombOffRes );

    textLabel2_3_2_2_3_2_3_4 = new QLabel( groupBox4, "textLabel2_3_2_2_3_2_3_4" );
    textLabel2_3_2_2_3_2_3_4->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2_3_2_3_4->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2_3_2_3_4->addWidget( textLabel2_3_2_2_3_2_3_4 );
    layout59_2_3_2_3_2->addLayout( layout1_3_2_2_3_2_3_4 );

    groupBox4Layout->addLayout( layout59_2_3_2_3_2, 3, 0 );

    layout59_2_3 = new QHBoxLayout( 0, 0, 6, "layout59_2_3"); 

    textLabel5_2_3 = new QLabel( groupBox4, "textLabel5_2_3" );
    textLabel5_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_3->sizePolicy().hasHeightForWidth() ) );
    layout59_2_3->addWidget( textLabel5_2_3 );

    layout1_3_2_2_3 = new QHBoxLayout( 0, 0, 6, "layout1_3_2_2_3"); 

    m_edCombPW = new QLineEdit( groupBox4, "m_edCombPW" );
    m_edCombPW->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edCombPW->sizePolicy().hasHeightForWidth() ) );
    m_edCombPW->setMaximumSize( QSize( 80, 32767 ) );
    layout1_3_2_2_3->addWidget( m_edCombPW );

    textLabel2_3_2_2_3 = new QLabel( groupBox4, "textLabel2_3_2_2_3" );
    textLabel2_3_2_2_3->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel2_3_2_2_3->sizePolicy().hasHeightForWidth() ) );
    layout1_3_2_2_3->addWidget( textLabel2_3_2_2_3 );
    layout59_2_3->addLayout( layout1_3_2_2_3 );

    groupBox4Layout->addLayout( layout59_2_3, 0, 0 );

    layout85_3_3_2_2 = new QHBoxLayout( 0, 0, 6, "layout85_3_3_2_2"); 

    textLabel2_3_4_2_2 = new QLabel( groupBox4, "textLabel2_3_4_2_2" );
    layout85_3_3_2_2->addWidget( textLabel2_3_4_2_2 );

    m_cmbCombFunc = new QComboBox( FALSE, groupBox4, "m_cmbCombFunc" );
    layout85_3_3_2_2->addWidget( m_cmbCombFunc );

    groupBox4Layout->addLayout( layout85_3_3_2_2, 2, 0 );

    layout47 = new QHBoxLayout( 0, 0, 6, "layout47"); 

    textLabel5_2_2_2_2 = new QLabel( groupBox4, "textLabel5_2_2_2_2" );
    textLabel5_2_2_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, textLabel5_2_2_2_2->sizePolicy().hasHeightForWidth() ) );
    layout47->addWidget( textLabel5_2_2_2_2 );

    m_dblCombLevel = new KDoubleNumInput( groupBox4, "m_dblCombLevel" );
    m_dblCombLevel->setAcceptDrops( FALSE );
    m_dblCombLevel->setValue( 0 );
    m_dblCombLevel->setMinValue( 0 );
    m_dblCombLevel->setMaxValue( 0 );
    m_dblCombLevel->setPrecision( 1 );
    layout47->addWidget( m_dblCombLevel );

    groupBox4Layout->addLayout( layout47, 1, 0 );
    layout50->addWidget( groupBox4 );

    FrmPulserLayout->addLayout( layout50, 0, 1 );

    // toolbars

    languageChange();
    resize( QSize(795, 495).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // tab order
    setTabOrder( m_ckbOutput, m_cmbRTMode );
    setTabOrder( m_cmbRTMode, m_edRT );
    setTabOrder( m_edRT, m_edTau );
    setTabOrder( m_edTau, m_edPW1 );
    setTabOrder( m_edPW1, m_dblP1Level );
    setTabOrder( m_dblP1Level, m_cmbP1Func );
    setTabOrder( m_cmbP1Func, m_edPW2 );
    setTabOrder( m_edPW2, m_dblP2Level );
    setTabOrder( m_dblP2Level, m_cmbP2Func );
    setTabOrder( m_cmbP2Func, m_dblMasterLevel );
    setTabOrder( m_dblMasterLevel, m_cmbCombMode );
    setTabOrder( m_cmbCombMode, m_edCombPW );
    setTabOrder( m_edCombPW, m_dblCombLevel );
    setTabOrder( m_dblCombLevel, m_cmbCombFunc );
    setTabOrder( m_cmbCombFunc, m_edCombOffRes );
    setTabOrder( m_edCombOffRes, m_numCombNum );
    setTabOrder( m_numCombNum, m_edCombPT );
    setTabOrder( m_edCombPT, m_edCombP1 );
    setTabOrder( m_edCombP1, m_edCombP1Alt );
    setTabOrder( m_edCombP1Alt, m_btnMoreConfig );
    setTabOrder( m_btnMoreConfig, m_tblPulse );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmPulser::~FrmPulser()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmPulser::languageChange()
{
    setCaption( tr2i18n( "Pulser Control" ) );
    m_ckbOutput->setText( tr2i18n( "Output" ) );
    textLabel2_2->setText( tr2i18n( "RT Mode" ) );
    textLabel5_3->setText( tr2i18n( "Rep/Rest Time" ) );
    textLabel2_3_2_3->setText( tr2i18n( "ms" ) );
    textLabel5->setText( tr2i18n( "Tau" ) );
    textLabel2_3_2->setText( tr2i18n( "us" ) );
    groupBox3->setTitle( tr2i18n( "Pi/2 Pulse" ) );
    textLabel5_2->setText( tr2i18n( "Pulse Width" ) );
    textLabel2_3_2_2->setText( tr2i18n( "us" ) );
    textLabel2_3_4->setText( tr2i18n( "Waveform" ) );
    textLabel5_2_2_2_2_2->setText( tr2i18n( "Output Level" ) );
    m_dblP1Level->setSuffix( tr2i18n( " dB" ) );
    groupBox3_3->setTitle( tr2i18n( "Pi Pulse" ) );
    textLabel5_2_4->setText( tr2i18n( "Pulse Width" ) );
    textLabel2_3_2_2_4->setText( tr2i18n( "us" ) );
    textLabel5_2_2_2_2_2_2->setText( tr2i18n( "Output Level" ) );
    m_dblP2Level->setSuffix( tr2i18n( " dB" ) );
    textLabel2_3_4_2->setText( tr2i18n( "Waveform" ) );
    groupBox3_2->setTitle( tr2i18n( "Pulse Table (Select Region to Visualize)" ) );
    m_btnMoreConfig->setText( tr2i18n( "More Config." ) );
    textLabel5_2_3_3_2_2->setText( tr2i18n( "Master Level" ) );
    m_dblMasterLevel->setSuffix( tr2i18n( " dB" ) );
    textLabel2_3->setText( tr2i18n( "Comb Mode" ) );
    groupBox4->setTitle( tr2i18n( "Comb Pulse" ) );
    textLabel5_2_3_2_3->setText( tr2i18n( "<i>P</i><font size=\"-1\">1</font> (ALT)" ) );
    textLabel2_3_2_2_3_2_3->setText( tr2i18n( "ms" ) );
    textLabel5_2_3_2_2->setText( tr2i18n( "<i>P</i><font size=\"-1\">1</font>" ) );
    textLabel2_3_2_2_3_2_2->setText( tr2i18n( "ms" ) );
    textLabel5_2_3_2->setText( tr2i18n( "Periodic Term" ) );
    textLabel2_3_2_2_3_2->setText( tr2i18n( "us" ) );
    textLabel5_2_3_3->setText( tr2i18n( "# of Pulses" ) );
    textLabel5_2_3_2_3_4->setText( tr2i18n( "Off-Resonance" ) );
    textLabel2_3_2_2_3_2_3_4->setText( tr2i18n( "kHz" ) );
    textLabel5_2_3->setText( tr2i18n( "Pulse Width" ) );
    textLabel2_3_2_2_3->setText( tr2i18n( "us" ) );
    textLabel2_3_4_2_2->setText( tr2i18n( "Waveform" ) );
    textLabel5_2_2_2_2->setText( tr2i18n( "Output Level" ) );
    m_dblCombLevel->setSuffix( tr2i18n( " dB" ) );
}

#include "pulserdriverform.moc"
