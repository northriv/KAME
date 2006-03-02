/****************************************************************************
** Form interface generated from reading ui file '../../../../kame/analyzer/forms/recordreaderform.ui'
**
** Created: 木  3 2 16:47:57 2006
**      by: The User Interface Compiler ($Id: recordreaderform.h,v 1.1.2.1 2006/03/02 09:20:54 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMRECORDREADER_H
#define FRMRECORDREADER_H

#include <qvariant.h>
#include <qwidget.h>

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QLabel;
class KURLRequester;
class QLineEdit;
class QPushButton;
class QComboBox;

class FrmRecordReader : public QWidget
{
    Q_OBJECT

public:
    FrmRecordReader( QWidget* parent = 0, const char* name = 0, WFlags fl = 0 );
    ~FrmRecordReader();

    QLabel* textLabel1_;
    KURLRequester* urlBinRec;
    QLabel* textLabel3;
    QLineEdit* edTime;
    QPushButton* btnRW;
    QPushButton* btnStop;
    QPushButton* btnFF;
    QPushButton* btnFirst;
    QPushButton* btnBack;
    QPushButton* btnNext;
    QLabel* textLabel2;
    QComboBox* cmbSpeed;

protected:
    QGridLayout* FrmRecordReaderLayout;
    QSpacerItem* spacer6;
    QSpacerItem* spacer4;
    QGridLayout* layout12;
    QSpacerItem* spacer2;
    QSpacerItem* spacer3;
    QVBoxLayout* layout1;
    QVBoxLayout* layout6;
    QHBoxLayout* layout5;
    QVBoxLayout* layout11;
    QVBoxLayout* layout8;
    QHBoxLayout* layout2;
    QHBoxLayout* layout3;
    QHBoxLayout* layout10;

protected slots:
    virtual void languageChange();

};

#endif // FRMRECORDREADER_H
