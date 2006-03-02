/****************************************************************************
** Form interface generated from reading ui file '../../../../kame/script/forms/rubythreadtool.ui'
**
** Created: 木  3 2 16:23:05 2006
**      by: The User Interface Compiler ($Id: rubythreadtool.h,v 1.1.2.1 2006/03/02 09:19:54 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMRUBYTHREAD_H
#define FRMRUBYTHREAD_H

#include <qvariant.h>
#include <qwidget.h>

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QPushButton;
class QTextBrowser;
class QLabel;

class FrmRubyThread : public QWidget
{
    Q_OBJECT

public:
    FrmRubyThread( QWidget* parent = 0, const char* name = 0, WFlags fl = 0 );
    ~FrmRubyThread();

    QPushButton* m_pbtnKill;
    QPushButton* m_pbtnResume;
    QTextBrowser* m_ptxtDefout;
    QLabel* m_plblFilename;
    QLabel* m_plblStatus;
    QLabel* m_plblFilename_2_2;

protected:
    QGridLayout* FrmRubyThreadLayout;
    QSpacerItem* spacer1;

protected slots:
    virtual void languageChange();

};

#endif // FRMRUBYTHREAD_H
