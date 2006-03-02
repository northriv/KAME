/****************************************************************************
** Form interface generated from reading ui file '../../../../kame/graph/forms/graphnurlform.ui'
**
** Created: 木  3 2 16:25:22 2006
**      by: The User Interface Compiler ($Id: graphnurlform.h,v 1.1.2.1 2006/03/02 09:19:26 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMGRAPHNURL_H
#define FRMGRAPHNURL_H

#include <qvariant.h>
#include <qpixmap.h>
#include <qmainwindow.h>

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QAction;
class QActionGroup;
class QToolBar;
class QPopupMenu;
class XQGraph;
class KURLRequester;
class QPushButton;

class FrmGraphNURL : public QMainWindow
{
    Q_OBJECT

public:
    FrmGraphNURL( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~FrmGraphNURL();

    KURLRequester* m_url;
    QPushButton* m_btnDump;
    XQGraph* m_graphwidget;

protected:
    QGridLayout* FrmGraphNURLLayout;
    QHBoxLayout* layout1;

protected slots:
    virtual void languageChange();

private:
    QPixmap image0;
    QPixmap image1;

};

#endif // FRMGRAPHNURL_H
