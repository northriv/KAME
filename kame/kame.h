/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef KAME_H
#define KAME_H

#include "support.h"
#include <qvariant.h>
#include <qpixmap.h>
#include <kmdimainfrm.h>

class FrmRecordReader;
class FrmGraphList;
class FrmCalTable;
class FrmInterface;
class FrmDriver;
class FrmEntry;
class FrmSequence;
class QTimer;
class QAction;
class QPopupMenu;
class QMenuBar;
class QToolBar;
class XMeasure;

#include "xnodeconnector.h"

/*! Main form widget of KAME.
 * use \a g_pFrmMain to access this.
 * \sa g_pFrmMain
 */
class FrmKameMain : public KMdiMainFrm
{
	Q_OBJECT
public:
    FrmKameMain();
    ~FrmKameMain();
    
    QMenuBar *m_pMenubar;
    QPopupMenu *m_pFileMenu;
    QPopupMenu *m_pMeasureMenu;
    QPopupMenu *m_pScriptMenu;
    QPopupMenu *m_pHelpMenu;
    QToolBar *m_pToolbar;
    QAction* m_pFileOpenAction;
    QAction* m_pFileSaveAction;
    QAction* m_pFileExitAction;
    QAction* m_pHelpContentsAction;
    QAction* m_pHelpIndexAction;
    QAction* m_pHelpAboutAction;
    QAction* m_pFileLogAction;
//    QAction* m_pMesRunAction;
    QAction* m_pMesStopAction;
    QAction* m_pScriptRunAction;
    QAction* m_pScriptDotSaveAction;
    QAction* m_pFileCloseAction;
    
    FrmRecordReader *m_pFrmRecordReader;
    FrmGraphList *m_pFrmGraphList;
    FrmCalTable *m_pFrmCalTable;
    FrmInterface *m_pFrmInterface;
    FrmDriver *m_pFrmDriver;
    FrmEntry *m_pFrmScalarEntry;
    
    int openMes(const QString &filename);
public slots:
virtual void fileCloseAction_activated();
    virtual void fileExitAction_activated();
    virtual void fileOpenAction_activated();
    virtual void fileSaveAction_activated();
    virtual void helpAboutAction_activated();
    virtual void helpContentsAction_activated();
    virtual void helpIndexAction_activated();
//    virtual void mesRunAction_activated();
    virtual void mesStopAction_activated();
    virtual void scriptRunAction_activated();
    virtual void scriptDotSaveAction_activated();
    virtual void fileLogAction_toggled( bool var );
protected slots:
virtual void aboutToQuit();
	virtual void processSignals();
private:
    void closeEvent( QCloseEvent* ce );
    QTimer *m_pTimer;
    shared_ptr<XMeasure> m_measure;
    xqcon_ptr m_conMeasRubyThread;
    std::deque<xqcon_ptr> m_conRubyThreadList;
};

#endif /*KAME_H*/
