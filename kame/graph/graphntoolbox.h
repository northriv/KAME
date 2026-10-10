/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------

#ifndef graphntoolboxH
#define graphntoolboxH
//---------------------------------------------------------------------------

#include "xnodeconnector.h"
#include "graph.h"
#include <fstream>
#include <functional>

class XQGraph;
class QLineEdit;
class QAbstractButton;
class QPushButton;
class Ui_FrmGraphNURL;
typedef QForm<QWidget, Ui_FrmGraphNURL> FrmGraphNURL;

class DECLSPEC_KAME XGraphNToolBox: public XNode {
public:
    XGraphNToolBox(const char *name, bool runtime, FrmGraphNURL *item);
    XGraphNToolBox(const char *name, bool runtime, XQGraph *graphwidget,
        QLineEdit *ed = nullptr, QAbstractButton *btn = nullptr, QPushButton *btndump = nullptr,
        const char *selfilter = "Data files (*.dat);;All files (*.*)");
    virtual ~XGraphNToolBox();

    const shared_ptr<XGraph> &graph() const { return m_graph;}
    const shared_ptr<XStringNode> &filename() const { return m_filename;}

    const shared_ptr<XTouchableNode> &dump() const { return m_dump;}

    //! The next file of the series \a templ names: "<stem>_<NNNN>_<YYYYMMDD-HHMMSS>.<ext>"
    //! in its folder, \a time in the name.  \a seq is the last number used
    //! (0: look up the highest already in the folder first) and becomes the
    //! one returned.  Shared by one-file-per-shot dumps and anything else
    //! saving images into a numbered series (XImageTrigger).
    static QString nextNumberedPath(const QString &templ, unsigned int &seq, const XTime &time);
    //! \a write fills a temporary file beside \a path, which is then renamed
    //! to it, so a synced folder (iCloud) never picks up half a file.
    //! \return an error message, empty on success.
    static XString writeAtomically(const QString &path, const std::function<bool(const QString &tmp)> &write);

    struct DECLSPEC_KAME Payload : public XNode::Payload {
        const Talker<bool> &onIconChanged() const { return m_tlkOnIconChanged;}
        Talker<bool> &onIconChanged() { return m_tlkOnIconChanged;}
    private:
        friend class XGraphNToolBox;
        Talker<bool> m_tlkOnIconChanged;
    };
protected:
    virtual void dumpToFileThreaded(std::fstream &, const Snapshot &, const std::string &ext) = 0;
    //! True when a dump in format \a ext must be a file of its own: an image
    //! is a whole file, and one appended to another is unreadable past the
    //! first.  FileName is then a template, each Dump writing
    //! "<stem>_<NNNN>_<YYYYMMDD-HHMMSS>.<ext>" beside it -- numbered on from
    //! the highest already there, so a restart never overwrites -- through a
    //! temporary file renamed into place, so a synced folder (iCloud) never
    //! picks up half a file.  Formats that accumulate dumps in one file, like
    //! the .dat text, answer false and keep appending.
    virtual bool dumpsOneFilePerShot(const std::string &/*ext*/) const {return false;}

    std::deque<xqcon_ptr> m_conUIs;
private:
    QPushButton * const m_btnDump;

    const shared_ptr<XGraph> m_graph;

    const shared_ptr<XTouchableNode> m_dump;
    const shared_ptr<XStringNode> m_filename;

//    const shared_ptr<XBoolNode> m_axisSelectionTool, m_planeSelectionTool;

    shared_ptr<Listener> m_lsnOnDumpTouched, m_lsnOnFilenameChanged,
        m_lsnOnIconChanged;

    void onDumpTouched(const Snapshot &shot, XTouchableNode *);
    void onFilenameChanged(const Snapshot &shot, XValueNodeBase *);
    void onIconChanged(const Snapshot &shot, bool );
    //! One dump into a file of its own; under m_filemutex.
    void dumpOneShot(const Snapshot &shot);

    xqcon_ptr m_conFilename, m_conDump;

    unique_ptr<XThread> m_threadDump;
    std::fstream m_stream;
    std::string m_ext;
    bool m_oneFilePerShot = false; //!< under m_filemutex, as are the two above.
    unsigned int m_shotSeq = 0; //!< last number written for FileName; 0: not yet looked up.
    XMutex m_filemutex;
};
#endif
