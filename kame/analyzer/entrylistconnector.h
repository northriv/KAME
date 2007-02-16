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
#ifndef entrylistconnectorH
#define entrylistconnectorH

#include "xnodeconnector.h"
//---------------------------------------------------------------------------

class QTable;

class XScalarEntry;
class XChartList;
class XScalarEntryList;
class XDriver;

class XEntryListConnector : public XListQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XEntryListConnector
  (const shared_ptr<XScalarEntryList> &node, QTable *item, const shared_ptr<XChartList> &chartlist);
 public:
  virtual ~XEntryListConnector() {}
 protected:
  virtual void onCatch(const shared_ptr<XNode> &node);
  virtual void onRelease(const shared_ptr<XNode> &node);
 protected slots:
    void clicked ( int row, int col, int button, const QPoint& );
 private:
  const shared_ptr<XChartList> m_chartList;

  struct tcons {
      struct tlisttext {
        QLabel *label;
        shared_ptr<std::string> str;
      };
    xqcon_ptr constore, condelta;
    QLabel *label;
    shared_ptr<XScalarEntry> entry;
    shared_ptr<XDriver> driver;
    shared_ptr<XTalker<tlisttext> > tlkOnRecordRedirected;
    shared_ptr<XListener> lsnOnRecordRedirected;
    void onRecordRedirected(const tlisttext &);
  };
  typedef std::deque<shared_ptr<tcons> > tconslist;
  tconslist m_cons;
  shared_ptr<XListener> m_lsnOnRecord;
  void onRecord(const shared_ptr<XDriver> &driver);
};

#endif
