/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef xjournalreplayH
#define xjournalreplayH

#include "support.h"
#include "xtime.h"

#include <map>
#include <vector>

//! One journal file (`.kamj`), read into memory.
//!
//! Reading, not replaying: this class knows the format and nothing about the
//! tree.  It is what both halves need -- the record reader, which replays a
//! run beside its `.kamb`, and anything offline that wants to ask what a
//! setting was at a given moment.
//!
//! Whole-file, because journals are small: a session of a real ODMR run
//! compressed to 62 kB, and a run journal is a fraction of that.  A streaming
//! reader would buy nothing and cost the ability to step backwards.
//!
//! \sa doc/design/PROVENANCE.md, XJournal (the writer).
class XJournalFile {
public:
    //! A node as the journal saw it: enough to find it again in a live tree,
    //! or to say what could not be found.
    struct NodeInfo {
        uint32_t id = 0;
        uint32_t parent = 0;
        int index = -1;         //!< position among siblings; -1 for the root
        XString name;
        XString path;           //!< "/Drivers/ODMR2D/Average"
        XString type;           //!< registry key, when the node has one
        XString cls;            //!< class name, never an instruction
        bool runtime = false;
        bool isList = false;
        bool isAliasList = false;
    };
    //! One line of the body, in the order it was written.
    //!
    //! Order is the timeline, not the timestamps: a value that arrives with
    //! a node (a driver created mid-run brings its whole subtree) has no
    //! stamp of its own, and belongs exactly where it appears.
    struct Event {
        enum class Kind {
            NODE,       //!< a node appeared, and here is its identity
            VALUE,      //!< a value: either an entry, or a node's initial state
            RELEASED,   //!< a node left the tree
            MARKER,     //!< run/session start and end
        };
        Kind kind = Kind::MARKER;
        XTime when;             //!< inherited from the stream position when the line carried none
        bool stamped = false;   //!< false when inherited
        uint32_t id = 0;
        int64_t serial = 0;
        bool fromDump = false;  //!< the state a node was in when first seen, not a change
        bool request = false;   //!< attributed to a human; a report is a driver talking about itself
        XString value;
        double exact = 0.0;
        bool hasExact = false;
        XString marker;         //!< "run start", "session end", ... for Kind::MARKER
        XString file;           //!< the run file a marker refers to
    };

    XJournalFile() {}

    //! Reads the whole file.  \return false with \a errmsg set, and nothing changed.
    bool open(const XString &path, XString &errmsg);
    bool isOpen() const {return m_isOpen;}

    const XString &path() const {return m_path;}
    //! "run", "session", or "save".
    const XString &kind() const {return m_kind;}
    const XString &session() const {return m_session;}
    //! The tier the run was recorded at; empty for a session or a save.
    const XString &mode() const {return m_mode;}
    //! The raw stream written beside this journal, as the base name the
    //! header records.  Empty when the run kept no raw stream -- and a
    //! session journal never has one.
    const XString &rawFile() const {return m_rawFile;}
    //! That name resolved against this journal's own directory: where the
    //! pair actually is, which is what a reader needs.  Empty when there is none.
    XString rawPath() const;

    const std::vector<Event> &events() const {return m_events;}
    const std::map<uint32_t, NodeInfo> &nodes() const {return m_nodes;}
    //! Lines the reader did not understand -- a journal written by a later
    //! KAME, most likely.  Counted rather than refused: a file that says more
    //! than we know is still worth everything it does say.
    unsigned int unknownLines() const {return m_unknown;}

    //! `run042.kamb` -> `run042.kamj`, the sibling a raw stream was written
    //! beside.  Empty when that file does not exist.
    static XString journalBeside(const XString &rawpath);

private:
    bool m_isOpen = false;
    XString m_path, m_kind, m_session, m_mode, m_rawFile;
    std::vector<Event> m_events;
    std::map<uint32_t, NodeInfo> m_nodes;
    unsigned int m_unknown = 0;
};

#endif /*xjournalreplayH*/
