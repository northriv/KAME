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

#include <functional>
#include <map>

//! A journal file (`.kamj`), read as a stream.
//!
//! Reading, not replaying: this class knows the format and nothing about the
//! tree.  It is what both halves need -- the record reader, which replays a
//! run beside its `.kamb`, and anything offline that wants to ask what a
//! setting was at a given moment.
//!
//! **Nothing here is bounded by the length of the file.**  KAME is left
//! running for a month at a time, and a Logbook produces about 3 kB/s on
//! disk, so a journal is a gigabyte-scale object and has to be treated as
//! one.  What is held is the head -- the header and the dump -- which is
//! bounded by the size of the *tree*, plus one line at a time of the body.
//!
//! Opening therefore reads only as far as the first timestamped line: the
//! dump ends there by construction, since a dump line is the one kind that
//! carries no time.  The body is then walked forward by advanceTo(), which
//! hands each line to a caller that knows what to do with it.
//!
//! \sa doc/design/PROVENANCE.md, XJournalWriter (the writer).
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
    //! One line, valid only for the duration of the call it is handed to.
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
    typedef std::function<void(const Event &)> Apply;

    XJournalFile() {}
    ~XJournalFile();
    XJournalFile(XJournalFile &&);
    XJournalFile &operator=(XJournalFile &&);

    //! Reads the header and the dump, and leaves the cursor at the first
    //! entry.  \a apply sees the dump, which is the baseline everything after
    //! it is a change to.  \return false with \a errmsg set, and nothing opened.
    bool open(const XString &path, const Apply &apply, XString &errmsg);
    void close();
    bool isOpen() const {return m_gz;}

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

    //! Walks forward, handing every line up to and including \a until to
    //! \a apply, and stops holding the first line after it.  \return false at
    //! end of file, which is not an error.
    bool advanceTo(const XTime &until, const Apply &apply);
    //! Back to the first entry, re-reading the head.  \a apply sees the dump
    //! again, so a caller that rebuilt its state from it can do so again.
    //!
    //! This is how backwards is done, and on a long journal it is expensive:
    //! reaching a moment costs everything before it.  Bounded seeking wants
    //! checkpoints -- the cursor's offset paired with a copy of the caller's
    //! state, every so often -- which is worth building when someone actually
    //! steps backwards through a month.
    bool rewind(const Apply &apply);

    //! The time of the next line, without consuming it.  \return false at
    //! end of file.  What a caller stepping one entry at a time needs: the
    //! step is "everything stamped with that instant", and the instant has to
    //! be known before advanceTo() can be asked for it.
    bool peekTime(XTime *when);
    //! Where the cursor is in the COMPRESSED file, which is what a position
    //! readout can be made of: the unpacked length is not knowable without
    //! unpacking it.  -1 when closed.
    int64_t offset() const;

    //! Where the cursor is: the time of the last line handed over.
    const XTime &at() const {return m_at;}
    //! Identities seen so far.  Grows as the body introduces nodes, and is
    //! bounded by how many have ever existed, not by the length of the file.
    const std::map<uint32_t, NodeInfo> &nodes() const {return m_nodes;}
    //! False when the body carries `ts` but not `tu` -- a journal from
    //! before the epoch stamp existed.  Its lines are still in order, but
    //! none of them can be placed against a raw record's clock, so a replay
    //! driven by time would silently restore the wrong state.  Say so instead.
    bool timesKnown() const {return m_timesKnown;}
    //! Lines that could not be understood -- a journal from a later KAME, or
    //! the half-written last line a killed process leaves.  Counted rather
    //! than refused: a file that says more than we know still says it.
    unsigned int unknownLines() const {return m_unknown;}

    //! `run042.kamb` -> `run042.kamj`, the sibling a raw stream was written
    //! beside.  Empty when that file does not exist.
    static XString journalBeside(const XString &rawpath);

private:
    XJournalFile(const XJournalFile &) = delete;
    XJournalFile &operator=(const XJournalFile &) = delete;
    //! \return false at end of file.
    bool readHead(const Apply &apply, XString &errmsg);
    bool nextLine(Event &e);

    void *m_gz = nullptr;
    XString m_path, m_kind, m_session, m_mode, m_rawFile;
    std::map<uint32_t, NodeInfo> m_nodes;
    XTime m_at;
    unsigned int m_unknown = 0;
    bool m_timesKnown = true;
    bool m_held = false;    //!< a line was read past the cursor and not yet used
    Event m_holding;
};

#endif /*xjournalreplayH*/
