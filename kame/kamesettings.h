/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef KAMESETTINGS_H
#define KAMESETTINGS_H

#include <QSettings>
#include <QString>

//! Where the things that outlive a measurement are kept: window geometries,
//! the appearance, the graph theme, the directories dialogs were last used in.
//!
//! An INI file rather than the platform's native store, and deliberately: on
//! macOS a plist is owned by cfprefsd, which caches it and writes it back, so
//! deleting the file -- what somebody does when a layout has gone wrong --
//! does not reliably take.  This one is a text file at ~/.config/kame/kame.ini
//! on every platform Qt puts it there, macOS included (measured: 6.10.1 does
//! NOT use ~/Library/Preferences for an IniFormat store), and it can be read,
//! edited or thrown away.
//!
//! What belongs to a MEASUREMENT does not belong here: driver settings, graph
//! settings and the tree itself are the .kam file's, and the line between the
//! two is worth keeping sharp.
struct KameSettings : public QSettings {
    KameSettings() : QSettings(QSettings::IniFormat, QSettings::UserScope,
        "kame", "kame") {}
};

//! The directory a dialog of this kind last landed in, empty if there is none.
//!
//! \a key names the KIND, not the dialog: measurements, scripts and dump files
//! live in different places on a working machine, and one shared "last
//! directory" would send each of them to wherever the previous one went.
QString kameLastDir(const char *key);
//! \a path may be a file; its directory is what gets stored.
void kameStoreLastDir(const char *key, const QString &path);

#endif
