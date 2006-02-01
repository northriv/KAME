/****************************************************************************
** XQGraphDialogConnector meta object code from reading C++ file 'graphdialogconnector.h'
**
** Created: Wed Feb 1 03:45:49 2006
**      by: The Qt MOC ($Id: graphdialogconnector.moc.cpp,v 1.1 2006/02/01 18:45:36 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#undef QT_NO_COMPAT
#include "../../../kame/graph/graphdialogconnector.h"
#include <qmetaobject.h>
#include <qapplication.h>

#include <private/qucomextra_p.h>
#if !defined(Q_MOC_OUTPUT_REVISION) || (Q_MOC_OUTPUT_REVISION != 26)
#error "This file was generated using the moc from 3.3.5. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

const char *XQGraphDialogConnector::className() const
{
    return "XQGraphDialogConnector";
}

QMetaObject *XQGraphDialogConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XQGraphDialogConnector( "XQGraphDialogConnector", &XQGraphDialogConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XQGraphDialogConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQGraphDialogConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XQGraphDialogConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQGraphDialogConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XQGraphDialogConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XQConnector::staticMetaObject();
    metaObj = QMetaObject::new_metaobject(
	"XQGraphDialogConnector", parentObject,
	0, 0,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XQGraphDialogConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XQGraphDialogConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XQGraphDialogConnector" ) )
	return this;
    return XQConnector::qt_cast( clname );
}

bool XQGraphDialogConnector::qt_invoke( int _id, QUObject* _o )
{
    return XQConnector::qt_invoke(_id,_o);
}

bool XQGraphDialogConnector::qt_emit( int _id, QUObject* _o )
{
    return XQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XQGraphDialogConnector::qt_property( int id, int f, QVariant* v)
{
    return XQConnector::qt_property( id, f, v);
}

bool XQGraphDialogConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES
