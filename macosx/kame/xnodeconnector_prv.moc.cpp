/****************************************************************************
** _XQConnectorHolder meta object code from reading C++ file 'xnodeconnector_prv.h'
**
** Created: Wed Feb 1 04:04:44 2006
**      by: The Qt MOC ($Id: xnodeconnector_prv.moc.cpp,v 1.1 2006/02/01 18:44:36 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#undef QT_NO_COMPAT
#include "../../kame/xnodeconnector_prv.h"
#include <qmetaobject.h>
#include <qapplication.h>

#include <private/qucomextra_p.h>
#if !defined(Q_MOC_OUTPUT_REVISION) || (Q_MOC_OUTPUT_REVISION != 26)
#error "This file was generated using the moc from 3.3.5. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

const char *_XQConnectorHolder::className() const
{
    return "_XQConnectorHolder";
}

QMetaObject *_XQConnectorHolder::metaObj = 0;
static QMetaObjectCleanUp cleanUp__XQConnectorHolder( "_XQConnectorHolder", &_XQConnectorHolder::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString _XQConnectorHolder::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "_XQConnectorHolder", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString _XQConnectorHolder::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "_XQConnectorHolder", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* _XQConnectorHolder::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = QObject::staticMetaObject();
    static const QUMethod slot_0 = {"destroyed", 0, 0 };
    static const QMetaData slot_tbl[] = {
	{ "destroyed()", &slot_0, QMetaData::Protected }
    };
    metaObj = QMetaObject::new_metaobject(
	"_XQConnectorHolder", parentObject,
	slot_tbl, 1,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp__XQConnectorHolder.setMetaObject( metaObj );
    return metaObj;
}

void* _XQConnectorHolder::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "_XQConnectorHolder" ) )
	return this;
    return QObject::qt_cast( clname );
}

bool _XQConnectorHolder::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: destroyed(); break;
    default:
	return QObject::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool _XQConnectorHolder::qt_emit( int _id, QUObject* _o )
{
    return QObject::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool _XQConnectorHolder::qt_property( int id, int f, QVariant* v)
{
    return QObject::qt_property( id, f, v);
}

bool _XQConnectorHolder::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES
