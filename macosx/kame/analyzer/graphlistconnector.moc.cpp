/****************************************************************************
** XGraphListConnector meta object code from reading C++ file 'graphlistconnector.h'
**
** Created: Wed Feb 1 04:02:28 2006
**      by: The Qt MOC ($Id: graphlistconnector.moc.cpp,v 1.1 2006/02/01 18:45:27 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#undef QT_NO_COMPAT
#include "../../../kame/analyzer/graphlistconnector.h"
#include <qmetaobject.h>
#include <qapplication.h>

#include <private/qucomextra_p.h>
#if !defined(Q_MOC_OUTPUT_REVISION) || (Q_MOC_OUTPUT_REVISION != 26)
#error "This file was generated using the moc from 3.3.5. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

const char *XGraphListConnector::className() const
{
    return "XGraphListConnector";
}

QMetaObject *XGraphListConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XGraphListConnector( "XGraphListConnector", &XGraphListConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XGraphListConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XGraphListConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XGraphListConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XGraphListConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XGraphListConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XListQConnector::staticMetaObject();
    static const QUParameter param_slot_0[] = {
	{ "row", &static_QUType_int, 0, QUParameter::In },
	{ "col", &static_QUType_int, 0, QUParameter::In },
	{ "button", &static_QUType_int, 0, QUParameter::In },
	{ 0, &static_QUType_varptr, "\x0e", QUParameter::In }
    };
    static const QUMethod slot_0 = {"clicked", 4, param_slot_0 };
    static const QMetaData slot_tbl[] = {
	{ "clicked(int,int,int,const QPoint&)", &slot_0, QMetaData::Protected }
    };
    metaObj = QMetaObject::new_metaobject(
	"XGraphListConnector", parentObject,
	slot_tbl, 1,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XGraphListConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XGraphListConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XGraphListConnector" ) )
	return this;
    return XListQConnector::qt_cast( clname );
}

bool XGraphListConnector::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: clicked((int)static_QUType_int.get(_o+1),(int)static_QUType_int.get(_o+2),(int)static_QUType_int.get(_o+3),(const QPoint&)*((const QPoint*)static_QUType_ptr.get(_o+4))); break;
    default:
	return XListQConnector::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool XGraphListConnector::qt_emit( int _id, QUObject* _o )
{
    return XListQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XGraphListConnector::qt_property( int id, int f, QVariant* v)
{
    return XListQConnector::qt_property( id, f, v);
}

bool XGraphListConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES
