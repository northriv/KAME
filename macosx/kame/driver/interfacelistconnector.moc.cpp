/****************************************************************************
** XInterfaceListConnector meta object code from reading C++ file 'interfacelistconnector.h'
**
** Created: Thu Mar 2 16:25:10 2006
**      by: The Qt MOC ($Id: interfacelistconnector.moc.cpp,v 1.1.2.1 2006/03/02 09:19:33 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#undef QT_NO_COMPAT
#include "../../../kame/driver/interfacelistconnector.h"
#include <qmetaobject.h>
#include <qapplication.h>

#include <private/qucomextra_p.h>
#if !defined(Q_MOC_OUTPUT_REVISION) || (Q_MOC_OUTPUT_REVISION != 26)
#error "This file was generated using the moc from 3.3.5. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

const char *XInterfaceListConnector::className() const
{
    return "XInterfaceListConnector";
}

QMetaObject *XInterfaceListConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XInterfaceListConnector( "XInterfaceListConnector", &XInterfaceListConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XInterfaceListConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XInterfaceListConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XInterfaceListConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XInterfaceListConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XInterfaceListConnector::staticMetaObject()
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
	"XInterfaceListConnector", parentObject,
	slot_tbl, 1,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XInterfaceListConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XInterfaceListConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XInterfaceListConnector" ) )
	return this;
    return XListQConnector::qt_cast( clname );
}

bool XInterfaceListConnector::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: clicked((int)static_QUType_int.get(_o+1),(int)static_QUType_int.get(_o+2),(int)static_QUType_int.get(_o+3),(const QPoint&)*((const QPoint*)static_QUType_ptr.get(_o+4))); break;
    default:
	return XListQConnector::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool XInterfaceListConnector::qt_emit( int _id, QUObject* _o )
{
    return XListQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XInterfaceListConnector::qt_property( int id, int f, QVariant* v)
{
    return XListQConnector::qt_property( id, f, v);
}

bool XInterfaceListConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES
