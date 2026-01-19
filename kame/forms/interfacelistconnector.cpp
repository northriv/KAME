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
//---------------------------------------------------------------------------
#include "interfacelistconnector.h"
#include "driver.h"
#include "icon.h"

#include <QLineEdit>
#include <QComboBox>
#include <QPushButton>
#include <QSpinBox>
#include <QTableWidget>
#include <QApplication>
#include <QMouseEvent>
#include <QMenu>
#include <QDir>

#if defined WINDOWS || defined __WIN32__ || defined _WIN32
    #include <windows.h>
#endif

XInterfaceListConnector::XInterfaceListConnector(
    const shared_ptr<XInterfaceList> &node, QTableWidget *item)
	: XListQConnector(node, item), m_interfaceList(node) {
    connect(m_pItem, SIGNAL(cellClicked( int, int)),
            this, SLOT(cellClicked( int, int)) );
	item->setColumnCount(5);
	double def = 45;
	item->setColumnWidth(0, (int)(def * 1.5));
    item->setColumnWidth(1, (int)(def * 1.5));
	item->setColumnWidth(2, (int)(def * 2));
    item->setColumnWidth(3, (int)(def * 3));
	item->setColumnWidth(4, (int)(def * 1));
	QStringList labels;
	labels += i18n("Driver");
	labels += i18n("Control");
	labels += i18n("Device");
	labels += i18n("Port");
	labels += i18n("Addr");
	item->setHorizontalHeaderLabels(labels);

	Snapshot shot( *node);
	if(shot.size()) {
		for(int idx = 0; idx < shot.size(); ++idx) {
			XListNodeBase::Payload::CatchEvent e;
			e.emitter = node.get();
			e.caught = shot.list()->at(idx);
			e.index = idx;
			onCatch(shot, e);
		}
	}
}

void
XInterfaceListConnector::onControlChanged(const Snapshot &shot, XValueNodeBase *node) {
	for(auto it = m_cons.begin(); it != m_cons.end(); it++) {
        if(it->xinterface->control().get() == node) {
            if(shot[ *it->xinterface->control()]) {
                if(shot[ *node].isUIEnabled()) {
                    it->btn->setIcon( QIcon( *g_pIconRotate) );
                    it->btn->setText(i18n("&STOP"));
                }
                else {
                    it->btn->setIcon(QApplication::style()->standardIcon(QStyle::SP_MediaStop));
                    it->btn->setText(i18n("&STOP"));
                }
			}
			else {
                it->btn->setIcon(
                    QApplication::style()->standardIcon(QStyle::SP_MediaPlay));
                it->btn->setText(i18n("&RUN"));
			}
		}
	}
}
void
XInterfaceListConnector::onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e) {
    auto xinterface = static_pointer_cast<XInterface>(e.caught);
	int i = m_pItem->rowCount();
	m_pItem->insertRow(i);
    m_pItem->setItem(i, 0, new QTableWidgetItem(xinterface->getLabel().c_str()));
	m_cons.push_back(tcons());
	tcons &con(m_cons.back());
    con.xinterface = xinterface;
	con.btn = new QPushButton(m_pItem);
	con.btn->setCheckable(true);
	con.btn->setAutoDefault(false);
	con.btn->setFlat(true);
    con.concontrol = xqcon_create<XQToggleButtonConnector>(xinterface->control(), con.btn);
	m_pItem->setCellWidget(i, 1, con.btn);
    QComboBox *cmbdev(new QComboBox(m_pItem) );
    con.condev = xqcon_create<XQComboBoxConnector>(xinterface->device(), cmbdev, Snapshot( *xinterface->device()));
    m_pItem->setCellWidget(i, 2, cmbdev);
    con.edport = new QLineEdit(m_pItem);
    con.edport->installEventFilter(this); //for popup.
    con.conport = xqcon_create<XQLineEditConnector>(xinterface->port(), con.edport, false);
    m_pItem->setCellWidget(i, 3, con.edport);
    QSpinBox *numAddr(new QSpinBox(m_pItem) );
    //Ranges should be preset in prior to connectors.
    numAddr->setRange(0, 255);
	numAddr->setSingleStep(1);
    con.conaddr = xqcon_create<XQSpinBoxUnsignedConnector>(xinterface->address(), numAddr);
	m_pItem->setCellWidget(i, 4, numAddr);
    {
        Snapshot shot = xinterface->iterate_commit([=, &con](Transaction &tr){
            con.lsnOnControlChanged = tr[ *xinterface->control()].onValueChanged().connectWeakly(
                shared_from_this(), &XInterfaceListConnector::onControlChanged,
                Listener::FLAG_MAIN_THREAD_CALL);
        });
        onControlChanged(shot, xinterface->control().get());
    }
}
void
XInterfaceListConnector::onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e) {
	for(auto it = m_cons.begin(); it != m_cons.end();) {
        if(it->xinterface == e.released) {
			for(int i = 0; i < m_pItem->rowCount(); i++) {
				if(m_pItem->cellWidget(i, 1) == it->btn) m_pItem->removeRow(i);
			}
			it = m_cons.erase(it);
		}
		else {
			it++;
		}    
	}  
}
void
XInterfaceListConnector::cellClicked ( int row, int ) {
    for(auto &&con: m_cons) {
        if(m_pItem->cellWidget(row, 1) == con.btn)
            con.xinterface->driver()->showForms();
	}
}
bool
XInterfaceListConnector::eventFilter(QObject *obj, QEvent *event) {
    for(auto &&con: m_cons) {
        if(obj == con.edport) {
            Snapshot shot( *con.xinterface->device());
            bool has_ni4882 = false;
            auto items = shot[ *con.xinterface->device()].itemStrings();
            for(auto &&item: items) {
                if(item.label == "PrologixGPIB")
                    has_ni4882= true; //Both NI488.2 and PrologixGPIB exists.
            }
            XString dev = shot[ *con.xinterface->device()].to_str();
            switch (event->type()) {
            case QEvent::MouseButtonPress:
                if((dev == "SERIAL") || ( !has_ni4882 && (dev == "GPIB")) || (dev == "PrologixGPIB")) {
                    auto *mouseevent = static_cast<QMouseEvent *>(event);
                    if(((mouseevent->button() == Qt::LeftButton) && (con.edport->text().isEmpty())) ||
                            (mouseevent->button() == Qt::RightButton)) {
                        auto menu = std::make_unique<QMenu>(con.edport);
                        std::map<XString, XString> map_ui_dev;
                        //listing serial port devices.
#if defined __MACOSX__ || defined __APPLE__ || defined __linux__
                        QStringList filters;
                        filters << "ttyUSB*" << "ttyACM*" << "tty.usbserial-*"; //<< "ttyS*"
                        QStringList ttys = QDir("/dev").entryList(filters,
                            QDir::Files | QDir::System | QDir::NoDotAndDotDot, QDir::Time); //QDir::Name
                        foreach(QString tty, ttys) {
                            map_ui_dev.insert({tty, "/dev/" + tty});
                            menu->addAction(tty);
                        }
#endif
#if defined WINDOWS || defined __WIN32__ || defined _WIN32
                        HKEY hKey;
                        if(RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DEVICEMAP\\SERIALCOMM", 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
                            char valueName[256];
                            BYTE data[256];
                            for(DWORD i = 0; ; i++) {
                                DWORD valLen = sizeof(valueName), dataLen = sizeof(data), type;
                                LONG res = RegEnumValueA(hKey, i, valueName, &valLen, NULL, &type, data, &dataLen);
                                if(res == ERROR_NO_MORE_ITEMS)
                                    break;
                                if(res == ERROR_SUCCESS) {
                                    XString name(reinterpret_cast<char*>(valueName));
                                    XString comstr(reinterpret_cast<char*>(data));
                                    map_ui_dev.insert({comstr, comstr});
                                    menu->addAction(comstr);
                                }
                            }
                            RegCloseKey(hKey);
                        }
//The following can obtain friendly name but, may require some DLL.
//                        HDEVINFO hDevInfo = SetupDiGetClassDevs( &GUID_DEVINTERFACE_COMPORT, NULL, NULL,
//                            DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
//                        if(hDevInfo == INVALID_HANDLE_VALUE)
//                            break;

//                        SP_DEVINFO_DATA devInfoData;
//                        devInfoData.cbSize = sizeof(SP_DEVINFO_DATA);

//                        for (DWORD i = 0; SetupDiEnumDeviceInfo(hDevInfo, i, &devInfoData); i++) {
//                            char friendlyName[256];
//                            if(SetupDiGetDeviceRegistryPropertyA(hDevInfo, &devInfoData, SPDRP_FRIENDLYNAME, NULL,
//                                (PBYTE)friendlyName, sizeof(friendlyName), NULL)) {
//                                std::string str(friendlyName);
//                                auto leftpos = str.rfind("(") + 1;
//                                auto simple = str.substr(leftpos, str.find(")", leftpos));
//                                map_ui_dev.insert({friendlyName, simple});
//                                menu->addAction(friendlyName);
//                            }
//                        }
//                        SetupDiDestroyDeviceInfoList(hDevInfo);
#endif

                        if( !menu->isEmpty()) {
                            auto *action = menu->exec(con.edport->mapToGlobal(mouseevent->pos()));
                            try {
                                if(action)
                                    con.edport->setText(map_ui_dev.at(action->text()));

                                con.edport->setContextMenuPolicy(Qt::NoContextMenu); //No standard context menu pls.
                                return true;
                            }
                            catch (std::out_of_range &) {
                            }
                        }
                    }
                }
                con.edport->setContextMenuPolicy(Qt::DefaultContextMenu); //for copy-n-paste etc.
            default:
                break;
            }
        }
    }
    return QObject::eventFilter(obj, event);
}
