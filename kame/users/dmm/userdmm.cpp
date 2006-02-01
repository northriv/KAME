//---------------------------------------------------------------------------

#include "userdmm.h"
#include "interface.h"
//---------------------------------------------------------------------------

void
XDMMSCPI::changeFunction()
{
    QString func = function()->to_str();
    if(!func.isEmpty())
        interface()->sendf(":CONF:%s", func.latin1());
}
double
XDMMSCPI::fetch()
{
    interface()->query(":FETC?");
    return interface()->toDouble();
}
double
XDMMSCPI::oneShotRead()
{
    interface()->query(":READ?");
    return interface()->toDouble();
}
double
XDMMSCPI::measure(const QString &func)
{
    interface()->queryf(":MEAS:%s?", func.latin1());
    return interface()->toDouble();
}
