#include "dynamicfusionui.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	DynamicFusionUI w;
	w.show();
	return a.exec();
}
