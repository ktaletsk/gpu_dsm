#include "mainwindow.h"
#include <QApplication>

extern int gpu_check();
extern int main_cuda(bool* run_flag, int job_ID, char *savefile, char *loadfile, int device_ID, bool distr);

int main(int argc, char *argv[])
{
    int gpu_status = gpu_check();

    if (argc > 1) { //Console mode
        qDebug() << "console mode";

        bool* run_flag = new bool(true);
        int job_ID = 0;
        char *savefile = NULL;
        char *loadfile = NULL;
        int device_ID = 0;
        bool distr = false;

        //Processing command line parameters
        int k = 1;
        while (k < argc) {
            if (k == 1) {	//processing job_ID
                job_ID = atoi(argv[k]);
            }
            if ((strcmp(argv[k], "-s") == 0) && (k + 1 < argc)) {	//Save file
                savefile = new char[strlen(argv[k + 1]) + 1];
                strcpy(savefile, argv[k + 1]);
                qDebug() << "final chain conformations will be saved to " << savefile
                        << '\n';
                k++;
            }
            if ((strcmp(argv[k], "-l") == 0) && (k + 1 < argc)) {	//Load file
                loadfile = new char[strlen(argv[k + 1]) + 1];
                strcpy(loadfile, argv[k + 1]);
                k++;
            }
            if ((strcmp(argv[k], "-d") == 0) && (k + 1 < argc)) {//Determine nvidia gpu device_ID
                device_ID = atoi(argv[k + 1]);
                k++;
            }
            if ((strcmp(argv[k], "-distr") == 0) && (k < argc)) {
                distr = 1;
            }
            k++;
        }

        main_cuda(run_flag,job_ID,savefile,loadfile,device_ID,distr);
        return 0;
    }
    else { //GUI mode
        QApplication a(argc, argv);
        a.setStyle("fusion");

        if(gpu_status==-1)
        {
            QMessageBox::information(0,"GPU check report",QString("You don't have NVIDIA GPU"));
            return -1;
        }
        else if (gpu_status == -2)
        {
            QMessageBox::information(0,"GPU check report",QString("NVIDIA driver version is insufficient. Update your driver at http://www.geforce.com/drivers"));
            return -1;
        }

        MainWindow w;
        QObject::connect(&a, SIGNAL(lastWindowClosed()), &w, SLOT(on_MainWindow_destroyed()));
        w.show();

        return a.exec();

    }
}
