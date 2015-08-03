#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPointer>
#include <QFile>
#include <QDebug>
#include <QTextStream>
#include <QTableWidgetItem>
#include <QProcess>
#include <QMutex>
#include <QObject>
#include "qcustomplot.h"

namespace Ui {
class MainWindow;
}

class QComboBox;
class QCheckBox;
class Worker;

class MainWindow : public QMainWindow
{
    Q_OBJECT
    QStringList Architecturelistfull;
    QStringList Flowlistfull;
    QList<QPointer<QCheckBox> > Deletelist;
    QList<QPointer<QComboBox> > Architecturelist;
    QList<QPointer<QComboBox> > Flowlist;
    QStringList fields;
    QString line;
    QThread* thread;
    Worker* worker;

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
public slots:
    void SlotDetectFinish(int exitCode, QProcess::ExitStatus exitStatus);
private slots:

    void on_add_clicked();
    void on_remove_clicked();
    void on_selectall_clicked();
    void on_deselect_clicked();
    void on_actionOpen_triggered();
    void load_settings();
    void on_actionSave_triggered();
    void save_settings();
    void on_combo_chemistry_probe_currentIndexChanged(int index);
    void on_lineEdit_5_editingFinished();
    void on_lineEdit_4_editingFinished();
    void on_edit_beta_FSM_editingFinished();
    void on_combo_flow_type_currentIndexChanged(int index);
    void on_pushButton_clicked();
    void setuRealtimeData();
    void on_combo_arch_probe_activated(const QString);
    void on_MainWindow_destroyed();
    void on_edit_rate_textChanged(const QString &arg1);
    void on_edit_rate_returnPressed();
    void on_pushButton_save_pdf_clicked();
    void on_pushButton_save_jpg_clicked();
    void on_radio_flow_clicked();
    void on_radio_eq_clicked();

private:
    Ui::MainWindow *ui;
    QString setting_file_name;
};

class Worker : public QObject {
    Q_OBJECT
public:
    explicit Worker(QObject *parent = 0);
    void requestWork(); //Start process
    void abort();
    bool get_working();
private:
    bool _run; // process is aborted when true
    bool _working; // process is doing work when true
     QMutex mutex; //Protects access to _run
public slots:
    void doWork();
signals:
    void workRequested(); //Worker request to Work
    void finished(); //process is finished
};

#endif // MAINWINDOW_H
