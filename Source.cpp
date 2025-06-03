#include "MyForm.h"

using namespace System;
using namespace System::Windows::Forms;


[STAThread]
int main(array<String^>^ args) {
    Application::EnableVisualStyles();
    Application::SetCompatibleTextRenderingDefault(false);
    
    ObjectDetection::MyForm form;
    form.Width = 1920;
    form.Height = 1080;
    //MessageBox::Show(__cplusplus.ToString());
    Application::Run(% form);

    return 0;
}