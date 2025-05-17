#pragma once

namespace ObjectDetection {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Summary for MyForm
	/// </summary>
	public ref class MyForm : public System::Windows::Forms::Form
	{
	public:
		MyForm(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
		}
	public:
		String^ chosenModel;
	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~MyForm()
		{
			if (components)
			{
				delete components;
			}
		}
	//main menu ui
	private: 
		System::Windows::Forms::GroupBox^ mainMenu;
		System::Windows::Forms::Button^ useModel;
		System::Windows::Forms::Button^ trainModel;

		//choose model ui
		System::Windows::Forms::GroupBox^ models;
		System::Windows::Forms::Label^ chooseModel;
		System::Windows::Forms::ListView^ modelsView;

		System::Windows::Forms::Label^ selected;
		System::Windows::Forms::Button^ confirmModel;

		//train model ui
		System::Windows::Forms::GroupBox^ train;

		System::Windows::Forms::Button^ back;
	
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container ^components;

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->back = (gcnew System::Windows::Forms::Button());
			//main menu
			this->mainMenu = (gcnew System::Windows::Forms::GroupBox());
			this->useModel = (gcnew System::Windows::Forms::Button());
			this->trainModel = (gcnew System::Windows::Forms::Button());
			//choose model menu
			this->models = (gcnew System::Windows::Forms::GroupBox());
			this->confirmModel = (gcnew System::Windows::Forms::Button());
			this->selected = (gcnew System::Windows::Forms::Label());
			this->modelsView = (gcnew System::Windows::Forms::ListView());
			this->chooseModel = (gcnew System::Windows::Forms::Label());
			//train model menu
			this->train = (gcnew System::Windows::Forms::GroupBox());
			this->mainMenu->SuspendLayout();
			this->models->SuspendLayout();
			this->SuspendLayout();
			// 
			// back
			// 
			this->back->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 10));
			this->back->Location = System::Drawing::Point(12, 17);
			this->back->Name = L"back";
			this->back->Size = System::Drawing::Size(81, 42);
			this->back->TabIndex = 2;
			this->back->Text = L"<-Back";
			this->back->UseVisualStyleBackColor = true;
			this->back->Click += gcnew System::EventHandler(this, &MyForm::back_Click);
			// 
			// mainMenu
			// 
			this->mainMenu->Controls->Add(this->useModel);
			this->mainMenu->Controls->Add(this->trainModel);
			this->mainMenu->Location = System::Drawing::Point(0, 0);
			this->mainMenu->Name = L"mainMenu";
			this->mainMenu->Size = System::Drawing::Size(1920, 1080);
			this->mainMenu->TabIndex = 0;
			this->mainMenu->TabStop = false;
			// 
			// useModel
			// 
			this->useModel->BackColor = System::Drawing::SystemColors::ActiveCaptionText;
			this->useModel->FlatAppearance->BorderSize = 0;
			this->useModel->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 20));
			this->useModel->ForeColor = System::Drawing::SystemColors::ButtonFace;
			this->useModel->Location = System::Drawing::Point(1190, 468);
			this->useModel->Name = L"useModel";
			this->useModel->Size = System::Drawing::Size(246, 124);
			this->useModel->TabIndex = 0;
			this->useModel->Text = L"Use existing model";
			this->useModel->UseVisualStyleBackColor = false;
			this->useModel->Click += gcnew System::EventHandler(this, &MyForm::useModel_Click);
			// 
			// trainModel
			// 
			this->trainModel->BackColor = System::Drawing::SystemColors::ActiveCaptionText;
			this->trainModel->FlatAppearance->BorderSize = 0;
			this->trainModel->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 20));
			this->trainModel->ForeColor = System::Drawing::SystemColors::ButtonFace;
			this->trainModel->Location = System::Drawing::Point(730, 468);
			this->trainModel->Name = L"trainModel";
			this->trainModel->Size = System::Drawing::Size(246, 124);
			this->trainModel->TabIndex = 1;
			this->trainModel->Text = L"Train new model";
			this->trainModel->UseVisualStyleBackColor = false;
			this->trainModel->Click += gcnew System::EventHandler(this, &MyForm::trainModel_Click);
			// 
			// models
			// 
			this->models->Controls->Add(this->confirmModel);
			this->models->Controls->Add(this->selected);
			this->models->Controls->Add(this->modelsView);
			this->models->Controls->Add(this->chooseModel);
			this->models->Controls->Add(this->back);
			this->models->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 20));
			this->models->Location = System::Drawing::Point(0, 0);
			this->models->Name = L"models";
			this->models->Size = System::Drawing::Size(1920, 1080);
			this->models->TabIndex = 0;
			this->models->TabStop = false;
			// 
			// confirmModel
			// 
			this->confirmModel->BackColor = System::Drawing::Color::Lime;
			this->confirmModel->Location = System::Drawing::Point(1100, 952);
			this->confirmModel->Name = L"confirmModel";
			this->confirmModel->Size = System::Drawing::Size(154, 40);
			this->confirmModel->TabIndex = 4;
			this->confirmModel->Text = L"Confirm";
			this->confirmModel->UseVisualStyleBackColor = false;
			this->confirmModel->Visible = false;
			this->confirmModel->Click += gcnew System::EventHandler(this, &MyForm::confirmModel_click);
			// 
			// selected
			// 
			this->selected->AutoSize = true;
			this->selected->Location = System::Drawing::Point(876, 917);
			this->selected->Name = L"selected";
			this->selected->Size = System::Drawing::Size(0, 31);
			this->selected->TabIndex = 3;
			// 
			// modelsView
			// 
			this->modelsView->HideSelection = false;
			this->modelsView->Location = System::Drawing::Point(112, 37);
			this->modelsView->MultiSelect = false;
			this->modelsView->Name = L"modelsView";
			this->modelsView->Scrollable = false;
			this->modelsView->Size = System::Drawing::Size(671, 737);
			this->modelsView->TabIndex = 1;
			this->modelsView->TileSize = System::Drawing::Size(20, 20);
			this->modelsView->UseCompatibleStateImageBehavior = false;
			this->modelsView->Visible = false;
			// 
			// chooseModel
			// 
			this->chooseModel->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 20));
			this->chooseModel->Location = System::Drawing::Point(829, 9);
			this->chooseModel->Name = L"chooseModel";
			this->chooseModel->Size = System::Drawing::Size(288, 50);
			this->chooseModel->TabIndex = 0;
			this->chooseModel->Text = L"Select model to use";
			// 
			// train
			// 
			this->train->Location = System::Drawing::Point(0, 0);
			this->train->Name = L"train";
			this->train->Size = System::Drawing::Size(1920, 1080);
			this->train->TabIndex = 0;
			this->train->TabStop = false;
			// 
			// MyForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(1904, 1041);
			this->Controls->Add(this->mainMenu);
			this->Name = L"MyForm";
			this->Text = L"MyForm";
			this->mainMenu->ResumeLayout(false);
			this->models->ResumeLayout(false);
			this->models->PerformLayout();
			this->ResumeLayout(false);

		}
		System::Void useModel_Click(System::Object^ sender, System::EventArgs^ e) {
			this->Controls->Remove(this->mainMenu);
			this->Controls->Add(this->models);

			if(this->train->Controls->Contains(back))
				this->train->Controls->Remove(back);

			if (!this->models->Controls->Contains(back))
				this->models->Controls->Add(back);

			for (int i = 0; i < 5; i++) {
				Button^ model = gcnew System::Windows::Forms::Button();
				model->TabIndex = 2;
				model->UseVisualStyleBackColor = true;
				model->Name = i.ToString();
				model->Text = L"a model";
				model->Size = System::Drawing::Size(200, 50);
				model->Location = System::Drawing::Point(122, 49+ i * model->Size.Height + 5);
				model->Click += gcnew System::EventHandler(this, &MyForm::modelSelected);
				//model->BackColor = System::Drawing::SystemColors::ActiveCaptionText;
				this->models->Controls->Add(model);
			}
		}

		System::Void modelSelected(System::Object^ sender, System::EventArgs^ e) {
			Button^ clickedButton = dynamic_cast<Button^>(sender);
			this->confirmModel->Visible = true;
			this->selected->Text = "Selected model: " + clickedButton->Name;
			//MessageBox::Show(clickedButton->Name);
		}
		System::Void confirmModel_click(System::Object^ sender, System::EventArgs^ e) {
			String^ subString = "Selected model: ";
			chosenModel = this->selected->Text->Substring(subString->Length);
			MessageBox::Show(chosenModel);
		}

		System::Void trainModel_Click(System::Object^ sender, System::EventArgs^ e) {
			this->models->Controls->Remove(back);
			this->train->Controls->Add(back);
		}
		System::Void back_Click(System::Object^ sender, System::EventArgs^ e) {
			if (this->Controls->Contains(models))
				this->Controls->Remove(models);
			if (this->Controls->Contains(train))
				this->Controls->Remove(train);

			this->Controls->Add(mainMenu);
		}
	};
}
