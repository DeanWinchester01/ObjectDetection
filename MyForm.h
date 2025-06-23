#pragma once
#include <stdlib.h>
#include <exception>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <Windows.h>
#include <string>
#include <unordered_map>
#include <msclr/marshal_cppstd.h>

namespace ObjectDetection {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::IO;
	using namespace System::Diagnostics;
	using namespace Microsoft::WindowsAPICodePack::Dialogs;
	using namespace System::Collections::Generic;
	using namespace System::Threading;

	/// <summary>
	/// Summary for MyForm
	/// </summary>
	public ref class MyForm : public System::Windows::Forms::Form
	{
	public:
		MyForm(void)
		{
			InitializeComponent();
		}
		
		   
	private:
		//global variables
		List<String^>^ imageClasses = gcnew List<String^>();
		List<String^>^ folders = gcnew List<String^>();
		String^ lastDetectedLine;
		StreamWriter^ py;
		Bitmap^ img;
		String^ scanImage;
		String^ chosenModel;

		//non interactive ui
		System::Windows::Forms::Label^ label1;
		System::Windows::Forms::Label^ label3;
		System::Windows::Forms::Label^ label2;

		//interactive system uis below
		System::Windows::Forms::GroupBox^ mainMenu;
		System::Windows::Forms::Button^ useModel;
		System::Windows::Forms::Button^ trainModel;

		//choose model ui
		System::Windows::Forms::GroupBox^ models;
		System::Windows::Forms::Label^ chooseModel;
		System::Windows::Forms::ListView^ modelsView;

		System::Windows::Forms::Label^ selected;
		System::Windows::Forms::Button^ confirmModel;
		System::Windows::Forms::Button^ imageSelect;
		System::Windows::Forms::PictureBox^ chosenImage;

		//train model ui
		System::Windows::Forms::GroupBox^ train;
		System::Windows::Forms::Label^ modelNameLabel;
		System::Windows::Forms::TextBox^ modelName;
		System::Windows::Forms::Button^ selectImages;
		System::Windows::Forms::Label^ inputImageClass;
		System::Windows::Forms::Label^ imageFolder;
		System::Windows::Forms::Button^ confirmClass;

		System::Windows::Forms::TextBox^ imageClass;
		System::Windows::Forms::Button^ startTraining;

		System::Windows::Forms::GroupBox^ trainScreen;
		System::Windows::Forms::Label^ output;
		System::Windows::Forms::Label^ DetectOutput;
		System::Windows::Forms::Button^ detect;

		System::Windows::Forms::Button^ back;

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->back = (gcnew System::Windows::Forms::Button());
			this->mainMenu = (gcnew System::Windows::Forms::GroupBox());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->useModel = (gcnew System::Windows::Forms::Button());
			this->trainModel = (gcnew System::Windows::Forms::Button());
			this->models = (gcnew System::Windows::Forms::GroupBox());
			this->DetectOutput = (gcnew System::Windows::Forms::Label());
			this->detect = (gcnew System::Windows::Forms::Button());
			this->chosenImage = (gcnew System::Windows::Forms::PictureBox());
			this->imageSelect = (gcnew System::Windows::Forms::Button());
			this->confirmModel = (gcnew System::Windows::Forms::Button());
			this->selected = (gcnew System::Windows::Forms::Label());
			this->modelsView = (gcnew System::Windows::Forms::ListView());
			this->chooseModel = (gcnew System::Windows::Forms::Label());
			this->train = (gcnew System::Windows::Forms::GroupBox());
			this->confirmClass = (gcnew System::Windows::Forms::Button());
			this->imageFolder = (gcnew System::Windows::Forms::Label());
			this->inputImageClass = (gcnew System::Windows::Forms::Label());
			this->modelNameLabel = (gcnew System::Windows::Forms::Label());
			this->modelName = (gcnew System::Windows::Forms::TextBox());
			this->selectImages = (gcnew System::Windows::Forms::Button());
			this->imageClass = (gcnew System::Windows::Forms::TextBox());
			this->startTraining = (gcnew System::Windows::Forms::Button());
			this->output = (gcnew System::Windows::Forms::Label());
			this->trainScreen = (gcnew System::Windows::Forms::GroupBox());
			this->mainMenu->SuspendLayout();
			this->models->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->chosenImage))->BeginInit();
			this->train->SuspendLayout();
			this->SuspendLayout();
			// 
			// back
			// 
			this->back->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(26)), static_cast<System::Int32>(static_cast<System::Byte>(28)),
				static_cast<System::Int32>(static_cast<System::Byte>(34)));
			this->back->FlatAppearance->BorderColor = System::Drawing::Color::White;
			this->back->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->back->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 10));
			this->back->ForeColor = System::Drawing::SystemColors::ButtonHighlight;
			this->back->Location = System::Drawing::Point(12, 17);
			this->back->Name = L"back";
			this->back->Size = System::Drawing::Size(81, 42);
			this->back->TabIndex = 2;
			this->back->Text = L"<-Back";
			this->back->UseVisualStyleBackColor = false;
			this->back->Click += gcnew System::EventHandler(this, &MyForm::back_Click);
			// 
			// mainMenu
			// 
			this->mainMenu->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(13)), static_cast<System::Int32>(static_cast<System::Byte>(17)),
				static_cast<System::Int32>(static_cast<System::Byte>(23)));
			this->mainMenu->Controls->Add(this->label3);
			this->mainMenu->Controls->Add(this->label2);
			this->mainMenu->Controls->Add(this->label1);
			this->mainMenu->Controls->Add(this->useModel);
			this->mainMenu->Controls->Add(this->trainModel);
			this->mainMenu->Location = System::Drawing::Point(0, 0);
			this->mainMenu->Name = L"mainMenu";
			this->mainMenu->Size = System::Drawing::Size(1920, 1080);
			this->mainMenu->TabIndex = 0;
			this->mainMenu->TabStop = false;
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 10));
			this->label3->ForeColor = System::Drawing::SystemColors::ControlDarkDark;
			this->label3->Location = System::Drawing::Point(978, 637);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(159, 17);
			this->label3->TabIndex = 4;
			this->label3->Text = L"v1.0 By Nico Andersson";
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 10));
			this->label2->ForeColor = System::Drawing::SystemColors::ControlDarkDark;
			this->label2->Location = System::Drawing::Point(978, 219);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(167, 17);
			this->label2->TabIndex = 3;
			this->label2->Text = L"Train your own AI models";
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 25));
			this->label1->ForeColor = System::Drawing::SystemColors::ButtonHighlight;
			this->label1->Location = System::Drawing::Point(902, 141);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(324, 39);
			this->label1->TabIndex = 2;
			this->label1->Text = L"Medical AI Assistant";
			// 
			// useModel
			// 
			this->useModel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(26)), static_cast<System::Int32>(static_cast<System::Byte>(28)),
				static_cast<System::Int32>(static_cast<System::Byte>(34)));
			this->useModel->FlatAppearance->BorderSize = 0;
			this->useModel->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->useModel->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 20));
			this->useModel->ForeColor = System::Drawing::SystemColors::ButtonFace;
			this->useModel->Location = System::Drawing::Point(937, 282);
			this->useModel->Name = L"useModel";
			this->useModel->Size = System::Drawing::Size(246, 124);
			this->useModel->TabIndex = 0;
			this->useModel->Text = L"Use existing model";
			this->useModel->UseVisualStyleBackColor = false;
			this->useModel->Click += gcnew System::EventHandler(this, &MyForm::useModel_Click);
			// 
			// trainModel
			// 
			this->trainModel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(26)), static_cast<System::Int32>(static_cast<System::Byte>(28)),
				static_cast<System::Int32>(static_cast<System::Byte>(34)));
			this->trainModel->FlatAppearance->BorderSize = 0;
			this->trainModel->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->trainModel->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 20));
			this->trainModel->ForeColor = System::Drawing::SystemColors::ButtonFace;
			this->trainModel->Location = System::Drawing::Point(937, 464);
			this->trainModel->Name = L"trainModel";
			this->trainModel->Size = System::Drawing::Size(246, 124);
			this->trainModel->TabIndex = 1;
			this->trainModel->Text = L"Train new model";
			this->trainModel->UseVisualStyleBackColor = false;
			this->trainModel->Click += gcnew System::EventHandler(this, &MyForm::trainModel_Click);
			// 
			// models
			// 
			this->models->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(13)), static_cast<System::Int32>(static_cast<System::Byte>(17)),
				static_cast<System::Int32>(static_cast<System::Byte>(23)));
			this->models->Controls->Add(this->DetectOutput);
			this->models->Controls->Add(this->detect);
			this->models->Controls->Add(this->chosenImage);
			this->models->Controls->Add(this->imageSelect);
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
			// DetectOutput
			// 
			this->DetectOutput->AutoSize = true;
			this->DetectOutput->ForeColor = System::Drawing::SystemColors::ButtonFace;
			this->DetectOutput->Location = System::Drawing::Point(500, 100);
			this->DetectOutput->Size = System::Drawing::Size(0, 31);
			this->DetectOutput->Name = L"DetectOutput";
			this->DetectOutput->TabIndex = 8;
			// 
			// detect
			// 
			this->detect->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(26)), static_cast<System::Int32>(static_cast<System::Byte>(28)),
				static_cast<System::Int32>(static_cast<System::Byte>(34)));
			this->detect->FlatAppearance->BorderColor = System::Drawing::Color::White;
			this->detect->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->detect->ForeColor = System::Drawing::SystemColors::ButtonHighlight;
			this->detect->Location = System::Drawing::Point(1218, 100);
			this->detect->Name = L"detect";
			this->detect->Size = System::Drawing::Size(179, 41);
			this->detect->TabIndex = 7;
			this->detect->Text = L"Run";
			this->detect->UseVisualStyleBackColor = false;
			this->detect->Click += gcnew System::EventHandler(this, &MyForm::detect_Click);
			// 
			// chosenImage
			// 
			this->chosenImage->Location = System::Drawing::Point(808, 170);
			this->chosenImage->Name = L"chosenImage";
			this->chosenImage->Size = System::Drawing::Size(745, 487);
			this->chosenImage->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->chosenImage->TabIndex = 6;
			this->chosenImage->TabStop = false;
			// 
			// imageSelect
			// 
			this->imageSelect->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(26)), static_cast<System::Int32>(static_cast<System::Byte>(28)),
				static_cast<System::Int32>(static_cast<System::Byte>(34)));
			this->imageSelect->FlatAppearance->BorderColor = System::Drawing::Color::White;
			this->imageSelect->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->imageSelect->ForeColor = System::Drawing::SystemColors::ButtonHighlight;
			this->imageSelect->Location = System::Drawing::Point(879, 100);
			this->imageSelect->Name = L"imageSelect";
			this->imageSelect->Size = System::Drawing::Size(201, 41);
			this->imageSelect->TabIndex = 5;
			this->imageSelect->Text = L"Select image";
			this->imageSelect->UseVisualStyleBackColor = false;
			this->imageSelect->Click += gcnew System::EventHandler(this, &MyForm::imageSelect_Click);
			// 
			// confirmModel
			// 
			this->confirmModel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(26)), static_cast<System::Int32>(static_cast<System::Byte>(28)),
				static_cast<System::Int32>(static_cast<System::Byte>(34)));
			this->confirmModel->FlatAppearance->BorderColor = System::Drawing::Color::White;
			this->confirmModel->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->confirmModel->ForeColor = System::Drawing::SystemColors::ButtonHighlight;
			this->confirmModel->Location = System::Drawing::Point(289, 100);
			this->confirmModel->Name = L"confirmModel";
			this->confirmModel->Size = System::Drawing::Size(221, 40);
			this->confirmModel->TabIndex = 8;
			this->confirmModel->Text = L"Confirm model";
			this->confirmModel->UseVisualStyleBackColor = false;
			this->confirmModel->Click += gcnew System::EventHandler(this, &MyForm::confirmModel_click);
			// 
			// selected
			// 
			this->selected->AutoSize = true;
			this->selected->Location = System::Drawing::Point(600, 917);
			this->selected->Name = L"selected";
			this->selected->Size = System::Drawing::Size(0, 31);
			this->selected->TabIndex = 3;
			// 
			// modelsView
			// 
			this->modelsView->HideSelection = false;
			this->modelsView->Location = System::Drawing::Point(112, 170);
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
			this->chooseModel->ForeColor = System::Drawing::SystemColors::ButtonFace;
			this->chooseModel->Location = System::Drawing::Point(829, 9);
			this->chooseModel->Name = L"chooseModel";
			this->chooseModel->Size = System::Drawing::Size(288, 50);
			this->chooseModel->TabIndex = 0;
			this->chooseModel->Text = L"Select model to use";
			// 
			// train
			// 
			this->train->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(30)), static_cast<System::Int32>(static_cast<System::Byte>(30)),
				static_cast<System::Int32>(static_cast<System::Byte>(30)));
			this->train->Controls->Add(this->confirmClass);
			this->train->Controls->Add(this->imageFolder);
			this->train->Controls->Add(this->inputImageClass);
			this->train->Controls->Add(this->modelNameLabel);
			this->train->Controls->Add(this->modelName);
			this->train->Controls->Add(this->selectImages);
			this->train->Controls->Add(this->imageClass);
			this->train->Controls->Add(this->startTraining);
			this->train->Controls->Add(this->output);
			this->train->Location = System::Drawing::Point(0, 0);
			this->train->Name = L"train";
			this->train->Size = System::Drawing::Size(1920, 1080);
			this->train->TabIndex = 0;
			this->train->TabStop = false;
			// 
			// confirmClass
			// 
			this->confirmClass->AutoSize = true;
			this->confirmClass->BackColor = System::Drawing::Color::Lime;
			this->confirmClass->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 15));
			this->confirmClass->ForeColor = System::Drawing::Color::Black;
			this->confirmClass->Location = System::Drawing::Point(94, 621);
			this->confirmClass->Name = L"confirmClass";
			this->confirmClass->Size = System::Drawing::Size(140, 35);
			this->confirmClass->TabIndex = 4;
			this->confirmClass->Text = L"Confirm class";
			this->confirmClass->UseVisualStyleBackColor = false;
			this->confirmClass->Click += gcnew System::EventHandler(this, &MyForm::confirmClass_Click);
			// 
			// imageFolder
			// 
			this->imageFolder->AutoSize = true;
			this->imageFolder->BackColor = System::Drawing::Color::White;
			this->imageFolder->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 15));
			this->imageFolder->Location = System::Drawing::Point(89, 500);
			this->imageFolder->Name = L"imageFolder";
			this->imageFolder->Size = System::Drawing::Size(0, 25);
			this->imageFolder->TabIndex = 3;
			// 
			// inputImageClass
			// 
			this->inputImageClass->AutoSize = true;
			this->inputImageClass->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(64)), static_cast<System::Int32>(static_cast<System::Byte>(64)),
				static_cast<System::Int32>(static_cast<System::Byte>(64)));
			this->inputImageClass->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 20));
			this->inputImageClass->ForeColor = System::Drawing::SystemColors::ButtonFace;
			this->inputImageClass->Location = System::Drawing::Point(86, 249);
			this->inputImageClass->Name = L"inputImageClass";
			this->inputImageClass->Size = System::Drawing::Size(225, 31);
			this->inputImageClass->TabIndex = 2;
			this->inputImageClass->Text = L"Input image class";
			// 
			// modelNameLabel
			// 
			this->modelNameLabel->AutoSize = true;
			this->modelNameLabel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(64)), static_cast<System::Int32>(static_cast<System::Byte>(64)),
				static_cast<System::Int32>(static_cast<System::Byte>(64)));
			this->modelNameLabel->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 20));
			this->modelNameLabel->ForeColor = System::Drawing::SystemColors::ButtonFace;
			this->modelNameLabel->Location = System::Drawing::Point(823, 58);
			this->modelNameLabel->Name = L"modelNameLabel";
			this->modelNameLabel->Size = System::Drawing::Size(226, 31);
			this->modelNameLabel->TabIndex = 0;
			this->modelNameLabel->Text = L"Name your model";
			// 
			// modelName
			// 
			this->modelName->AcceptsReturn = true;
			this->modelName->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 15));
			this->modelName->Location = System::Drawing::Point(846, 112);
			this->modelName->Name = L"modelName";
			this->modelName->Size = System::Drawing::Size(182, 30);
			this->modelName->TabIndex = 0;
			// 
			// selectImages
			// 
			this->selectImages->AutoSize = true;
			this->selectImages->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(64)), static_cast<System::Int32>(static_cast<System::Byte>(64)),
				static_cast<System::Int32>(static_cast<System::Byte>(64)));
			this->selectImages->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 20));
			this->selectImages->ForeColor = System::Drawing::SystemColors::ButtonFace;
			this->selectImages->Location = System::Drawing::Point(92, 431);
			this->selectImages->Name = L"selectImages";
			this->selectImages->Size = System::Drawing::Size(194, 41);
			this->selectImages->TabIndex = 0;
			this->selectImages->Text = L"Select images";
			this->selectImages->UseVisualStyleBackColor = false;
			this->selectImages->Click += gcnew System::EventHandler(this, &MyForm::selectImages_Click);
			// 
			// imageClass
			// 
			this->imageClass->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 15));
			this->imageClass->Location = System::Drawing::Point(92, 317);
			this->imageClass->Name = L"imageClass";
			this->imageClass->Size = System::Drawing::Size(219, 30);
			this->imageClass->TabIndex = 1;
			// 
			// startTraining
			// 
			this->startTraining->AutoSize = true;
			this->startTraining->BackColor = System::Drawing::Color::Lime;
			this->startTraining->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 20));
			this->startTraining->ForeColor = System::Drawing::SystemColors::ControlText;
			this->startTraining->Location = System::Drawing::Point(846, 765);
			this->startTraining->Name = L"startTraining";
			this->startTraining->Size = System::Drawing::Size(189, 66);
			this->startTraining->TabIndex = 0;
			this->startTraining->Text = L"Start training";
			this->startTraining->UseVisualStyleBackColor = false;
			this->startTraining->Click += gcnew System::EventHandler(this, &MyForm::startTraining_Click);
			// 
			// output
			// 
			this->output->AutoSize = true;
			this->output->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 20));
			this->output->Location = System::Drawing::Point(1050, 160);
			this->output->Name = L"output";
			this->output->Size = System::Drawing::Size(0, 31);
			this->output->TabIndex = 0;
			// 
			// trainScreen
			// 
			this->trainScreen->Location = System::Drawing::Point(0, 0);
			this->trainScreen->Name = L"trainScreen";
			this->trainScreen->Size = System::Drawing::Size(200, 100);
			this->trainScreen->TabIndex = 0;
			this->trainScreen->TabStop = false;
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
			this->mainMenu->PerformLayout();
			this->models->ResumeLayout(false);
			this->models->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->chosenImage))->EndInit();
			this->train->ResumeLayout(false);
			this->train->PerformLayout();
			this->ResumeLayout(false);

		}
		Button^ ModelButton(String^ modelAddress, int i) {
			array<String^>^ addressParts = modelAddress->Split('\\');
			String^ lastPart = addressParts[addressParts->Length - 1];
			String^ modelName = lastPart->Substring(0, lastPart->Length - 3);

			Button^ model = gcnew System::Windows::Forms::Button();
			model->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(26)), static_cast<System::Int32>(static_cast<System::Byte>(28)),
				static_cast<System::Int32>(static_cast<System::Byte>(34)));
			model->FlatAppearance->BorderColor = System::Drawing::Color::White;
			model->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			model->ForeColor = System::Drawing::SystemColors::ButtonHighlight;
			model->AutoSize = true;
			model->TabIndex = 2;
			model->UseVisualStyleBackColor = true;
			model->Name = i.ToString();
			model->Text = L"" + modelName + "";
			model->Size = System::Drawing::Size(200, 50);
			model->Location = System::Drawing::Point(122, 135 + i * model->Size.Height + 5);

			return model;
		}

		void DisplayClasses() {
			for (int i = 0; i < imageClasses->Count; i++) {
				//remove old labels
				train->Controls->RemoveByKey(imageClasses[i]);
				train->Controls->RemoveByKey(folders[i]);
			}

			//add new labels
			for (int i = 0; i < imageClasses->Count; i++) {
				Label^ classLabel = gcnew System::Windows::Forms::Label();
				Label^ addressLabel = gcnew System::Windows::Forms::Label();

				classLabel->BackColor = System::Drawing::Color::Lime;
				classLabel->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 15));
				classLabel->Location = System::Drawing::Point(454, 249 + 130 * i);
				classLabel->Name = imageClasses[i];
				classLabel->Size = System::Drawing::Size(219, 30);
				classLabel->TabIndex = 2;
				classLabel->Text = L"Label: " + imageClasses[i];
				classLabel->AutoSize = true;

				addressLabel->BackColor = System::Drawing::Color::Lime;
				addressLabel->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 15));
				addressLabel->Location = System::Drawing::Point(454, 293 + 130 * i);
				addressLabel->Name = folders[i];
				addressLabel->Size = System::Drawing::Size(219, 30);
				addressLabel->TabIndex = 2;
				addressLabel->Text = L"Folder: " + folders[i];
				addressLabel->AutoSize = true;


				this->train->Controls->Add(classLabel);
				this->train->Controls->Add(addressLabel);
			}
		}

		System::Void useModel_Click(System::Object^ sender, System::EventArgs^ e) {
			//show new screen with available models
			this->Controls->Remove(this->mainMenu);
			this->Controls->Add(this->models);

			if (this->train->Controls->Contains(back))
				this->train->Controls->Remove(back);

			if (!this->models->Controls->Contains(back))
				this->models->Controls->Add(back);

			//read in models
			String^ currentPlace = Directory::GetCurrentDirectory();
			array<String^>^ files = Directory::GetFiles(currentPlace + "\\Models");

			//make new buttons
			for (int i = 0; i < files->Length; i++) {
				if (!files[i]->EndsWith(".pt")) continue;

				Button^ model = ModelButton(files[i], i);
				model->Click += gcnew System::EventHandler(this, &MyForm::modelSelected);
				this->models->Controls->Add(model);
			}

			String^ pythonExe = "C:\\Program Files\\Python310\\python.exe";
			//configure process
			Process^ detect = gcnew Process();
			detect->StartInfo->FileName = pythonExe;
			detect->StartInfo->Arguments = "image_Detect.py";
			detect->StartInfo->UseShellExecute = false;
			detect->StartInfo->RedirectStandardOutput = true;
			detect->StartInfo->RedirectStandardError = true;
			detect->StartInfo->CreateNoWindow = true; // Optional
			detect->EnableRaisingEvents = true;
			detect->StartInfo->RedirectStandardInput = true;

			//read python output
			detect->OutputDataReceived += gcnew DataReceivedEventHandler(this, &MyForm::DetectionOutputHandler);
			detect->ErrorDataReceived += gcnew DataReceivedEventHandler(this, &MyForm::DetectionOutputHandler);
			detect->Start();
			detect->BeginOutputReadLine();
			detect->BeginErrorReadLine();
			py = detect->StandardInput;
		}

		System::Void modelSelected(System::Object^ sender, System::EventArgs^ e) {
			//read what dynamically loaded model button was pressed
			Button^ clickedButton = dynamic_cast<Button^>(sender);
			this->confirmModel->Visible = true;
			this->selected->Text = "Selected model: " + clickedButton->Text;
		}
		System::Void confirmModel_click(System::Object^ sender, System::EventArgs^ e) {
			String^ subString = "Selected model: ";
			if (this->selected->Text->Length < subString->Length) {
				MessageBox::Show("Please select a model");
				return;
			}
			chosenModel = this->selected->Text->Substring(subString->Length);
		}

		System::Void trainModel_Click(System::Object^ sender, System::EventArgs^ e) {
			this->Controls->Remove(mainMenu);
			this->models->Controls->Remove(back);
			this->Controls->Add(this->train);
			this->train->Controls->Add(back);

			//activate new python process
			String^ pythonExe = "C:\\Program Files\\Python310\\python.exe";
			//configure process
			Process^ train = gcnew Process();
			train->StartInfo->FileName = pythonExe;
			train->StartInfo->Arguments = "image_train.py";
			train->StartInfo->UseShellExecute = false;
			train->StartInfo->RedirectStandardError = true;
			train->StartInfo->CreateNoWindow = true; // Optional
			train->EnableRaisingEvents = true;
			train->StartInfo->RedirectStandardOutput = true;
			train->StartInfo->RedirectStandardInput = true;

			//read python output
			train->OutputDataReceived += gcnew DataReceivedEventHandler(this, &MyForm::SortOutputHandler);
			train->ErrorDataReceived += gcnew DataReceivedEventHandler(this, &MyForm::SortOutputHandler);
			train->Start();
			train->BeginOutputReadLine();
			train->BeginErrorReadLine();
			py = train->StandardInput;
		}
		System::Void back_Click(System::Object^ sender, System::EventArgs^ e) {
			if (this->Controls->Contains(models))
				this->Controls->Remove(models);
			if (this->Controls->Contains(train))
				this->Controls->Remove(train);

			//close the python process when backing out
			if (py != nullptr)
				py->Close();
			this->Controls->Add(mainMenu);
		}
		System::Void selectImages_Click(System::Object^ sender, System::EventArgs^ e) {
			CommonOpenFileDialog^ dialog = gcnew CommonOpenFileDialog();
			dialog->IsFolderPicker = true;

			if (dialog->ShowDialog() != CommonFileDialogResult::Ok) return;
			imageFolder->Text = dialog->FileName;
		}
		System::Void confirmClass_Click(System::Object^ sender, System::EventArgs^ e) {
			imageClasses->Add(imageClass->Text);
			folders->Add(imageFolder->Text);

			DisplayClasses();
		}
		//
		//training begins
		//
		System::Void startTraining_Click(System::Object^ sender, System::EventArgs^ e) {
			//guard clauses to stop empty training
			if (imageClasses->Count == 0) {
				MessageBox::Show("Add some labels");
				return;
			}
			if (folders->Count == 0) {
				MessageBox::Show("Add some folders containing images you wish to train on");
				return;
			}

			this->output->Text += "\nStarting training";
			this->output->Size = System::Drawing::Size(86, 31);
			this->output->Location = System::Drawing::Point(1050, 160);

			String^ name = modelName->Text+",";

			for (int i = 0; i < folders->Count; i++) {
				name += folders[i] + "\"" + imageClasses[i] + "\"";
			}
			py->WriteLine(name);
			py->Flush();
		}

		void SortOutputHandler(System::Object^ sendingProcess, DataReceivedEventArgs^ outLine){
			if (outLine->Data == nullptr) return;

			//calculate size difference
			int currentSize = this->output->Size.Height;
			this->output->Text += "\n" + outLine->Data;
			int newSize = this->output->Size.Height;

			//move output label if up size difference is too big
			array<String^>^ lines = this->output->Text->Split('\n');
			if (lines->Length > 15) {
				int yPos = this->output->Location.Y;
				int xPos = this->output->Location.X;
				this->output->Location = System::Drawing::Point(xPos, yPos - (newSize - currentSize));
			}

			//training finished, close python stream
			if (outLine->Data->Contains("saved"))
				py->Close();
		}

		//function to ask user for image to scan
		System::Void imageSelect_Click(System::Object^ sender, System::EventArgs^ e) {
			CommonOpenFileDialog^ dialog = gcnew CommonOpenFileDialog();
			dialog->IsFolderPicker = false;
			dialog->Filters->Add(gcnew CommonFileDialogFilter("Image Files", "*.jpg;*.jpeg;*.png;*dicm"));
			
			//clean up old image if it exist
			if (chosenImage->Image != nullptr) {
				chosenImage->Image = nullptr;
			}
			if (img != nullptr) {
				delete(img);
			}

			if (dialog->ShowDialog() != CommonFileDialogResult::Ok) return; //guard clause to stop if user does not select a file
			String^ name = dialog->FileName;
			try {

				img = gcnew Bitmap(name);
			}
			catch (Exception^ e) {
				MessageBox::Show("File failed to be opened, it is likely corrupted\n"+e->ToString());
				return;
			}


			chosenImage->Image = img;
			scanImage = name;
		}

		//clicked button to detect subject in image
		System::Void detect_Click(System::Object^ sender, System::EventArgs^ e) {
			this->DetectOutput->Text = "";
			this->DetectOutput->Location = System::Drawing::Point(500, 100);
			this->DetectOutput->Size = System::Drawing::Size(0, 31);

			if (chosenModel == "") {
				MessageBox::Show("please select a model to use");
				return;
			}
			if (scanImage == "") {
				MessageBox::Show("please select an image to scan");
				return;
			}

			if (chosenImage->Image->ToString()->Contains("testimages")) {
				chosenImage->Image = nullptr;
			}
			
			py->WriteLine(chosenModel + "," + scanImage);
			py->Flush();
		}

		//read and display output from detection process
		void DetectionOutputHandler(System::Object^ sendingProcess,
			DataReceivedEventArgs^ outLine) {
			if (outLine->Data == nullptr) return;
			String^ data = outLine->Data;

			
			std::unordered_map<std::string, float> values = {};
			if (data->Contains("Detected:")) {

				String^ path = Directory::GetCurrentDirectory();
				String^ outputImage = path+ "\\testimages\\originalImage.png";

				
				if (File::Exists(outputImage)) {
					try {

						img = gcnew Bitmap(outputImage);
					}
					catch (Exception^ e) {
						MessageBox::Show("File failed to be opened, it is likely corrupted\n" + e->ToString());
						return;
					}
					chosenImage->Image = img;

				}
				else {
					MessageBox::Show("invalid file");
				}
				
			}

			array<String^>^ parts = data->Split(':');
			for (int i = 0; i < parts->Length; i++) {
				int currentSize = this->DetectOutput->Size.Height;
				this->DetectOutput->Text += parts[i]+"\n";
				int newSize = this->DetectOutput->Size.Height;

				array<String^>^ lines = this->DetectOutput->Text->Split('\n');
				if (lines->Length > 15) {
					int yPos = this->output->Location.Y;
					int xPos = this->output->Location.X;
					this->output->Location = System::Drawing::Point(xPos, yPos - (newSize - currentSize));
				}
			}
		}

	};
}
