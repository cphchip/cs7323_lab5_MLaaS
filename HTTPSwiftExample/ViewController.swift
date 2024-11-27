//
//  ViewController.swift
//  HTTPSwiftExample
//
//  Created by Eric Larson on 3/30/15.
//  Copyright (c) 2015 Eric Larson. All rights reserved.
//  Updated 2024

// This example is meant to be run with the python example:
//              fastapi_turicreate.py
//              from the course GitHub repository

import AVFoundation
import UIKit

class ViewController: UIViewController, AVCapturePhotoCaptureDelegate,
    UITextFieldDelegate
{

    //Ref Cite:  ChatGPT
    // This section of code was generated with the assistance of ChatGPT, an AI language model by OpenAI.
    // Date: 11/22/24
    // Source: OpenAI's ChatGPT (https://openai.com/chatgpt)
    // Prompt: capture a picture taken with phone camera using AVFoundation
    // Modifications: updated to integrate with Mlaas Model

    // MARK: Class Properties

    // interacting with server
    let client = MLClient()  // how we will interact with the server

    // Photo capture properties
    var captureSession: AVCaptureSession!
    var photoOutput: AVCapturePhotoOutput!
    var previewLayer: AVCaptureVideoPreviewLayer!
    var isCameraRunning = false  // To track the camera state


    var objDetectMenuItems: [String] = [] //pull down menu items of objects to detect in images
    var imageCount = 0                    // count of images taken
    var currentObjectSelected = "none"
    var currentResizedImage: UIImage!     // current image to process

    var trainMode: Bool = false           // training mode
    var selectedModel: String = "svc"     // selected model


    // Predict or Train mode selection
    var trainPredictMode: Bool = false    // train/predict selection

    // User Interface properties
    @IBOutlet weak var capturedImageView: UIImageView!
    @IBOutlet weak var cameraFeedView: UIView!
    @IBOutlet weak var StartStopCamera: UIButton!
    @IBOutlet weak var imgCaptureButton: UIButton!
    @IBOutlet weak var imageCountLabel: UILabel!
    @IBOutlet weak var feedbackLabel: UILabel!
    @IBOutlet weak var objDetectPullDown: UIButton!
    @IBOutlet weak var newObjToDetect: UITextField!
    @IBOutlet weak var modelSelector: UISegmentedControl!
    @IBOutlet weak var trainPredictSegControl: UISegmentedControl!

    // MARK: View Controller Life Cycle
    override func viewDidLoad() {
        super.viewDidLoad()

        view.backgroundColor = .white

        // Initial state for the UIImageView
        capturedImageView.contentMode = .scaleAspectFit
        capturedImageView.isHidden = true  // Hide initially until a photo is cap

        // Set the button's initial title
        StartStopCamera.setTitle("Start Camera", for: .normal)

        // use delegation for interacting with client
        client.delegate = self

        // delegate for new ObjectToDetect textFied
        newObjToDetect.delegate = self

        // Get the labels from the server
        let labelDataSets = client.getLabels()

        // Extract labels from array of DataSets - [Dataset]
        let labels = labelDataSets.map { $0.label }
        // Set up the initial menu
        updateMenu(with: labels)
        objDetectMenuItems = labels
        trainMode = trainPredictSegControl.selectedSegmentIndex == 0
        print("Train Mode: \(trainMode)")

        selectedModel = modelSelector.selectedSegmentIndex == 0 ? "svc" : "rf"
        print("Selected Model: \(selectedModel)")

    }

    // Update the pull-down menu dynamically
    func updateMenu(with items: [String]) {
        var menuActions: [UIAction] = []
        print("Items: \(items)")
        if items.isEmpty {
            let defaultAction = UIAction(
                title: "No options available",
                handler: { _ in
                    print("No options available selected")
                })
            menuActions.append(defaultAction)
        } else {
            for item in items {
                let action = UIAction(
                    title: item,
                    handler: { _ in
                        print("\(item) selected")
                        self.updateButtonTitle(with: item)
                    })
                menuActions.append(action)
            }
        }

        let menu = UIMenu(title: "Options", children: menuActions)
        objDetectPullDown.menu = menu
        objDetectPullDown.showsMenuAsPrimaryAction = true
    }

    func updateButtonTitle(with item: String) {
        // Update the button's title to reflect the selected item
        objDetectPullDown.setTitle(item, for: .normal)
        currentObjectSelected = item
    }

    @IBAction func modelSelectValueChanged(_ sender: UISegmentedControl) {
        // Retrieve the selected index
        let selectedIndex = sender.selectedSegmentIndex

        // Get the title of the selected segment (if needed)
        let selectedTitle: String
        switch selectedIndex {
        case 0:
            selectedTitle = "svc"
        case 1:
            selectedTitle = "rf"
        default:
            selectedTitle = "Unknown"
        }

        print("Selected Index: \(selectedIndex)")
        print("Selected Title: \(selectedTitle)")

        selectedModel = selectedTitle

    }

    // Photo capture button pressed. Capture photo
    @IBAction func capturePhotoButtonTapped(_ sender: UIButton) {
        if isCameraRunning {
            let settings = AVCapturePhotoSettings()
            photoOutput.capturePhoto(with: settings, delegate: self)
        }
    }

    // AVCapturePhotoCaptureDelegate method
    func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?
    ) {
        if let error = error {
            print("Error capturing photo: \(error.localizedDescription)")
            return
        }

        // Convert the captured photo to UIImage
        if let photoData = photo.fileDataRepresentation(),
            let image = UIImage(data: photoData)
        {

            // Resize the image to 512x512
            let targetSize = CGSize(width: 512, height: 512)
            let resizedImage = resizeImage(image: image, targetSize: targetSize)

            // Convert UIImage to JPEG data
            if let jpegData = resizedImage?.jpegData(compressionQuality: 1.0)
            {  // Compression quality: 1.0 = maximum quality

                //save current resized image to send to training/prediction tasks
                currentResizedImage = UIImage(data: jpegData) ?? UIImage()  // if error, provide empty image

                DispatchQueue.main.async {
                    // self.capturedImageView.image = image
                    self.capturedImageView.image = resizedImage
                    self.capturedImageView.isHidden = false

                    // Stop the camera after capturing the photo
                    self.stopCamera()
                    self.restoreUI()  // Restore the initial UI state
                }
            } else {
                print("Error converting image to JPEG format")
            }
        }
    }

    func resizeImage(image: UIImage, targetSize: CGSize) -> UIImage? {
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }

    // MARK:
    // Allow the user to change the IP via text field.
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        if textField == newObjToDetect {
            createNewLabel()
            newObjToDetect.resignFirstResponder()  // Dismiss the keyboard
        }
        return true
    }

    // Process the input for menu items
    func createNewLabel() {
        guard let newItem = newObjToDetect.text, !newItem.isEmpty else {
            print("No item entered")
            return
        }
        print("current items: \(objDetectMenuItems)")
        // Prevent duplicate items
        guard !objDetectMenuItems.contains(newItem) else {
            print("Item already exists")
            newObjToDetect.text = ""  // Clear the text field
            newObjToDetect.resignFirstResponder()  // Dismiss the keyboard
            return
        }

        // Add the new item and update the menu
        objDetectMenuItems.append(newItem)
        print("New item added: \(newItem)")
        updateMenu(with: objDetectMenuItems)

        // Add the new item to the MLClient
        client.addLabel(newItem)

        // Clear the text field and dismiss the keyboard
        newObjToDetect.text = ""
        newObjToDetect.resignFirstResponder()
    }

    @IBAction func trainPredictSeg(_ sender: Any) {
        let selectedIndex = trainPredictSegControl.selectedSegmentIndex
        trainMode = selectedIndex == 0
        print("Train Mode: \(trainMode)")
    }

    @IBAction func startStopCameraOps(_ sender: Any) {
        if isCameraRunning {  // If Camera is active, stop camera and restore UI
            stopCamera()
            restoreUI()
        } else {
            startCamera()  //If Camera not active, start camera
        }
    }

    func startCamera() {
        // Initialize the capture session if not already initialized
        if captureSession == nil {
            captureSession = AVCaptureSession()
            captureSession.sessionPreset = .photo

            // Configure the camera device
            guard
                let camera = AVCaptureDevice.default(
                    .builtInWideAngleCamera, for: .video, position: .back)
            else {
                fatalError("No back camera available")
            }

            do {
                let input = try AVCaptureDeviceInput(device: camera)
                if captureSession.canAddInput(input) {
                    captureSession.addInput(input)
                }
            } catch {
                fatalError("Error setting up camera input: \(error)")
            }

            // Configure the photo output
            photoOutput = AVCapturePhotoOutput()
            if captureSession.canAddOutput(photoOutput) {
                captureSession.addOutput(photoOutput)
            }
        }

        // Recreate the preview layer if it was removed
        if previewLayer == nil {
            previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
            previewLayer.videoGravity = .resizeAspectFill
            //previewLayer.frame = view.bounds
            //view.layer.insertSublayer(previewLayer, at: 0)
            previewLayer.frame = cameraFeedView.bounds  //Match the size of the UIView
            cameraFeedView.layer.addSublayer(previewLayer)
        }

        // Start the capture session on a background thread
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
            DispatchQueue.main.async {
                // Update UI-related elements on the main thread
                self.isCameraRunning = true
                self.StartStopCamera.setTitle("Stop Camera", for: .normal)
                self.capturedImageView.isHidden = true  // Hide the image view when the camera starts
            }
        }
    }

    func stopCamera() {
        // Stop the capture session
        captureSession.stopRunning()
        isCameraRunning = false
        StartStopCamera.setTitle("Start Camera", for: .normal)

        // Remove the preview layer to restore the initial background/UI
        if previewLayer != nil {
            previewLayer.removeFromSuperlayer()
            previewLayer = nil
        }
    }

    func restoreUI() {
        // Reset the UIImageView and other UI elements to their initial state
        //capturedImageView.image = nil // Clear the captured image
        //capturedImageView.isHidden = true // Hide the image view
        view.backgroundColor = .white  // Reset to the initial background color
    }

    @IBAction func trainButtonClicked(_ sender: AnyObject) {

        feedbackLabel.text = "Training!"

        print("Train BTN: model sel = \(selectedModel)")
        sleep(2)

        if let dsid = client.getLabel(byName: currentObjectSelected)?.dsid {
            print("VC-trainModel selected: dsid = \(dsid)")
            client.trainModel(dsid: dsid, model_t: selectedModel)
        }

    }

    func uploadTrainingImage() {
        feedbackLabel.text = "Uploading!"

        if let dsid = client.getLabel(byName: currentObjectSelected)?.dsid {
            print("VC-uploadTrainingImage selected: dsid = \(dsid)")
            client.uploadImage(image: currentResizedImage, dsid: dsid)
            imageCount += 1
            imageCountLabel.text = "\(imageCount) / 5"

        }
    }

    func predict() {
        feedbackLabel.text = "Predicting!"

        print("predictModel BTN: model sel = \(selectedModel)")

        if let dsid = client.getLabel(byName: currentObjectSelected)?.dsid {
            print("VC-tpredictModel selected: dsid = \(dsid)")
            client.predict(
                image: currentResizedImage, dsid: dsid, model_t: selectedModel)
        }
    }

    @IBAction func objSelectionChange(_ sender: Any) {
        imageCountLabel.text = "0 / 5"
        imageCount = 0
    }
    
    
    @IBAction func uploadImageClicked(_ sender: Any) {

        if trainMode {
            uploadTrainingImage()
        } else {
            predict()
        }

    }

}

//MARK: MLClient Protocol Required Functions
extension ViewController: MLClientProtocol {
    // function to print the labels fetched
    func didFetchLabels(labels: [Dataset]) {
        print(labels)
    }
    // function to print the label added
    func labelAdded(label: Dataset?, error: APIError?) {
        if let error = error {
            print(error.localizedDescription)
        } else {
            print("Label added: \(label?.label ?? "")")
        }
    }
    // function to indicate whether the image was uploaded successfully
    func uploadImageComplete(success: Bool, errMsg: String?) {
        if success {
            print("Image uploaded successfully")
            feedbackLabel.text = "Image uploaded"
        } else {
            print("Image upload failed: \(errMsg ?? "")")
            feedbackLabel.text = "upload failed"
        }
    }
    
    // function to indicate model training complete
    func modelTrainingComplete(result: [String: Any]?, error: APIError?) {
        print("Model training complete: \(result)")
        if let result = result,
            let accuracy = result["accuracy"] as? Double
        {
            let accuracyFormatted = String(format: "%.2f", accuracy * 100)
            feedbackLabel.text =
                "accuracy = \(accuracyFormatted)%"
        } else {
            feedbackLabel.text = "No result"
        }
    }
    // function to indicate model prediction complete
    func predictionComplete(result: [String: Any]?, error: APIError?) {
        print("Prediction complete: \(result)")
        if let result = result,
            let prediction = result["prediction"] as? Int
        {
            feedbackLabel.text = "Prediction: \(prediction == 1)"
        } else {
            feedbackLabel.text = "No result"
        }

    }
}
