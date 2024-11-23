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



import UIKit
import AVFoundation

class ViewController: UIViewController,AVCapturePhotoCaptureDelegate, ClientDelegate, UITextFieldDelegate {
    
    //Ref Cite:  ChatGPT
    // This section of code was generated with the assistance of ChatGPT, an AI language model by OpenAI.
    // Date: 11/22/24
    // Source: OpenAI's ChatGPT (https://openai.com/chatgpt)
    // Prompt: capture a picture taken with phone camera using AVFoundation
    // Modifications: updated to integrate with Mlaas Model
    
    
    // MARK: Class Properties
    
    // interacting with server
    //let client = MlaasModel() // how we will interact with the server
    
    // operation queues
    //let calibrationOperationQueue = OperationQueue()

    // TODO: relace in storyboard....@IBOutlet wek var captureButton: UIButton!
    
    
    // Photo capture properties
    var captureSession: AVCaptureSession!
    var photoOutput: AVCapturePhotoOutput!
    var previewLayer: AVCaptureVideoPreviewLayer!
    var isCameraRunning = false // To track the camera state


    // state variables
    var isCalibrating = false

    // User Interface properties
    @IBOutlet weak var dsidLabel: UILabel!
    @IBOutlet weak var ipTextField: UITextField!
    @IBOutlet weak var capturedImageView: UIImageView!
    @IBOutlet weak var Calibrate: UIButton!
    
    
    // MARK: Class Properties with Observers
    enum CalibrationStage:String {
        case notCalibrating = "notCalibrating"
    }

    var calibrationStage:CalibrationStage = .notCalibrating {
        didSet{
            //self.setInterfaceForCalibrationStage()
        }
    }


    // MARK: View Controller Life Cycle
    override func viewDidLoad() {
        super.viewDidLoad()
        
        view.backgroundColor = .white
        
        // Initial state for the UIImageView
        capturedImageView.contentMode = .scaleAspectFit
        capturedImageView.isHidden = true // Hide initially until a photo is cap
        
        // Set the button's initial title
         Calibrate.setTitle("Start Camera", for: .normal)

        // use delegation for interacting with client
        //client.delegate = self
        //client.updateDsid(5) // set default dsid to start with

        //ipTextField.delegate = self
        //ipTextField.text = client.server_ip

    }
    
    // Photo capture button pressed. Capture photo
    @IBAction func capturePhotoButtonTapped(_ sender: UIButton) {
        if isCameraRunning {
            capturePhoto()
        }
    }
    
    // Capture a photo
    func capturePhoto() {
        let settings = AVCapturePhotoSettings()
        photoOutput.capturePhoto(with: settings, delegate: self)
    }

    // AVCapturePhotoCaptureDelegate method
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let error = error {
            print("Error capturing photo: \(error.localizedDescription)")
            return
        }
        
        // Convert the captured photo to UIImage
        if let photoData = photo.fileDataRepresentation(),
           let image = UIImage(data: photoData) {
            
            // Convert UIImage to JPEG data
            if let jpegData = image.jpegData(compressionQuality: 1.0) { // Compression quality: 1.0 = maximum quality
                // Save JPEG data to disk or use it as needed
                saveJPEGToDisk(data: jpegData) // Optional function to save
                DispatchQueue.main.async {
                    self.capturedImageView.image = image
                    self.capturedImageView.isHidden = false
                    
                    // Stop the camera after capturing the photo
                    self.stopCamera()
                    self.restoreUI() // Restore the initial UI state
                }
            } else {
                print("Error converting image to JPEG format")
            }
        }
    }
    
    func saveJPEGToDisk(data: Data) {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let filePath = documentsPath.appendingPathComponent("captured_photo.jpg")
        
        do {
            try data.write(to: filePath)
            print("JPEG saved to: \(filePath)")
        } catch {
            print("Error saving JPEG to disk: \(error.localizedDescription)")
        }
    }

    
    // MARK:
    // Allow the user to change the IP via text field.
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        if let ipText = ipTextField.text, !ipText.isEmpty {
            //client.setServerIp(ip: ipText)
            //print("IP set to ", client.server_ip)
        } else {
            print("New IP is nil or empty")
        }

        ipTextField.resignFirstResponder()
        return true
    }

    //MARK: UI Buttons
    @IBAction func getDataSetId(_ sender: AnyObject) {
        //client.getNewDsid() // protocol used to update dsid
    }

    @IBAction func startCalibration(_ sender: AnyObject) {
        //nextCalibrationStage() // kick off the calibration stages
        if isCameraRunning {   // If Camera is active, stop camera and restore UI
            stopCamera()
            restoreUI()
        } else {
            startCamera()      //If Camera not active, start camera
        }
    }
    
    func startCamera() {
        // Initialize the capture session if not already initialized
        if captureSession == nil {
            captureSession = AVCaptureSession()
            captureSession.sessionPreset = .photo
            
            // Configure the camera device
            guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
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
            previewLayer.frame = view.bounds
            view.layer.insertSublayer(previewLayer, at: 0)
        }
        
        // Start the capture session on a background thread
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
            DispatchQueue.main.async {
                // Update UI-related elements on the main thread
                self.isCameraRunning = true
                self.Calibrate.setTitle("Stop Camera", for: .normal)
                self.capturedImageView.isHidden = true // Hide the image view when the camera starts
            }
        }
    }

    func stopCamera() {
        // Stop the capture session
        captureSession.stopRunning()
        isCameraRunning = false
        Calibrate.setTitle("Start Camera", for: .normal)
        
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
        view.backgroundColor = .white // Reset to the initial background color
    }

    @IBAction func makeModel(_ sender: AnyObject) {
        //client.trainModel()
    }
    
}

    
//MARK: Protocol Required Functions
extension ViewController {
    func updateDsid(_ newDsid:Int){
        // delegate function completion handler
        DispatchQueue.main.async{
            // update label when set
           // self.dsidLabel.layer.add(self.animation, forKey: nil)
            self.dsidLabel.text = "Current DSID: \(newDsid)"
        }
    }

    func receivedPrediction(_ prediction:[String:Any]){
        if let labelResponse = prediction["prediction"] as? String{
            print(labelResponse)
            //self.displayLabelResponse(labelResponse)
        }
        else{
            print("Received prediction data without label.")
        }
    }
}



