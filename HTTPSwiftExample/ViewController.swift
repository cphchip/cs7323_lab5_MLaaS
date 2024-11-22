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
    // Modifications: capture a picture taken with iphone camera using AVFoundation
    
    
    // MARK: Class Properties
    
    // interacting with server
    let client = MlaasModel() // how we will interact with the server
    
    // operation queues
    //let calibrationOperationQueue = OperationQueue()

    // TODO: relace in storyboard....@IBOutlet wek var captureButton: UIButton!
    
    
    // Photo capture properties
    var captureSession: AVCaptureSession!
    var photoOutput: AVCapturePhotoOutput!
    var previewLayer: AVCaptureVideoPreviewLayer!
    var capturedImageView: UIImageView!


    // state variables
    var isCalibrating = false
    //var isWaitingForMotionData = false

    // User Interface properties
    //let animation = CATransition()
    @IBOutlet weak var dsidLabel: UILabel!
    //@IBOutlet weak var upArrow: UILabel!
    //@IBOutlet weak var rightArrow: UILabel!
    //@IBOutlet weak var downArrow: UILabel!
    //@IBOutlet weak var leftArrow: UILabel!
    @IBOutlet weak var ipTextField: UITextField!
    //@IBOutlet weak var largeMotionMagnitude: UIProgressView!

    // MARK: Class Properties with Observers
    enum CalibrationStage:String {
        case notCalibrating = "notCalibrating"
       // case up = "up"
       // case right = "right"
       // case down = "down"
       // case left = "left"
    }

    var calibrationStage:CalibrationStage = .notCalibrating {
        didSet{
            //self.setInterfaceForCalibrationStage()
        }
    }

   // @IBAction func magnitudeChanged(_ sender: UISlider) {
   //     self.magThreshold = Double(sender.value)
   // }


    // MARK: View Controller Life Cycle
    override func viewDidLoad() {
        super.viewDidLoad()
        
        view.backgroundColor = .black
        
        // Set up the camera
        setupCamera()
        
        // Additional setup for UI elements
        setupCapturedImageView()

        // create reusable animation
        //animation.timingFunction = CAMediaTimingFunction(name: //CAMediaTimingFunctionName.easeInEaseOut)
        //animation.type = CATransitionType.fade
        //animation.duration = 0.5

        // setup core motion handlers
        //startMotionUpdates()

        // use delegation for interacting with client
        client.delegate = self
        client.updateDsid(5) // set default dsid to start with

        ipTextField.delegate = self
        ipTextField.text = client.server_ip

    }
    
    func setupCamera() {
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
        
        // Configure the preview layer
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = view.bounds
        view.layer.addSublayer(previewLayer)
        
        // Start the session
        captureSession.startRunning()
    }
    
    func setupCapturedImageView() {
        // Configure the image view to display captured images
        capturedImageView = UIImageView(frame: CGRect(x: 20, y: 20, width: view.frame.width - 40, height: 300))
        capturedImageView.contentMode = .scaleAspectFit
        capturedImageView.isHidden = true
        view.addSubview(capturedImageView)
    }
    
    //TODO: Add IBAction for the capture button
//        @IBAction func capturePhotoButtonTapped(_ sender: UIButton) {
//            capturePhoto()
//        }
  
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
        
        if let photoData = photo.fileDataRepresentation(),
           let image = UIImage(data: photoData) {
            DispatchQueue.main.async {
                self.capturedImageView.image = image
                self.capturedImageView.isHidden = false
            }
        }
    }
    
    
    // MARK:
    // Allow the user to change the IP via text field.
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        if let ipText = ipTextField.text, !ipText.isEmpty {
            client.setServerIp(ip: ipText)
            print("IP set to ", client.server_ip)
        } else {
            print("New IP is nil or empty")
        }

        ipTextField.resignFirstResponder()
        return true
    }

    //MARK: UI Buttons
    @IBAction func getDataSetId(_ sender: AnyObject) {
        client.getNewDsid() // protocol used to update dsid
    }

    @IBAction func startCalibration(_ sender: AnyObject) {
        //self.isWaitingForMotionData = false // dont do anything yet
        //nextCalibrationStage() // kick off the calibration stages

    }

    @IBAction func makeModel(_ sender: AnyObject) {
        client.trainModel()
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
    
//==TO BE DELETED====
//    // MARK: Class Properties
//    
//    // interacting with server
//    let client = MlaasModel() // how we will interact with the server
//    
//    // operation queues
//    let motionOperationQueue = OperationQueue()
//    let calibrationOperationQueue = OperationQueue()
//    
//    // motion data properties
//    var ringBuffer = RingBuffer()
//    let motion = CMMotionManager()
//    var magThreshold = 0.1
//
//
//    
//    // state variables
//    var isCalibrating = false
//    var isWaitingForMotionData = false
//    
//    // User Interface properties
//    let animation = CATransition()
//    @IBOutlet weak var dsidLabel: UILabel!
//    @IBOutlet weak var upArrow: UILabel!
//    @IBOutlet weak var rightArrow: UILabel!
//    @IBOutlet weak var downArrow: UILabel!
//    @IBOutlet weak var leftArrow: UILabel!
//    @IBOutlet weak var ipTextField: UITextField!
//    @IBOutlet weak var largeMotionMagnitude: UIProgressView!
//    
//    // MARK: Class Properties with Observers
//    enum CalibrationStage:String {
//        case notCalibrating = "notCalibrating"
//        case up = "up"
//        case right = "right"
//        case down = "down"
//        case left = "left"
//    }
//    
//    var calibrationStage:CalibrationStage = .notCalibrating {
//        didSet{
//            self.setInterfaceForCalibrationStage()
//        }
//    }
//        
//    @IBAction func magnitudeChanged(_ sender: UISlider) {
//        self.magThreshold = Double(sender.value)
//    }
//       
//    
//    // MARK: View Controller Life Cycle
//    override func viewDidLoad() {
//        super.viewDidLoad()
//
//        // create reusable animation
//        animation.timingFunction = CAMediaTimingFunction(name: CAMediaTimingFunctionName.easeInEaseOut)
//        animation.type = CATransitionType.fade
//        animation.duration = 0.5
//        
//        // setup core motion handlers
//        startMotionUpdates()
//        
//        // use delegation for interacting with client 
//        client.delegate = self
//        client.updateDsid(5) // set default dsid to start with
//        
//        ipTextField.delegate = self
//        ipTextField.text = client.server_ip
//
//    }
//    
//    // MARK: Flipped Assignment 4 question 8
//    // Allow the user to change the IP via text field.
//    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
//        if let ipText = ipTextField.text, !ipText.isEmpty {
//            client.setServerIp(ip: ipText)
//            print("IP set to ", client.server_ip)
//        } else {
//            print("New IP is nil or empty")
//        }
//        
//        ipTextField.resignFirstResponder()
//        return true
//    }
//    
//    //MARK: UI Buttons
//    @IBAction func getDataSetId(_ sender: AnyObject) {
//        client.getNewDsid() // protocol used to update dsid
//    }
//    
//    @IBAction func startCalibration(_ sender: AnyObject) {
//        self.isWaitingForMotionData = false // dont do anything yet
//        nextCalibrationStage() // kick off the calibration stages
//        
//    }
//    
//    @IBAction func makeModel(_ sender: AnyObject) {
//        client.trainModel()
//    }
//
//}
//
////MARK: Protocol Required Functions
//extension ViewController {
//    func updateDsid(_ newDsid:Int){
//        // delegate function completion handler
//        DispatchQueue.main.async{
//            // update label when set
//            self.dsidLabel.layer.add(self.animation, forKey: nil)
//            self.dsidLabel.text = "Current DSID: \(newDsid)"
//        }
//    }
//    
//    func receivedPrediction(_ prediction:[String:Any]){
//        if let labelResponse = prediction["prediction"] as? String{
//            print(labelResponse)
//            self.displayLabelResponse(labelResponse)
//        }
//        else{
//            print("Received prediction data without label.")
//        }
//    }
//}
//
//
////MARK: Motion Extension Functions
//extension ViewController {
//    // Core Motion Updates
//    func startMotionUpdates(){
//        // some internal inconsistency here: we need to ask the device manager for device
//        
//        if self.motion.isDeviceMotionAvailable{
//            self.motion.deviceMotionUpdateInterval = 1.0/200
//            self.motion.startDeviceMotionUpdates(to: motionOperationQueue, withHandler: self.handleMotion )
//        }
//    }
//    
//    func handleMotion(_ motionData:CMDeviceMotion?, error:Error?){
//        if let accel = motionData?.userAcceleration {
//            self.ringBuffer.addNewData(xData: accel.x, yData: accel.y, zData: accel.z)
//            let mag = fabs(accel.x)+fabs(accel.y)+fabs(accel.z)
//            
//            DispatchQueue.main.async{
//                //show magnitude via indicator
//                self.largeMotionMagnitude.progress = Float(mag)/0.2
//            }
//            
//            if mag > self.magThreshold {
//                // buffer up a bit more data and then notify of occurrence
//                DispatchQueue.main.asyncAfter(deadline: .now() + 0.05, execute: {
//                    self.calibrationOperationQueue.addOperation {
//                        // something large enough happened to warrant
//                        self.largeMotionEventOccurred()
//                    }
//                })
//            }
//        }
//    }
//    
//    // Calibration event has occurred, send to server
//    func largeMotionEventOccurred(){
//        if(self.isCalibrating){
//            //send a labeled example
//            if(self.calibrationStage != .notCalibrating && self.isWaitingForMotionData)
//            {
//                self.isWaitingForMotionData = false
//                
//                // send data to the server with label
//                self.client.sendData(self.ringBuffer.getDataAsVector(),
//                                     withLabel: self.calibrationStage.rawValue)
//                
//                self.nextCalibrationStage()
//            }
//        }
//        else
//        {
//            if(self.isWaitingForMotionData)
//            {
//                self.isWaitingForMotionData = false
//                //predict a label
//                self.client.sendData(self.ringBuffer.getDataAsVector())
//                // dont predict again for a bit
//                setDelayedWaitingToTrue(2.0)
//                
//            }
//        }
//    }
//}
//
////MARK: Calibration UI Functions
//extension ViewController {
//    
//    func setDelayedWaitingToTrue(_ time:Double){
//        DispatchQueue.main.asyncAfter(deadline: .now() + time, execute: {
//            self.isWaitingForMotionData = true
//        })
//    }
//    
//    func setAsCalibrating(_ label: UILabel){
//        label.layer.add(animation, forKey:nil)
//        label.backgroundColor = UIColor.red
//    }
//    
//    func setAsNormal(_ label: UILabel){
//        label.layer.add(animation, forKey:nil)
//        label.backgroundColor = UIColor.white
//    }
//    
//    // blink the UILabel
//    func blinkLabel(_ label:UILabel){
//        DispatchQueue.main.async {
//            self.setAsCalibrating(label)
//            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0, execute: {
//                self.setAsNormal(label)
//            })
//        }
//    }
//    
//    func displayLabelResponse(_ response:String){
//        switch response {
//        case "['up']","up":
//            blinkLabel(upArrow)
//            break
//        case "['down']","down":
//            blinkLabel(downArrow)
//            break
//        case "['left']","left":
//            blinkLabel(leftArrow)
//            break
//        case "['right']","right":
//            blinkLabel(rightArrow)
//            break
//        default:
//            print("Unknown")
//            break
//        }
//    }
//    
//    func setInterfaceForCalibrationStage(){
//        switch calibrationStage {
//        case .up:
//            self.isCalibrating = true
//            DispatchQueue.main.async{
//                self.setAsCalibrating(self.upArrow)
//                self.setAsNormal(self.rightArrow)
//                self.setAsNormal(self.leftArrow)
//                self.setAsNormal(self.downArrow)
//            }
//            break
//        case .left:
//            self.isCalibrating = true
//            DispatchQueue.main.async{
//                self.setAsNormal(self.upArrow)
//                self.setAsNormal(self.rightArrow)
//                self.setAsCalibrating(self.leftArrow)
//                self.setAsNormal(self.downArrow)
//            }
//            break
//        case .down:
//            self.isCalibrating = true
//            DispatchQueue.main.async{
//                self.setAsNormal(self.upArrow)
//                self.setAsNormal(self.rightArrow)
//                self.setAsNormal(self.leftArrow)
//                self.setAsCalibrating(self.downArrow)
//            }
//            break
//            
//        case .right:
//            self.isCalibrating = true
//            DispatchQueue.main.async{
//                self.setAsNormal(self.upArrow)
//                self.setAsCalibrating(self.rightArrow)
//                self.setAsNormal(self.leftArrow)
//                self.setAsNormal(self.downArrow)
//            }
//            break
//        case .notCalibrating:
//            self.isCalibrating = false
//            DispatchQueue.main.async{
//                self.setAsNormal(self.upArrow)
//                self.setAsNormal(self.rightArrow)
//                self.setAsNormal(self.leftArrow)
//                self.setAsNormal(self.downArrow)
//            }
//            break
//        }
//    }
//    
//    func nextCalibrationStage(){
//        switch self.calibrationStage {
//        case .notCalibrating:
//            //start with up arrow
//            self.calibrationStage = .up
//            setDelayedWaitingToTrue(1.0)
//            break
//        case .up:
//            //go to right arrow
//            self.calibrationStage = .right
//            setDelayedWaitingToTrue(1.0)
//            break
//        case .right:
//            //go to down arrow
//            self.calibrationStage = .down
//            setDelayedWaitingToTrue(1.0)
//            break
//        case .down:
//            //go to left arrow
//            self.calibrationStage = .left
//            setDelayedWaitingToTrue(1.0)
//            break
//            
//        case .left:
//            //end calibration
//            self.calibrationStage = .notCalibrating
//            setDelayedWaitingToTrue(1.0)
//            break
//        }
//    }


