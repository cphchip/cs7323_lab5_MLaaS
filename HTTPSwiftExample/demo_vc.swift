//
//  ViewController.swift
//  model_5
//
//  Created by Ches Smith on 11/24/24.
//

import UIKit

class ViewControllerDemo: UIViewController {

    let client = MLClient()

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        sleep(2)
        client.delegate = self
        client.addLabel("dog")
        client.addLabel("frog")
        client.addLabel("log")
        let labels = client.getLabels()
        print(labels)

        sleep(2)
        if let dsid = client.getLabel(byName: "frog")?.dsid {
            for i in Range(1...10) {
                print("finding image \(i)")
                if let image = UIImage(named: "frog\(i)") {
                    print("Uploading image \(i)")
                    self.client.uploadImage(image: image, dsid: dsid)
                    print("Image uploaded")
                }

            }
            sleep(10)
            print("Training model")
            client.trainModel(dsid: dsid, model_t: "svc")
            sleep(20)
            if let image = UIImage(named: "frog1") {
                print("Predicting frog")
                client.predict(image: image, dsid: dsid, model_t: "svc")
            }
            if let image = UIImage(named: "dog") {
                print("Predicting dog")
                client.predict(image: image, dsid: dsid, model_t: "svc")
            }
        } else {
            print("Label not found")
        }

    }

}

extension ViewControllerDemo: MLClientProtocol {

    func didFetchLabels(labels: [Dataset]) {
        print(labels)
    }

    func labelAdded(label: Dataset?, error: APIError?) {
        if let error = error {
            print(error.localizedDescription)
        } else {
            print("Label added: \(label?.label ?? "")")
        }
    }

    func uploadImageComplete(success: Bool, errMsg: String?) {
        if success {
            print("Image uploaded successfully")
        } else {
            print("Image upload failed: \(errMsg ?? "")")
        }
    }

    func modelTrainingComplete(result: [String: Any]?, error: APIError?) {
        print("Model training complete: \(result)")
    }

    func predictionComplete(result: [String: Any]?, error: APIError?) {
        print("Prediction complete: \(result)")
    }
}
