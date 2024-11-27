//
//  MLClient.swift
//  HTTPSwiftExample
//
//  Created by Ches Smith on 11/24/24.
//

import Foundation
import UIKit

protocol MLClientProtocol {
    func didFetchLabels(labels: [Dataset])
    func labelAdded(label: Dataset?, error: APIError?)
    func uploadImageComplete(success: Bool, errMsg: String?)
    func modelTrainingComplete(result: [String: Any]?, error: APIError?)
    func predictionComplete(result: [String: Any]?, error: APIError?)
}

class MLClient {
    // API endpoint
    private let API_BASE_ENDPOINT = "http://45.33.24.52:8000"

    // secret API token
    private let API_TOKEN =
        Bundle.main.object(forInfoDictionaryKey: "API_TOKEN") as? String ?? ""
    
    public var delegate: MLClientProtocol?
    
    // are we currently fetching labels?
    private var fetchingLabels = false
    
    // local copy of labels
    private var labels: [Dataset] = []

    // get all labels
    func getLabels() -> [Dataset] {
        // Wait for the labels to be fetched if they are currently being fetched
        if fetchingLabels {
            var count = 0
            while fetchingLabels && count < 10 {
                count += 1
                sleep(1)
            }
        }
        return labels
    }

    // Add a new label
    func addLabel(_ name: String) {
        // Check if the label name already exists locally
        guard !labels.contains(where: { $0.label == name }) else {
            delegate?.labelAdded(label: nil, error: .labelAlreadyExists)
            return
        }

        // Create the label on the server
        createLabel(name) { [weak self] result in
            switch result {
            case .success(let newLabel):
                // Update local labels and notify delegate
                self?.labels.append(newLabel)
                self?.delegate?.labelAdded(label: newLabel, error: nil)

            case .failure(let error):
                // Notify delegate of the error
                self?.delegate?.labelAdded(label: nil, error: error)
            }
        }
    }

    // Create a new label (POST to the server)
    private func createLabel(
        _ name: String,
        completion: @escaping (Result<Dataset, APIError>) -> Void
    ) {
        guard let url = URL(string: "\(API_BASE_ENDPOINT)/labels/\(name)")
        else {
            completion(.failure(APIError.invalidURL))
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue(API_TOKEN, forHTTPHeaderField: "x-api-token")
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        let task = URLSession.shared.dataTask(with: request) {
            data, response, error in
            // Handle network errors
            if let error = error {
                completion(.failure(.networkError(error)))
                return
            }

            // Ensure the server responded with a success status code
            guard let httpResponse = response as? HTTPURLResponse,
                httpResponse.statusCode == 201
            else {
                var detail = ""
                if let data = data {
                    let json = self.convertDataToDictionary(with: data)
                    detail =
                        json["detail"] as? String ?? "No detail from server"
                }
                let statusCode =
                    (response as? HTTPURLResponse)?.statusCode ?? -1
                completion(
                    .failure(APIError.serverError(statusCode, detail))
                )
                return
            }

            // Decode the response into a Label
            if let data = data {
                do {
                    let label = try JSONDecoder().decode(
                        Dataset.self, from: data)
                    completion(.success(label))
                } catch {
                    completion(
                        .failure(
                            APIError.decodingError(error.localizedDescription)
                        ))
                }
            } else {
                completion(.failure(APIError.noData))
            }
        }

        task.resume()
    }

    // Get a label by ID
    func getLabel(byID id: Int) -> Dataset? {
        return labels.first { $0.dsid == id }
    }

    // Get a label by name
    func getLabel(byName name: String) -> Dataset? {
        return labels.first { $0.label == name }
    }

    // Fetch all labels from the server
    func fetchLabels() async throws -> [Dataset] {
        if fetchingLabels {
            // If a labels request is already in progress, return an error
            throw APIError.labelsRequestInProgress
        }
        fetchingLabels = true
        defer { fetchingLabels = false } // reset the flag when done

        guard let url = URL(string: "\(API_BASE_ENDPOINT)/labels") else {
            throw APIError.invalidURL
        }
    
        // Create the request
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.addValue(API_TOKEN, forHTTPHeaderField: "x-api-token")
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Send the request
        do {
            let (data, response) = try await URLSession.shared.data(
                for: request)
            if let httpResponse = response as? HTTPURLResponse,
                httpResponse.statusCode == 200
            {
                // successful response
                print("Response: \(String(data: data, encoding: .utf8) ?? "")")
                let labels = try JSONDecoder().decode(
                    [Dataset].self, from: data)
                return labels
            } else {
                throw APIError.serverError(
                    (response as? HTTPURLResponse)?.statusCode ?? -1, nil)
            }
        } catch {
            throw APIError.networkError(error)
        }
    }

    init() {
        // Fetch labels on initialization
        Task {
            do {
                self.labels = try await fetchLabels()
            } catch {
                print("Error fetching labels: \(error.localizedDescription)")
            }
            DispatchQueue.main.async {
                print("Labels: \(self.labels)")
            }
        }
    }

    
    func uploadImage(image: UIImage, dsid: Int) {
        /// Upload an image to the server
        Task {
            // start the upload
            await callUploadImage(image: image, dsid: dsid) { result in
                // handle the result
                switch result {
                case .success(let message):
                    print("Image upload complete: \(message)")
                    // notify the delegate
                    DispatchQueue.main.async {
                        self.delegate?.uploadImageComplete(
                            success: true, errMsg: nil
                        )
                    }
                case .failure(let error):
                    print(
                        "Error uploading image: \(error.localizedDescription)")
                    // notify the delegate
                    DispatchQueue.main.async {
                        self.delegate?.uploadImageComplete(
                            success: false, errMsg: error.localizedDescription
                        )
                    }
                }
            }
        }
    }

    private func callUploadImage(
        image: UIImage, dsid: Int,
        completion: @escaping (Result<String, APIError>) -> Void
    ) async {
        // Validate the server URL
        guard let serverURL = URL(string: "\(API_BASE_ENDPOINT)/upload_image")
        else {
            completion(.failure(.invalidURL))
            return
        }
        
        // Compress the image
        guard let imageData = image.jpegData(compressionQuality: 0.8) else {
            completion(.failure(.decodingError("Unable to compress image")))
            return
        }

        // Prepare the multipart request
        var multipart = MultipartRequest()
        multipart.add(key: "dsid", value: String(dsid))  // Add ID
        multipart.add(
            key: "image",
            fileName: "image.jpg",
            fileMimeType: "image/jpeg",
            fileData: imageData
        )

        // Create the HTTP request
        var request = URLRequest(url: serverURL)
        request.httpMethod = "POST"
        request.setValue(
            multipart.httpContentTypeHeadeValue,
            forHTTPHeaderField: "Content-Type"
        )
        request.httpBody = multipart.httpBody

        // add api token to the request
        request.addValue(API_TOKEN, forHTTPHeaderField: "x-api-token")

        // Send the request
        do {
            let (data, response) = try await URLSession.shared.data(
                for: request)
            if let httpResponse = response as? HTTPURLResponse,
                (200...299).contains(httpResponse.statusCode)
            {
                // handle a successful response
                let json = convertDataToDictionary(with: data)
                let message = json["message"] as? String ?? "Upload successful"
                completion(.success(message))
                return
            }
            // bad response
            completion(.failure(.noData))
            return
        } catch {
            // really bad
            completion(.failure(.networkError(error)))
            return
        }
    }

    func predict(image: UIImage, dsid: Int, model_t: String) {
        /// request a prediction from the server
        Task {
            // make a request to the server
            await callPredict(image: image, dsid: dsid, model_t: model_t) {
                result in
                // handle the result
                switch result {
                case .success(let tid):
                    print("predicting, task id: \(tid)")
                    // after 3 seconds, start checking the task status
                    DispatchQueue.global().asyncAfter(deadline: .now() + 3) {
                        self.waitFor(taskId: tid) { result in
                            switch result {
                                // check the task result
                            case .success(let value):
                                print("prediction complete: \(value)")
                                // notify the delegate
                                DispatchQueue.main.async {
                                    self.delegate?.predictionComplete(
                                        result: value, error: nil)
                                }
                            case .failure(let error):
                                // task failed
                                print(
                                    "Error Making prediction: \(error.localizedDescription)"
                                )
                                DispatchQueue.main.async {
                                    self.delegate?.predictionComplete(
                                        result: nil, error: error)
                                }
                            }
                        }
                    }
                case .failure(let error):
                    // sending the request failed
                    print(
                        "Error requesting prediction: \(error.localizedDescription)"
                    )
                }
            }
        }
    }
    
    // make a prediction request to the server
    private func callPredict(
        image: UIImage, dsid: Int, model_t: String,
        completion: @escaping (Result<String, APIError>) -> Void
    ) async {
        guard let serverURL = URL(string: "\(API_BASE_ENDPOINT)/predict")
        else {
            // invalid URL
            completion(.failure(.invalidURL))
            return
        }

        // compress the image
        guard let imageData = image.jpegData(compressionQuality: 0.8) else {
            completion(.failure(.decodingError("Unable to compress image")))
            return
        }

        // Prepare the multipart request
        var multipart = MultipartRequest()
        multipart.add(key: "dsid", value: String(dsid))
        multipart.add(key: "model_t", value: model_t)
        multipart.add(
            key: "image",
            fileName: "image.jpg",
            fileMimeType: "image/jpeg",
            fileData: imageData
        )

        // Create the HTTP request
        var request = URLRequest(url: serverURL)
        request.httpMethod = "POST"
        request.addValue(API_TOKEN, forHTTPHeaderField: "x-api-token")
        request.setValue(
            multipart.httpContentTypeHeadeValue,
            forHTTPHeaderField: "Content-Type")
        request.httpBody = multipart.httpBody

        // Send the request
        do {
            let (data, response) = try await URLSession.shared.data(
                for: request)
            if let httpResponse = response as? HTTPURLResponse {
                let data = data
                let json = convertDataToDictionary(with: data)
                let tid = json["task_id"] as? String ?? ""
                completion(.success(tid))
                return
            }
            completion(.failure(.noData))
            return
        } catch {
            completion(.failure(.networkError(error)))
            return
        }
    }

    // wait for the selected task to complete then execute the completion
    private func waitFor(
        taskId tid: String, retries: Int = 20,
        completion: @escaping (Result<[String: Any], APIError>) -> Void
    ) {
        print("Waiting for task \(tid) to complete")
        
        // only wait for a limited number of retries
        if retries <= 0 {
            completion(.failure(.maximumRetriesExceeded))
            return
        }

        // validate the URL
        guard let url = URL(string: "\(API_BASE_ENDPOINT)/tasks/\(tid)") else {
            completion(.failure(.invalidURL))
            return
        }
        
        // create the request
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.addValue(API_TOKEN, forHTTPHeaderField: "x-api-token")
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // send the request
        let task = URLSession.shared.dataTask(with: request) {
            data, response, error in
            if let error = error {
                // something went wrong sending the request
                completion(.failure(.networkError(error)))
                return
            }
            
            // make sure the server responded with a success status code
            guard let httpResponse = response as? HTTPURLResponse,
                httpResponse.statusCode == 200
            else {
                // bad server response, try to get the details
                var detail = ""
                if let data = data {
                    let json = self.convertDataToDictionary(with: data)
                    print("Error**: \(json)")
                    detail =
                        json["detail"] as? String ?? "No detail from server"
                }
                let statusCode =
                    (response as? HTTPURLResponse)?.statusCode ?? -1
                completion(
                    .failure(APIError.serverError(statusCode, detail))
                )
                return
            }
            // success, parse the response
            if let data = data {
                do {
                    // decode the JSON
                    let json =
                        try JSONSerialization.jsonObject(
                            with: data,
                            options: []) as? [String: Any]
                    if let json = json {
                        let status = json["status"] as? String ?? ""
                        if status == "Success" {
                            // task completed successfully
                            print("Task completed: \(json)")
                            let result =
                                json["result"] as? [String: Any] ?? [:]
                            completion(.success(result))
                            return
                        } else if status == "Failed" {
                            // task failed
                            completion(
                                .failure(
                                    .taskFailed(
                                        json["result"] as? String ?? "\(json)"))
                            )
                            return
                        } else {
                            // Task is still pending, wait and try again after 3 seconds
                            DispatchQueue.global().asyncAfter(
                                deadline: .now() + 3
                            ) {
                                // recursive call to wait for the task to complete
                                // with one less retry
                                self.waitFor(
                                    taskId: tid, retries: retries - 1,
                                    completion: completion)
                            }
                            return
                        }
                    } else {
                        // failed to parse the JSON
                        print("Error: Unable to parse JSON")
                        print(
                            "Response: \(String(data: data, encoding: .utf8) ?? "No response data")"
                        )
                        completion(
                            .failure(.decodingError("Unable to parse JSON")))
                        return
                    }
                } catch {
                    // something went wrong
                    print("Error: \(error.localizedDescription)")
                    completion(
                        .failure(.decodingError(error.localizedDescription)))
                    return
                }
            }
        }
        task.resume()
    }

    
    func trainModel(dsid: Int, model_t: String) {
    /// Train a model on the server
        Task {
            // start the training
            callTrainModel(dsid: dsid, model_t: model_t) { result in
                // handle the response
                switch result {
                case .success(let tid):
                    print("Model training, task id: \(tid)")
                    // the training request was accepted, wait 3 seconds then start checking the task status
                    DispatchQueue.global().asyncAfter(deadline: .now() + 3) {
                        self.waitFor(taskId: tid) { result in
                            switch result {
                            case .success(let value):
                                // task completed successfully
                                print("Model training complete: \(value)")
                                DispatchQueue.main.async {
                                    self.delegate?.modelTrainingComplete(
                                        result: value, error: nil)
                                }
                            case .failure(let error):
                                // task failed
                                print(
                                    "Error training model: \(error.localizedDescription)"
                                )
                                DispatchQueue.main.async {
                                    self.delegate?.modelTrainingComplete(
                                        result: nil, error: error)
                                }
                            }
                        }
                    }
                case .failure(let error):
                    print(
                        "Error submitting training request: \(error.localizedDescription)"
                    )
                }
            }
        }
    }

    private func callTrainModel(
        dsid: Int, model_t: String,
        completion: @escaping (Result<String, APIError>) -> Void
    ) {
        // Validate the server URL
        guard let url = URL(string: "\(API_BASE_ENDPOINT)/train_model") else {
            print("Invalid URL")
            return
        }
        
        // train request to be added to the request body
        let trainReq = TrainRequest(dsid: dsid, model_t: model_t)

        // Create the request
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.addValue(API_TOKEN, forHTTPHeaderField: "x-api-token")
        do {
            request.httpBody = try JSONEncoder().encode(trainReq)
        } catch {
            print("Error: \(error.localizedDescription)")  // TODO: Add localized error
            return
        }

        // Send the request
        let task = URLSession.shared.dataTask(with: request) {
            data, response, error in
            if let error = error {
                // request not sent
                completion(.failure(.networkError(error)))
                return
            }

            // check the server response
            guard let httpResponse = response as? HTTPURLResponse,
                httpResponse.statusCode == 202
            else {
                // not a successful response
                var detail = ""
                if let data = data {
                    let json = self.convertDataToDictionary(with: data)
                    detail =
                        json["detail"] as? String ?? "No detail from server"
                }
                let statusCode =
                    (response as? HTTPURLResponse)?.statusCode ?? -1
                completion(
                    .failure(APIError.serverError(statusCode, detail))
                )
                return
            }
            
            // http response is good, check the value of the response
            if let data = data {
                do {
                    // parse the JSON response
                    let json =
                        try JSONSerialization.jsonObject(
                            with: data,
                            options: []) as? [String: Any]
                    if let json = json {
                        let tid = json["task_id"] as? String ?? ""
                        completion(.success(tid))
                        return
                    } else {
                        print("Error: Unable to parse JSON")
                        print(
                            "Response: \(String(data: data, encoding: .utf8) ?? "No response data")"
                        )
                        completion(
                            .failure(.decodingError("Unable to parse JSON")))
                        return
                    }
                } catch {
                    print("Error: \(error.localizedDescription)")
                    completion(
                        .failure(.decodingError(error.localizedDescription)))
                    return
                }
            }
        }

        task.resume()
    }

    private func convertDataToDictionary(with data: Data?) -> [String: Any] {
        // convenience function for getting Dictionary from server data
        do {  // try to parse JSON and deal with errors using do/catch block
            let jsonDictionary: [String: Any] =
                try JSONSerialization.jsonObject(
                    with: data!,
                    options: JSONSerialization.ReadingOptions.mutableContainers)
                as! [String: Any]

            return jsonDictionary

        } catch {
            print("json error: \(error.localizedDescription)")
            if let strData = String(
                data: data!,
                encoding: String.Encoding(
                    rawValue: String.Encoding.utf8.rawValue))
            {
                print("printing JSON received as string: " + strData)
            }
            return [String: Any]()  // just return empty
        }
    }
}
