//
//  Dataset.swift
//
//  Created by Ches Smith on 11/24/24.
//

import Foundation

struct Dataset: Codable, Equatable {
    var dsid: Int
    var label: String
}

enum APIError: Error, LocalizedError {
    case labelAlreadyExists
    case invalidURL
    case serverError(Int, String?)
    case decodingError(String)
    case noData
    case networkError(Error)
    case datasetDoesNotExist(Int)
    case maximumRetriesExceeded
    case taskFailed(String)
    
    var errorDescription: String? {
        switch self {
        case .labelAlreadyExists:
            return "A label with this name already exists."
        case .invalidURL:
            return "The URL is invalid."
        case .serverError(let statusCode, let detail):
            return "Server responded with an error. Status code: \(statusCode), detail: \(detail ?? "")"
        case .decodingError(let message):
            return "Failed to decode the server response: \(message)"
        case .noData:
            return "No data received from the server."
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .datasetDoesNotExist(let dsid):
            return "The dataset with dsid=\(dsid) does not exist."
        case .maximumRetriesExceeded:
            return "Maximum number of retries exceeded"
        case .taskFailed(let message):
            return "Task failed: \(message)"
        }
    
    }
}

struct TrainRequest: Codable {
    var dsid: Int
    var model_t: String
}

struct TaskStatus: Codable {
    var task_id: String
    var status: String
    var result: [String: String]? = nil
}
