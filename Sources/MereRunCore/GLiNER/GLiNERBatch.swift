import Foundation

enum GLiNERClassificationMerge {
    static func head(task: GLiNERClassificationTask, chunks: [GLiNERClassificationHead]) -> GLiNERClassificationHead {
        var probabilities: [String: Double] = [:]
        for label in task.labels {
            let values = chunks.compactMap { $0.probabilities[label] }
            probabilities[label] = task.multiLabel ? (values.max() ?? 0) :
                values.reduce(0, +) / Double(max(1, values.count))
        }
        let best: String
        if task.multiLabel {
            best = task.labels.max { (probabilities[$0] ?? 0) < (probabilities[$1] ?? 0) } ?? task.labels[0]
        } else {
            let votes = task.labels.map { label in chunks.filter { $0.labels.first == label }.count }
            let index = votes.firstIndex(of: votes.max() ?? 0) ?? 0
            best = task.labels[index]
        }
        let labels = task.multiLabel
            ? task.labels.filter { (probabilities[$0] ?? 0) >= task.threshold }
            : [best]
        return GLiNERClassificationHead(labels: labels.isEmpty ? [best] : labels, probabilities: probabilities)
    }
}

extension GLiNERClassificationOperation {
    public func prepareLong(_ request: GLiNERClassificationRequest, chunkSize: Int = 384,
                            chunkOverlap: Int = 64) throws -> [GLiNERClassificationPlan] {
        try request.validate()
        let chunks = try GLiNERTextChunks.make(text: request.text, size: chunkSize, overlap: chunkOverlap) { text in
            (try? self.prepare(GLiNERClassificationRequest(text: text, tasks: request.tasks))) != nil
        }
        return try chunks.map { try prepare(GLiNERClassificationRequest(text: $0.text, tasks: request.tasks)) }
    }

    public func predictBatch(_ requests: [GLiNERClassificationRequest]) throws -> [GLiNERClassificationResponse] {
        try requests.map { try predict($0) }
    }

    public func predictLong(_ request: GLiNERClassificationRequest, chunkSize: Int = 384,
                            chunkOverlap: Int = 64) throws -> GLiNERClassificationResponse {
        try request.validate()
        let chunks = try GLiNERTextChunks.make(text: request.text, size: chunkSize, overlap: chunkOverlap) { text in
            (try? self.prepare(GLiNERClassificationRequest(text: text, tasks: request.tasks))) != nil
        }
        let results = try chunks.map { chunk in
            try predict(GLiNERClassificationRequest(text: chunk.text, tasks: request.tasks))
        }
        var heads: [String: GLiNERClassificationHead] = [:]
        for task in request.tasks {
            let chunkHeads = results.compactMap { $0.heads[task.name] }
            heads[task.name] = GLiNERClassificationMerge.head(task: task, chunks: chunkHeads)
        }
        return GLiNERClassificationResponse(model: modelID, runtime: "native-swift-mlx-fp32",
                                            heads: heads, inputTokens: results.reduce(0) { $0 + $1.inputTokens })
    }

    public func predictBatchLong(_ requests: [GLiNERClassificationRequest], chunkSize: Int = 384,
                                 chunkOverlap: Int = 64) throws -> [GLiNERClassificationResponse] {
        try requests.map { try predictLong($0, chunkSize: chunkSize, chunkOverlap: chunkOverlap) }
    }
}
