/**
 * ARBridge.swift
 *
 * Thin web ↔ native bridge. Receives messages from the web app and presents
 * ARScanViewController. When scanning is done, serialises the point cloud and
 * calls window.onStageARResult() on the web page.
 *
 * Web → Native:
 *   window.webkit.messageHandlers.stageAR.postMessage({
 *     action: 'startScan',
 *     apiBase: 'https://api.example.com',   // optional — enables direct iOS→backend upload
 *     roomId: 'abc123',
 *     authToken: 'Bearer …',
 *     deviceToken: '…',
 *   })
 *
 * Native → Web:
 *   window.onStageARResult({ status: 'scanning' })
 *   window.onStageARResult({ status: 'projecting' })
 *   window.onStageARResult({ status: 'chunk', count: N, data: '<base64>' })
 *   window.onStageARResult({ status: 'done', pointCount: N, capturedAt: ms })
 *   window.onStageARResult({ error: '…' })
 */

import WebKit
import ARKit

final class ARBridge: NSObject {

    static let shared = ARBridge()
    weak var webView: WKWebView?

    /// Set by the JS startScan message — enables direct iOS→backend snapshot upload.
    var apiBase:     String? = nil
    var roomId:      String? = nil
    var authToken:   String? = nil
    var deviceToken: String? = nil

    override private init() {}

    // MARK: - Send helpers

    private func send(_ js: String) {
        DispatchQueue.main.async { [weak self] in
            self?.webView?.evaluateJavaScript(
                "window.onStageARResult && window.onStageARResult(\(js))", completionHandler: nil)
        }
    }

    private func sendError(_ msg: String) {
        let safe = msg
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "'",  with: "\\'")
        send("{ error: '\(safe)' }")
    }

    // MARK: - Scan

    private func startScan(body: [String: Any]) {
        // Capture optional backend credentials so iOS can upload snapshots directly.
        apiBase     = body["apiBase"]     as? String
        roomId      = body["roomId"]      as? String
        authToken   = body["authToken"]   as? String
        deviceToken = body["deviceToken"] as? String

        guard ARWorldTrackingConfiguration.isSupported else {
            sendError("ARKit is not supported on this device.")
            return
        }
        guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
            sendError("LiDAR depth sensing is not available. Requires iPhone 12 Pro or later.")
            return
        }

        DispatchQueue.main.async { [weak self] in
            guard let self,
                  let rootVC = self.webView?.window?.rootViewController else { return }

            let scanVC = ARScanViewController()
            scanVC.modalPresentationStyle = .fullScreen
            // Points are streamed in real-time via sendChunk; onComplete just signals dismiss.
            scanVC.onComplete = { _ in }

            rootVC.present(scanVC, animated: true) { [weak self] in
                self?.send("{ status: 'scanning' }")
            }
        }
    }

    // MARK: - Real-time streaming

    /// Called by ARScanViewController on every batch flush (~2 000 pts, ~48 KB base64).
    /// JS accumulates these in a PointCloudBuffer — no large end-of-scan blob needed.
    func sendChunk(_ floats: [Float]) {
        guard !floats.isEmpty else { return }
        let data  = floats.withUnsafeBytes { Data($0) }
        let b64   = data.base64EncodedString()
        let count = floats.count / 6
        send("{ status: 'chunk', count: \(count), data: '\(b64)' }")
    }

    /// Stream one RGB snapshot immediately so web can upload incrementally.
    func sendSnapshot(index: Int,
                      jpeg: Data,
                      c2w: [Float],
                      K: [Float],
                      fw: Int,
                      fh: Int) {
        let b64    = jpeg.base64EncodedString()
        let c2wStr = c2w.map { String($0) }.joined(separator: ",")
        let kStr   = K.map   { String($0) }.joined(separator: ",")
        send("{ status: 'snapshot', index: \(index), snapshot: { jpeg:'\(b64)', c2w:[\(c2wStr)], K:[\(kStr)], fw:\(fw), fh:\(fh) } }")
    }

    /// Upload the full colored point cloud directly from iOS to the backend (bypasses WKWebView).
    /// When complete, sends { status: 'uploadComplete', url: '...' } to JS so SpaceBuilder
    /// can skip the browser XHR and just use the URL that's already on the server.
    func uploadPointCloudDirect(floats: [Float]) {
        guard let base = apiBase, let rid = roomId,
              let url  = URL(string: "\(base)/api/rooms/\(rid)/pointcloud") else {
            print("[ARBridge] uploadPointCloudDirect skipped — no apiBase/roomId")
            return
        }
        let data = floats.withUnsafeBytes { Data($0) }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        if let auth = authToken   { req.setValue(auth, forHTTPHeaderField: "Authorization") }
        if let dev  = deviceToken { req.setValue(dev,  forHTTPHeaderField: "X-Device-Token") }
        req.httpBody = data
        URLSession.shared.dataTask(with: req) { [weak self] respData, _, err in
            if let err { print("[ARBridge] uploadPointCloudDirect failed: \(err)"); return }
            guard let respData,
                  let json = try? JSONSerialization.jsonObject(with: respData) as? [String: Any],
                  let urlPath = json["url"] as? String else {
                print("[ARBridge] uploadPointCloudDirect: unexpected response")
                return
            }
            self?.sendUploadComplete(url: urlPath)
        }.resume()
    }

    /// Notify JS that the iOS-side point cloud upload finished.
    private func sendUploadComplete(url: String) {
        let safe = url
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "'",  with: "\\'")
        send("{ status: 'uploadComplete', url: '\(safe)' }")
    }

    /// Upload one JPEG snapshot directly to the backend via URLSession (bypasses WKWebView).
    /// Fire-and-forget — failures are logged but don't block the scan.
    func uploadSnapshotDirect(index: Int, jpeg: Data, c2w: [Float], K: [Float], fw: Int, fh: Int) {
        guard let base = apiBase, let rid = roomId,
              let url  = URL(string: "\(base)/api/rooms/\(rid)/snapshots") else { return }

        let c2wStr = c2w.map { String($0) }.joined(separator: ",")
        let kStr   = K.map   { String($0) }.joined(separator: ",")
        let b64    = jpeg.base64EncodedString()
        let body   = """
        {"index":\(index),"jpeg":"\(b64)","c2w":[\(c2wStr)],"K":[\(kStr)],"fw":\(fw),"fh":\(fh)}
        """
        guard let bodyData = body.data(using: .utf8) else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let auth = authToken   { req.setValue(auth,  forHTTPHeaderField: "Authorization") }
        if let dev  = deviceToken { req.setValue(dev,   forHTTPHeaderField: "X-Device-Token") }
        req.httpBody = bodyData
        URLSession.shared.dataTask(with: req) { _, _, err in
            if let err { print("[ARBridge] snapshot upload \(index) failed: \(err)") }
        }.resume()
    }

    /// Signals that on-device photo projection has started.
    /// JS should show a "Projecting…" state and pause the upload pipeline.
    func sendProjecting() {
        send("{ status: 'projecting' }")
    }

    /// Called once when projection is complete and all colored chunks have been sent.
    func sendDone(pointCount: Int) {
        let ms = Int(Date().timeIntervalSince1970 * 1000)
        send("{ status: 'done', pointCount: \(pointCount), capturedAt: \(ms) }")
    }
}

// MARK: - WKScriptMessageHandler

extension ARBridge: WKScriptMessageHandler {
    func userContentController(_ ucc: WKUserContentController,
                               didReceive message: WKScriptMessage) {
        guard let body   = message.body as? [String: Any],
              let action = body["action"] as? String else { return }
        if action == "startScan" { startScan(body: body) }
        // stopScan handled natively via the Done button in ARScanViewController
    }
}
