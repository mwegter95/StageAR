/**
 * ARBridge.swift
 *
 * Thin web ↔ native bridge. Receives messages from the web app and presents
 * ARScanViewController. When scanning is done, serialises the point cloud and
 * calls window.onStageARResult() on the web page.
 *
 * Web → Native:
 *   window.webkit.messageHandlers.stageAR.postMessage({ action: 'startScan' })
 *
 * Native → Web:
 *   window.onStageARResult({ status: 'scanning' })
 *   window.onStageARResult({ status: 'done', pointCount: N, points: […], capturedAt: ms })
 *   window.onStageARResult({ error: '…' })
 */

import WebKit
import ARKit

final class ARBridge: NSObject {

    static let shared = ARBridge()
    weak var webView: WKWebView?

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

    /// Send a camera-frame snapshot to JS so the web app can re-project each
    /// LiDAR point through the best-covering photo and replace its low-res
    /// depth-sensor colour with the true high-res photo colour.
    ///
    /// - Parameters:
    ///   - jpegData:   JPEG bytes of the (already resized) camera frame
    ///   - transform:  column-major 4×4 camera→world matrix (16 floats)
    ///   - intrinsics: [fx, fy, cx, cy, imageWidth, imageHeight] scaled to the JPEG
    func sendSnapshot(_ jpegData: Data, transform: [Float], intrinsics: [Float]) {
        let b64   = jpegData.base64EncodedString()
        let tfStr = transform.map   { String(format: "%.5f", $0) }.joined(separator: ",")
        let inStr = intrinsics.map  { String(format: "%.2f", $0) }.joined(separator: ",")
        DispatchQueue.main.async { [weak self] in
            self?.webView?.evaluateJavaScript(
                "window.onStageARResult && window.onStageARResult({ status:'snapshot', dataUrl:'data:image/jpeg;base64,\(b64)', transform:[\(tfStr)], intrinsics:[\(inStr)] })",
                completionHandler: nil
            )
        }
    }

    // MARK: - Scan

    private func startScan() {
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

    /// Called once when the user taps Done. JS already has all points via chunk events.
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
        if action == "startScan" { startScan() }
        // stopScan handled natively via the Done button in ARScanViewController
    }
}
