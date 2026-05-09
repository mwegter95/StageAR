/**
 * ARScanViewController.swift
 *
 * Full-screen AR scanning view. Presented modally over the WKWebView when the
 * user starts a scan. ARSCNView automatically shows the live camera passthrough.
 * Depth samples are unprojected into world-space, accumulated, and flushed to
 * the scene periodically as coloured SCNGeometry point batches.
 *
 * When the user taps Done, onComplete is called (no data — points have already been
 * streamed in real-time via ARBridge.shared.sendChunk()) and the VC dismisses itself.
 */

import UIKit
import ARKit
import SceneKit

class ARScanViewController: UIViewController {

    // Called on the main thread after the scan VC is dismissed.
    var onComplete: (([Float]) -> Void)?   // signature kept for compat; array is always empty

    // MARK: - Private types

    // Tightly-packed 3-float struct (stride = 12 bytes, no SIMD padding).
    private struct Float3 { var x, y, z: Float }

    // MARK: - Views
    private var sceneView:   ARSCNView!
    private var bottomBar:   UIView!
    private var countLabel:  UILabel!
    private var doneButton:  UIButton!

    // MARK: - Point cloud state
    // Points are streamed to JS in real-time via ARBridge.sendChunk().
    // We no longer accumulate all points in memory — only the live batch.
    private var totalPointsSent = 0   // cumulative counter, main-thread only

    // Current in-progress batch waiting to be flushed to the scene + streamed
    private var batchPos  = [Float3]()
    private var batchCol  = [Float3]()

    private var frameIndex    = 0
    private let SAMPLE_EVERY  = 1    // process depth every ARKit frame (60 fps → ~0.33 s full sweep)
    private let BATCH_FLUSH   = 10   // flush to SceneKit every N sampled frames (fewer, larger batches)
    private var sampledFrames = 0

    // MARK: - Photo snapshot state
    // Every SNAPSHOT_EVERY *sampled* frames we capture a high-res JPEG + camera
    // pose so the web app can re-texture the point cloud with true photo colour.
    private let SNAPSHOT_EVERY = 120   // ~4 s between snapshots at 30 captured fps
    private let MAX_SNAPSHOTS  = 75    // supports ~5 minute scans at the default snapshot cadence
    private var snapshotFrameCount = 0
    private var snapshotCount      = 0
    private var isCapturing        = false   // debounce — one JPEG at a time
    private lazy var ciContext     = CIContext()
    private var lastSnapshotPos: SIMD3<Float>? = nil
    private var lastSnapshotFwd: SIMD3<Float>? = nil
    private let SNAPSHOT_MIN_MOVE: Float = 0.25
    private let SNAPSHOT_MIN_ROT_DOT: Float = 0.985

    // Archimedean spiral scan pattern — precomputed pixel coordinates from center outward.
    // Each sampled frame processes BATCH_PER_FRAME consecutive spiral steps, creating a
    // "record groove" scanning visual in the live SceneKit view as the cloud builds up.
    private var spiralCoords  = [(Int32, Int32)]()
    private var spiralPhase   = 0
    private let BATCH_PER_FRAME = 2500   // spiral steps per frame; full 49152-px sweep in ~0.33 s at 60 fps
    private let MIN_DEPTH: Float  = 0.15
    private let MAX_DEPTH: Float  = 8.0  // indoor rooms rarely exceed 8 m
    private var pointSampleCounter = 0

    // SceneKit child-node merge: every N flushes, collapse all children into one
    // geometry node so we never accumulate thousands of separate SCNNodes.
    private var flushCount   = 0
    private let MERGE_EVERY  = 30

    // Root node that holds all batch child nodes
    private var pointCloudNode: SCNNode!
    private let PREVIEW_POINT_BUDGET = 250_000
    private var previewPointsRendered = 0

    // MARK: - Lifecycle

    override func viewDidLoad() {
        super.viewDidLoad()
        setupScene()
        setupHUD()
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        startARSession()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sceneView.session.pause()
    }

    // MARK: - Setup

    private func setupScene() {
        sceneView = ARSCNView(frame: view.bounds)
        sceneView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        sceneView.scene = SCNScene()
        sceneView.showsStatistics = false
        sceneView.antialiasingMode = .none
        // ARSCNView automatically renders the camera passthrough from the session.
        view.addSubview(sceneView)

        sceneView.session.delegate = self

        pointCloudNode = SCNNode()
        sceneView.scene.rootNode.addChildNode(pointCloudNode)
    }

    private func setupHUD() {
        // Blurred bottom bar
        let blur = UIVisualEffectView(effect: UIBlurEffect(style: .systemThinMaterialDark))
        blur.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(blur)

        // Point count label
        countLabel = UILabel()
        countLabel.translatesAutoresizingMaskIntoConstraints = false
        countLabel.text = "Scanning…"
        countLabel.textColor = .white
        countLabel.font = .monospacedDigitSystemFont(ofSize: 17, weight: .medium)
        blur.contentView.addSubview(countLabel)

        // Done button
        doneButton = UIButton(type: .system)
        doneButton.translatesAutoresizingMaskIntoConstraints = false
        doneButton.setTitle("Done  ✓", for: .normal)
        doneButton.titleLabel?.font = .systemFont(ofSize: 16, weight: .semibold)
        doneButton.setTitleColor(.white, for: .normal)
        doneButton.backgroundColor = UIColor(red: 0.20, green: 0.78, blue: 0.60, alpha: 1)
        doneButton.layer.cornerRadius = 18
        doneButton.contentEdgeInsets = UIEdgeInsets(top: 0, left: 22, bottom: 0, right: 22)
        doneButton.addTarget(self, action: #selector(doneTapped), for: .touchUpInside)
        blur.contentView.addSubview(doneButton)

        let safeBottom = view.safeAreaLayoutGuide.bottomAnchor
        NSLayoutConstraint.activate([
            blur.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            blur.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            blur.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            blur.topAnchor.constraint(equalTo: safeBottom, constant: -72),

            countLabel.leadingAnchor.constraint(equalTo: blur.contentView.leadingAnchor, constant: 28),
            countLabel.bottomAnchor.constraint(equalTo: safeBottom, constant: -18),

            doneButton.trailingAnchor.constraint(equalTo: blur.contentView.trailingAnchor, constant: -28),
            doneButton.bottomAnchor.constraint(equalTo: safeBottom, constant: -18),
            doneButton.heightAnchor.constraint(equalToConstant: 36),
        ])
    }

    // MARK: - AR Session

    private func startARSession() {
        guard ARWorldTrackingConfiguration.isSupported,
              ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
            // Device doesn't support LiDAR depth — dismiss and report
            dismiss(animated: true) { [weak self] in
                self?.onComplete?([] )
            }
            return
        }

        let config = ARWorldTrackingConfiguration()

        // Prefer ultra-wide (0.5×) camera on iPhone 12 Pro+: wider FOV means more room
        // coverage per frame. Falls back to the default wide-angle on older/non-Pro devices.
        var cameraLabel = "Wide-angle"
        if #available(iOS 16.0, *) {
            let ultraWide = ARWorldTrackingConfiguration.supportedVideoFormats.filter {
                $0.captureDeviceType == .builtInUltraWideCamera
            }
            if let fmt = ultraWide.first {
                config.videoFormat = fmt
                cameraLabel = "Ultra-wide 0.5×"
            }
        }

        // Brief camera-mode indicator so the user can confirm which lens is active.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [weak self] in
            self?.countLabel.text = "📷 \(cameraLabel)"
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) { [weak self] in
                self?.countLabel.text = "Scanning…"
            }
        }

        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            config.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
        } else {
            config.frameSemantics = [.sceneDepth]
        }

        sceneView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    // MARK: - Actions

    @objc private func doneTapped() {
        sceneView.session.pause()
        let total = totalPointsSent
        dismiss(animated: true) { [weak self] in
            self?.onComplete?([])           // data already streamed; pass empty array
            ARBridge.shared.sendDone(pointCount: total)
        }
    }

    // MARK: - Depth processing

    /// Build an Archimedean spiral lookup table over the depth-map pixel grid.
    /// Runs once and caches result in `spiralCoords`. The spiral has ~2 px ring
    /// spacing, so it visits every pixel at approximately stride-2 density.
    private func buildSpiralCoords(dW: Int, dH: Int) {
        guard spiralCoords.isEmpty else { return }
        let cx     = Double(dW) * 0.5
        let cy     = Double(dH) * 0.5
        let maxR   = hypot(cx, cy) + 2.0    // slightly past the farthest corner
        // b = ring spacing / (2π) → 2-pixel spacing between adjacent rings
        let b      = 2.0 / (2.0 * Double.pi)
        let dTheta = 0.04                   // angular step in radians (~2.3°)
        var theta  = 0.0
        var seen   = Set<Int32>()
        while b * theta <= maxR {
            let r  = b * theta
            let px = Int(round(cx + r * cos(theta)))
            let py = Int(round(cy + r * sin(theta)))
            theta += dTheta
            guard px >= 0 && px < dW && py >= 0 && py < dH else { continue }
            let key = Int32(py * dW + px)
            if seen.insert(key).inserted {
                spiralCoords.append((Int32(px), Int32(py)))
            }
        }
    }

    private func processFrame(_ frame: ARFrame) {
        guard let depthData = frame.smoothedSceneDepth ?? frame.sceneDepth else { return }

        let depthMap = depthData.depthMap
        let confMap  = depthData.confidenceMap

        let dW = CVPixelBufferGetWidth(depthMap)
        let dH = CVPixelBufferGetHeight(depthMap)

        CVPixelBufferLockBaseAddress(depthMap,          .readOnly)
        if let c = confMap { CVPixelBufferLockBaseAddress(c, .readOnly) }
        CVPixelBufferLockBaseAddress(frame.capturedImage, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
            if let c = confMap { CVPixelBufferUnlockBaseAddress(c, .readOnly) }
            CVPixelBufferUnlockBaseAddress(frame.capturedImage, .readOnly)
        }

        guard let depthBase = CVPixelBufferGetBaseAddress(depthMap) else { return }
        let depthPtr = depthBase.assumingMemoryBound(to: Float32.self)

        var confPtr: UnsafeMutablePointer<UInt8>? = nil
        if let c = confMap, let b = CVPixelBufferGetBaseAddress(c) {
            confPtr = b.assumingMemoryBound(to: UInt8.self)
        }

        // Scale intrinsics from camera image resolution → depth map resolution
        let intr   = frame.camera.intrinsics
        let imgRes = frame.camera.imageResolution
        let sx     = Float(dW) / Float(imgRes.width)
        let sy     = Float(dH) / Float(imgRes.height)
        let fx = intr[0][0] * sx;  let cx = intr[2][0] * sx
        let fy = intr[1][1] * sy;  let cy = intr[2][1] * sy
        let T  = frame.camera.transform

        // Archimedean spiral scan: each frame processes a contiguous arc of the spiral,
        // advancing the phase pointer so subsequent frames continue where the last left off.
        // This creates a flowing groove-like scan pattern in the live SceneKit view.
        buildSpiralCoords(dW: dW, dH: dH)
        guard !spiralCoords.isEmpty else { return }

        let total = spiralCoords.count
        for i in 0..<BATCH_PER_FRAME {
            let (pxRaw, pyRaw) = spiralCoords[(spiralPhase + i) % total]
            let px = Int(pxRaw); let py = Int(pyRaw)
            let d = depthPtr[py * dW + px]
            guard d > MIN_DEPTH && d < MAX_DEPTH else { continue }
            if let cp = confPtr, cp[py * dW + px] < 1 { continue }

            // Adaptive sampling for long scans: preserve visual quality while
            // preventing unbounded growth that can crash older devices.
            let projected = totalPointsSent + batchPos.count
            let keepEvery: Int
            if projected < 2_000_000 {
                keepEvery = 1
            } else if projected < 4_000_000 {
                keepEvery = 2
            } else if projected < 8_000_000 {
                keepEvery = 3
            } else {
                keepEvery = 4
            }
            pointSampleCounter += 1
            if keepEvery > 1 && (pointSampleCounter % keepEvery != 0) { continue }

            // Camera space → world space (ARKit: camera looks along -Z, Y is up)
            let xc =  (Float(px) - cx) / fx * d
            let yc = -(Float(py) - cy) / fy * d
            let zc = -d
            let wp = T * SIMD4<Float>(xc, yc, zc, 1)

            let (r, g, b) = sampleYCbCr(frame.capturedImage,
                                        dx: px, dy: py, dW: dW, dH: dH)

            batchPos.append(Float3(x: wp.x, y: wp.y, z: wp.z))
            batchCol.append(Float3(x: r,    y: g,    z: b   ))
        }
        spiralPhase = (spiralPhase + BATCH_PER_FRAME) % total
    }

    /// Improved YCbCr→RGB: BT.709 coefficients, proper video-range scaling,
    /// and 3×3 neighbourhood averaging for smooth, accurate colour output.
    ///
    /// ARKit’s capturedImage is in BT.709 video range:
    ///   Y  ∈ [16, 235] (luma),  CbCr ∈ [16, 240] centred at 128 (chroma)
    private func sampleYCbCr(_ buf: CVPixelBuffer,
                              dx: Int, dy: Int, dW: Int, dH: Int) -> (Float, Float, Float) {
        let iW = CVPixelBufferGetWidthOfPlane(buf, 0)
        let iH = CVPixelBufferGetHeightOfPlane(buf, 0)
        guard let yBase = CVPixelBufferGetBaseAddressOfPlane(buf, 0),
              let cBase = CVPixelBufferGetBaseAddressOfPlane(buf, 1) else { return (0.5, 0.5, 0.5) }
        let yBPR = CVPixelBufferGetBytesPerRowOfPlane(buf, 0)
        let cBPR = CVPixelBufferGetBytesPerRowOfPlane(buf, 1)
        let yPtr = yBase.assumingMemoryBound(to: UInt8.self)
        let cPtr = cBase.assumingMemoryBound(to: UInt8.self)

        let ix0 = dx * iW / dW
        let iy0 = dy * iH / dH

        // Average a 3×3 neighbourhood (in image coords) for sub-pixel colour accuracy
        var sumY: Float = 0; var sumCb: Float = 0; var sumCr: Float = 0
        for dy2 in -1...1 {
            let iy  = max(0, min(iH - 1,     iy0 + dy2))
            let iy2 = max(0, min(iH / 2 - 1, iy  / 2  ))
            for dx2 in -1...1 {
                let ix  = max(0, min(iW - 1,     ix0 + dx2))
                let ix2 = max(0, min(iW / 2 - 1, ix  / 2  ))
                // BT.709 video range: Y ∈ [16,235] → /219,  CbCr ∈ [16,240] centred 128 → /224
                sumY  += (Float(yPtr[iy  * yBPR + ix       ]) - 16.0)  / 219.0
                sumCb += (Float(cPtr[iy2 * cBPR + ix2 * 2  ]) - 128.0) / 224.0
                sumCr += (Float(cPtr[iy2 * cBPR + ix2 * 2 + 1]) - 128.0) / 224.0
            }
        }
        let Y = sumY / 9.0; let Cb = sumCb / 9.0; let Cr = sumCr / 9.0

        // ITU-R BT.709 matrix (iPhone camera encodes in Rec.709 / sRGB primaries)
        let r = min(1, max(0, Y + 1.5748 * Cr))
        let g = min(1, max(0, Y - 0.1873 * Cb - 0.4681 * Cr))
        let b = min(1, max(0, Y + 1.8556 * Cb))
        return (r, g, b)
    }

    // MARK: - Snapshot capture
    // Captures a 960-px-wide JPEG of the current camera frame along with the
    // camera's world transform and intrinsics.  The web app uses these to
    // re-project each LiDAR point through the best-covering snapshot and replace
    // its low-res depth-sensor colour with the true high-res photo colour.
    private func captureSnapshot(_ frame: ARFrame) {
        guard !isCapturing else { return }
        isCapturing = true

        let pixBuf = frame.capturedImage
        let origW  = CVPixelBufferGetWidth(pixBuf)
        let origH  = CVPixelBufferGetHeight(pixBuf)

        // Scale to 640 wide for a safer memory profile on long scans.
        let targetW: CGFloat = 640
        let scale            = targetW / CGFloat(origW)
        let targetH          = CGFloat(origH) * scale

        let ci     = CIImage(cvPixelBuffer: pixBuf)
        let scaled = ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))

        guard let cgImg    = ciContext.createCGImage(scaled, from: scaled.extent),
              let jpegData = UIImage(cgImage: cgImg).jpegData(compressionQuality: 0.58)
        else { isCapturing = false; return }

        // Camera world transform — column-major 16 floats
        // ARKit: camera looks along -Z; Y is up; transform = camera→world matrix.
        let c = frame.camera.transform
        let transform: [Float] = [
            c.columns.0.x, c.columns.0.y, c.columns.0.z, c.columns.0.w,
            c.columns.1.x, c.columns.1.y, c.columns.1.z, c.columns.1.w,
            c.columns.2.x, c.columns.2.y, c.columns.2.z, c.columns.2.w,
            c.columns.3.x, c.columns.3.y, c.columns.3.z, c.columns.3.w,
        ]

        // Intrinsics scaled to the output JPEG resolution.
        // frame.camera.intrinsics is column-major simd_float3x3:
        //   col0 = (fx, 0, 0), col1 = (0, fy, 0), col2 = (cx, cy, 1)
        let intr = frame.camera.intrinsics
        let sx   = Float(scale)
        let sy   = Float(scale)
        let intrinsics: [Float] = [
            intr[0][0] * sx,  // fx
            intr[1][1] * sy,  // fy
            intr[2][0] * sx,  // cx
            intr[2][1] * sy,  // cy
            Float(targetW),   // imageWidth
            Float(targetH),   // imageHeight
        ]

        ARBridge.shared.sendSnapshot(jpegData, transform: transform, intrinsics: intrinsics)
        lastSnapshotPos = SIMD3<Float>(c.columns.3.x, c.columns.3.y, c.columns.3.z)
        lastSnapshotFwd = simd_normalize(SIMD3<Float>(-c.columns.2.x, -c.columns.2.y, -c.columns.2.z))
        isCapturing = false
    }

    private func shouldCaptureSnapshot(_ frame: ARFrame) -> Bool {
        guard let lastPos = lastSnapshotPos, let lastFwd = lastSnapshotFwd else { return true }
        let c = frame.camera.transform
        let pos = SIMD3<Float>(c.columns.3.x, c.columns.3.y, c.columns.3.z)
        let fwd = simd_normalize(SIMD3<Float>(-c.columns.2.x, -c.columns.2.y, -c.columns.2.z))
        let moved = simd_length(pos - lastPos)
        let dotv = simd_dot(fwd, lastFwd)
        return moved >= SNAPSHOT_MIN_MOVE || dotv <= SNAPSHOT_MIN_ROT_DOT
    }

    private func previewBatch(pos: [Float3], col: [Float3]) -> ([Float3], [Float3]) {
        let remaining = PREVIEW_POINT_BUDGET - previewPointsRendered
        guard remaining > 0 else { return ([], []) }
        guard pos.count > remaining else { return (pos, col) }

        let stride = max(1, Int(ceil(Double(pos.count) / Double(remaining))))
        var previewPos = [Float3]()
        var previewCol = [Float3]()
        previewPos.reserveCapacity(remaining)
        previewCol.reserveCapacity(remaining)

        var index = 0
        while index < pos.count && previewPos.count < remaining {
            previewPos.append(pos[index])
            previewCol.append(col[index])
            index += stride
        }
        return (previewPos, previewCol)
    }

    // MARK: - SceneKit geometry

    private func flushBatchToScene() {
        guard !batchPos.isEmpty else { return }
        let pos = batchPos;  let col = batchCol
        batchPos.removeAll();  batchCol.removeAll()

        // Interleave pos + col into [x,y,z,r,g,b, …] for the JS streaming bridge.
        // Happens on the background thread before the main-queue dispatch.
        var interleaved = [Float]()
        interleaved.reserveCapacity(pos.count * 6)
        for i in 0..<pos.count {
            interleaved.append(pos[i].x); interleaved.append(pos[i].y); interleaved.append(pos[i].z)
            interleaved.append(col[i].x); interleaved.append(col[i].y); interleaved.append(col[i].z)
        }
        ARBridge.shared.sendChunk(interleaved)   // dispatches to main internally

        flushCount += 1
        let shouldMerge = flushCount >= MERGE_EVERY
        if shouldMerge { flushCount = 0 }

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }

            self.totalPointsSent += pos.count    // main-thread-only counter

            let (previewPos, previewCol) = self.previewBatch(pos: pos, col: col)
            if !previewPos.isEmpty {
                let geo  = self.makePointGeo(pos: previewPos, col: previewCol)
                let node = SCNNode(geometry: geo)
                self.pointCloudNode.addChildNode(node)
                self.previewPointsRendered += previewPos.count

                // Periodically merge all child nodes into a single geometry to prevent
                // SceneKit from choking on hundreds of separate SCNNode objects.
                if shouldMerge {
                    self.mergePointCloudNodes()
                }
            }

            let n = self.totalPointsSent
            let baseLabel = n >= 1_000_000
                ? String(format: "%.2fM pts", Double(n) / 1_000_000.0)
                : n >= 1000
                    ? String(format: "%.1fk pts", Double(n) / 1000.0)
                    : "\(n) pts"
            self.countLabel.text = self.previewPointsRendered >= self.PREVIEW_POINT_BUDGET
                ? "\(baseLabel) • preview capped"
                : baseLabel
        }
    }

    /// Collapse all SCNNode children of pointCloudNode into a single node.
    /// This keeps the SceneKit scene graph flat even after hundreds of flush cycles.
    private func mergePointCloudNodes() {
        let children = pointCloudNode.childNodes
        guard children.count > 1 else { return }

        // Collect all position and colour data from every child geometry
        var mergedPos = [Float3]()
        var mergedCol = [Float3]()

        for child in children {
            guard let geo = child.geometry,
                  let posSrc = geo.sources(for: .vertex).first,
                  let colSrc = geo.sources(for: .color).first else { continue }

            let n = posSrc.vectorCount
            posSrc.data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                let ptr = raw.baseAddress!.assumingMemoryBound(to: Float3.self)
                mergedPos.append(contentsOf: UnsafeBufferPointer(start: ptr, count: n))
            }
            colSrc.data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                let ptr = raw.baseAddress!.assumingMemoryBound(to: Float3.self)
                mergedCol.append(contentsOf: UnsafeBufferPointer(start: ptr, count: n))
            }
        }

        guard !mergedPos.isEmpty else { return }

        // Remove all existing children
        children.forEach { $0.removeFromParentNode() }

        // Add a single merged node
        let merged = SCNNode(geometry: makePointGeo(pos: mergedPos, col: mergedCol))
        pointCloudNode.addChildNode(merged)
    }

    private func makePointGeo(pos: [Float3], col: [Float3]) -> SCNGeometry {
        let n = pos.count
        let posData = Data(bytes: pos, count: n * MemoryLayout<Float3>.stride)
        let colData = Data(bytes: col, count: n * MemoryLayout<Float3>.stride)

        let stride = MemoryLayout<Float3>.stride   // 12

        let posSrc = SCNGeometrySource(
            data: posData, semantic: .vertex,
            vectorCount: n, usesFloatComponents: true,
            componentsPerVector: 3, bytesPerComponent: 4,
            dataOffset: 0, dataStride: stride)

        let colSrc = SCNGeometrySource(
            data: colData, semantic: .color,
            vectorCount: n, usesFloatComponents: true,
            componentsPerVector: 3, bytesPerComponent: 4,
            dataOffset: 0, dataStride: stride)

        var indices = (0..<n).map { Int32($0) }
        let idxData = Data(bytes: &indices, count: n * 4)
        let element = SCNGeometryElement(data: idxData, primitiveType: .point,
                                         primitiveCount: n, bytesPerIndex: 4)
        element.pointSize                   = 8
        element.minimumPointScreenSpaceRadius = 2
        element.maximumPointScreenSpaceRadius = 12

        return SCNGeometry(sources: [posSrc, colSrc], elements: [element])
    }
}

// MARK: - ARSessionDelegate

extension ARScanViewController: ARSessionDelegate {

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        frameIndex += 1
        guard frameIndex % SAMPLE_EVERY == 0 else { return }

        processFrame(frame)

        sampledFrames += 1
        if sampledFrames >= BATCH_FLUSH {
            sampledFrames = 0
            flushBatchToScene()
        }

        // Photo snapshot — every SNAPSHOT_EVERY sampled frames, up to MAX_SNAPSHOTS
        snapshotFrameCount += 1
        if snapshotFrameCount >= SNAPSHOT_EVERY && snapshotCount < MAX_SNAPSHOTS {
            snapshotFrameCount = 0
            if shouldCaptureSnapshot(frame) {
                snapshotCount += 1
                captureSnapshot(frame)
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        dismiss(animated: true) { [weak self] in self?.onComplete?([]) }
    }
}
