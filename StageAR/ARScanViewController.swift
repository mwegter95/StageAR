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
    // Points are buffered in memory during the scan and photo-projected after Done.
    // The colored cloud is streamed to JS only after projection completes.
    private var totalPointsSent   = 0   // cumulative counter, main-thread only
    private var accumulatedCloud  = [Float]()   // [x,y,z,r,g,b …] — full scan buffer
    private var capturedSnapshots = [PhotoProjector.SnapshotCapture]()

    // Current in-progress batch waiting to be flushed to the scene + buffered
    private var batchPos  = [Float3]()
    private var batchCol  = [Float3]()

    private var frameIndex    = 0
    private let SAMPLE_EVERY  = 1    // process depth every ARKit frame (60 fps → ~0.33 s full sweep)
    private let BATCH_FLUSH   = 10   // flush to SceneKit every N sampled frames (fewer, larger batches)
    private var sampledFrames = 0
    private let CAPTURE_MAX_LINEAR_SPEED: Float = 1.05   // m/s; skip very unstable camera motion
    private let CAPTURE_MAX_ANGULAR_SPEED: Float = 3.4   // rad/s
    private var lastFramePos: SIMD3<Float>? = nil
    private var lastFrameFwd: SIMD3<Float>? = nil
    private var lastFrameTs: TimeInterval? = nil

    // Archimedean spiral scan pattern — precomputed pixel coordinates from center outward.
    // Each sampled frame processes BATCH_PER_FRAME consecutive spiral steps, creating a
    // "record groove" scanning visual in the live SceneKit view as the cloud builds up.
    private var spiralCoords  = [(Int32, Int32)]()
    private var spiralPhase   = 0
    private let BATCH_PER_FRAME_STABLE = 5200 // stable motion: denser cloud faster
    private let BATCH_PER_FRAME_MOVING = 3200 // moderate motion: balance density and stability
    private let BATCH_PER_FRAME_FAST   = 1800 // rapid motion: prioritize stability
    private let MIN_DEPTH: Float  = 0.15
    private let MAX_DEPTH: Float  = 8.0  // indoor rooms rarely exceed 8 m
    private var pointSampleCounter = 0

    private var lastSnapshotPos: SIMD3<Float>? = nil
    private var snapshotFrameCounter = 0
    private var snapshotIndex = 0
    private var pendingSnapshotEncodes = 0
    private var isFinishingScan = false
    private let SNAPSHOT_INTERVAL_FRAMES = 12  // ~0.2 s at 60 fps
    private let SNAPSHOT_MIN_MOVE: Float   = 0.08 // 8 cm minimum between snapshots
    private let SNAPSHOT_MAX_COUNT         = 48

    // Shared CIContext — CIContext is thread-safe; creating one per snapshot (up to 48)
    // exhausts Metal render-pipeline resources and makes createCGImage return nil.
    // Lazy so it's initialised the first time maybeCapture fires, not at VC load time.
    private lazy var snapshotCIContext: CIContext = {
        CIContext(options: [.workingColorSpace: CGColorSpaceCreateDeviceRGB(),
                            .useSoftwareRenderer: false])
    }()

    // Root node that holds all batch child nodes
    private var pointCloudNode: SCNNode!
    private let PREVIEW_BATCH_TARGET = 5_000
    private let PREVIEW_NODE_LIMIT   = 55
    private let PREVIEW_CELL_SIZE: Float = 0.045

    // Stabilization countdown — ARKit needs a few seconds to lock tracking
    // before depth poses are reliable. During this window we reject all depth
    // frames so shaky startup motion doesn't corrupt the cloud.
    private let STABILIZE_SECS: Double = 4.0
    private var stabilizeUntil: Date   = .distantPast    // set on first ARKit frame
    private var hasReceivedFirstFrame  = false            // guards the one-time timer start
    private var stabilizeBar: UIView!
    private var stabilizeBarFill: UIView!
    private var stabilizeLabel: UILabel!

    // Warm-up after stabilization: discard the first N depth frames so residual
    // tracking noise (from the brief period when ARKit finishes convergence)
    // doesn't produce jumpy dots at scan start.
    private let WARMUP_FRAMES = 30   // ~0.5 s at 60 fps
    private var warmupFramesRemaining = 0

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
        var doneConfig = doneButton.configuration ?? UIButton.Configuration.filled()
        doneConfig.contentInsets = NSDirectionalEdgeInsets(top: 0, leading: 22, bottom: 0, trailing: 22)
        doneButton.configuration = doneConfig
        doneButton.addTarget(self, action: #selector(doneTapped), for: .touchUpInside)
        blur.contentView.addSubview(doneButton)

        // Stabilization overlay — full-screen centered card
        let stabOverlay = UIView()
        stabOverlay.translatesAutoresizingMaskIntoConstraints = false
        stabOverlay.backgroundColor = UIColor.black.withAlphaComponent(0.55)
        stabOverlay.tag = 9001
        view.addSubview(stabOverlay)

        let stabCard = UIVisualEffectView(effect: UIBlurEffect(style: .systemThinMaterialDark))
        stabCard.translatesAutoresizingMaskIntoConstraints = false
        stabCard.layer.cornerRadius = 18
        stabCard.clipsToBounds = true
        stabOverlay.addSubview(stabCard)

        stabilizeLabel = UILabel()
        stabilizeLabel.translatesAutoresizingMaskIntoConstraints = false
        stabilizeLabel.text = "Stabilizing AR tracking…"
        stabilizeLabel.textColor = .white
        stabilizeLabel.font = .systemFont(ofSize: 15, weight: .medium)
        stabilizeLabel.textAlignment = .center
        stabCard.contentView.addSubview(stabilizeLabel)

        // Progress track
        let track = UIView()
        track.translatesAutoresizingMaskIntoConstraints = false
        track.backgroundColor = UIColor.white.withAlphaComponent(0.15)
        track.layer.cornerRadius = 3
        track.clipsToBounds = true
        stabCard.contentView.addSubview(track)

        stabilizeBarFill = UIView()
        stabilizeBarFill.translatesAutoresizingMaskIntoConstraints = false
        stabilizeBarFill.backgroundColor = UIColor(red: 0.20, green: 0.78, blue: 0.60, alpha: 1)
        stabilizeBarFill.layer.cornerRadius = 3
        stabilizeBarFill.clipsToBounds = true
        track.addSubview(stabilizeBarFill)
        stabilizeBar = track

        let stabHint = UILabel()
        stabHint.translatesAutoresizingMaskIntoConstraints = false
        stabHint.text = "Hold the camera steady"
        stabHint.textColor = UIColor.white.withAlphaComponent(0.55)
        stabHint.font = .systemFont(ofSize: 12, weight: .regular)
        stabHint.textAlignment = .center
        stabCard.contentView.addSubview(stabHint)

        NSLayoutConstraint.activate([
            stabOverlay.topAnchor.constraint(equalTo: view.topAnchor),
            stabOverlay.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            stabOverlay.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            stabOverlay.trailingAnchor.constraint(equalTo: view.trailingAnchor),

            stabCard.centerXAnchor.constraint(equalTo: stabOverlay.centerXAnchor),
            stabCard.centerYAnchor.constraint(equalTo: stabOverlay.centerYAnchor),
            stabCard.widthAnchor.constraint(equalToConstant: 260),

            stabilizeLabel.topAnchor.constraint(equalTo: stabCard.contentView.topAnchor, constant: 24),
            stabilizeLabel.leadingAnchor.constraint(equalTo: stabCard.contentView.leadingAnchor, constant: 20),
            stabilizeLabel.trailingAnchor.constraint(equalTo: stabCard.contentView.trailingAnchor, constant: -20),

            track.topAnchor.constraint(equalTo: stabilizeLabel.bottomAnchor, constant: 14),
            track.leadingAnchor.constraint(equalTo: stabCard.contentView.leadingAnchor, constant: 20),
            track.trailingAnchor.constraint(equalTo: stabCard.contentView.trailingAnchor, constant: -20),
            track.heightAnchor.constraint(equalToConstant: 6),

            stabilizeBarFill.topAnchor.constraint(equalTo: track.topAnchor),
            stabilizeBarFill.bottomAnchor.constraint(equalTo: track.bottomAnchor),
            stabilizeBarFill.leadingAnchor.constraint(equalTo: track.leadingAnchor),

            stabHint.topAnchor.constraint(equalTo: track.bottomAnchor, constant: 10),
            stabHint.leadingAnchor.constraint(equalTo: stabCard.contentView.leadingAnchor, constant: 20),
            stabHint.trailingAnchor.constraint(equalTo: stabCard.contentView.trailingAnchor, constant: -20),
            stabHint.bottomAnchor.constraint(equalTo: stabCard.contentView.bottomAnchor, constant: -20),
        ])
        // Width starts at 0 — updated each frame in didUpdate
        stabilizeBarFill.widthAnchor.constraint(equalToConstant: 0).isActive = true

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

        // Keep the default (wide-angle) ARKit video format for LiDAR scanning.
        //
        // Why NOT ultra-wide here:
        //   The LiDAR depth map is physically co-located with the MAIN camera and
        //   always covers the main camera's FOV, regardless of which color camera
        //   is selected.  If we set config.videoFormat to builtInUltraWideCamera,
        //   frame.camera.intrinsics returns ultra-wide K values, but
        //   frame.sceneDepth.depthMap still covers the main-camera FOV.
        //   processFrame() uses those K values to unproject depth pixels → the
        //   ultra-wide K applied to a wide-angle depth map places every 3D point
        //   at the wrong world position.  ARKit may also silently ignore the
        //   format when sceneDepth is requested.
        //
        //   Ultra-wide SNAPSHOTS use a separate AVCaptureSession on builtInUltraWideCamera
        //   (see startUWCaptureSession).  Per-frame K comes from the CMSampleBuffer
        //   intrinsics attachment; c2w remains the ARKit main-camera pose.
        let cameraLabel = "Wide + UW snaps"

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

        stabilizeUntil = .distantPast   // will be reset on the first ARKit frame
        hasReceivedFirstFrame = false
        warmupFramesRemaining = 0
        sceneView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        // UW session is started on the first ARKit frame (see ARSessionDelegate) so ARKit
        // has fully claimed its ISP resources before we compete for the UW sensor.
    }

    // MARK: - Actions

    @objc private func doneTapped() {
        doneButton.isEnabled = false
        isFinishingScan = true
        sceneView.session.pause()
        flushBatchToScene()
        // Release preview nodes before dismiss to avoid retaining large SceneKit buffers.
        let children = pointCloudNode.childNodes
        for node in children {
            node.geometry = nil
            node.removeFromParentNode()
        }
        finishScanIfReady()
    }

    private func finishScanIfReady() {
        guard isFinishingScan, pendingSnapshotEncodes == 0 else { return }
        isFinishingScan = false

        countLabel.text = "Projecting…"
        doneButton.isHidden = true

        let cloud     = accumulatedCloud
        let snapshots = capturedSnapshots
        let total     = accumulatedCloud.count / 6

        // Free the originals immediately — the locals above now own the data.
        // This reclaims the 240 MB on the main thread before we copy again inside
        // the background task, keeping peak memory at ~one copy instead of three.
        accumulatedCloud.removeAll(keepingCapacity: false)
        capturedSnapshots.removeAll()

        ARBridge.shared.sendProjecting()

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            var coloredCloud = cloud
            PhotoProjector.project(
                cloud:      &coloredCloud,
                pointCount: total,
                captures:   snapshots
            ) { pct in
                DispatchQueue.main.async { [weak self] in
                    self?.countLabel.text = String(format: "Projecting %.0f%%", pct * 100)
                }
            }

            DispatchQueue.main.async { [weak self] in
                guard let self else { return }

                // Fire direct iOS→backend upload (URLSession background task).
                ARBridge.shared.uploadPointCloudDirect(floats: coloredCloud)

                // Dismiss immediately so the web UI (name input, save button) is
                // interactive right away.  Chunk streaming continues on the main
                // thread after dismiss — the local function below holds coloredCloud
                // alive via closure capture until the last chunk is sent.
                self.dismiss(animated: true) {
                    self.onComplete?([])
                }

                // Stream one chunk per run-loop iteration.
                //
                // The old while-loop approach called sendChunk 100× in a tight loop.
                // Each sendChunk called ARBridge.send which dispatched async to main,
                // so all 100 closures queued up before any executed — each holding a
                // ~3.2 MB base64 string = ~320 MB piled in the queue.  Combined with
                // the two 240 MB cloud copies this pushed peak to ~800 MB → OOM crash.
                //
                // Here we send one chunk, yield to the run loop (so WKWebView's XPC
                // layer can digest it), then send the next.  Peak overhead: ~3 MB.
                let chunkFloats = 100_000 * 6
                func sendNext(offset: Int) {
                    guard offset < coloredCloud.count else {
                        ARBridge.shared.sendDone(pointCount: total)
                        return
                    }
                    let end = min(offset + chunkFloats, coloredCloud.count)
                    ARBridge.shared.sendChunk(Array(coloredCloud[offset..<end]))
                    DispatchQueue.main.async { sendNext(offset: end) }
                }
                sendNext(offset: 0)
            }
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

    private func samplesForMotion(linearSpeed: Float, angularSpeed: Float) -> Int {
        if linearSpeed > 0.8 || angularSpeed > 2.4 { return BATCH_PER_FRAME_FAST }
        if linearSpeed > 0.42 || angularSpeed > 1.45 { return BATCH_PER_FRAME_MOVING }
        return BATCH_PER_FRAME_STABLE
    }

    private func processFrame(_ frame: ARFrame, samplesPerFrame: Int) {
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
        for i in 0..<samplesPerFrame {
            let (pxRaw, pyRaw) = spiralCoords[(spiralPhase + i) % total]
            let px = Int(pxRaw); let py = Int(pyRaw)
            let d = depthPtr[py * dW + px]
            guard d > MIN_DEPTH && d < MAX_DEPTH else { continue }
            if let cp = confPtr, cp[py * dW + px] < 1 { continue }

            // Adaptive sampling for long scans: preserve visual quality while
            // preventing unbounded growth that can crash older devices.
            let projected = totalPointsSent + batchPos.count
            let keepEvery: Int
            if projected < 4_000_000 {
                keepEvery = 1
            } else if projected < 8_000_000 {
                keepEvery = 2
            } else if projected < 14_000_000 {
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
        spiralPhase = (spiralPhase + samplesPerFrame) % total
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

    private func previewBatch(pos: [Float3], col: [Float3]) -> ([Float3], [Float3]) {
        guard !pos.isEmpty else { return ([], []) }
        if pos.count <= PREVIEW_BATCH_TARGET { return (pos, col) }

        var covered = Set<String>()
        var previewPos = [Float3]()
        var previewCol = [Float3]()
        previewPos.reserveCapacity(PREVIEW_BATCH_TARGET)
        previewCol.reserveCapacity(PREVIEW_BATCH_TARGET)

        for index in 0..<pos.count {
            let p = pos[index]
            let ix = Int(floor(p.x / PREVIEW_CELL_SIZE))
            let iy = Int(floor(p.y / PREVIEW_CELL_SIZE))
            let iz = Int(floor(p.z / PREVIEW_CELL_SIZE))
            let key = "\(ix):\(iy):\(iz)"
            if covered.insert(key).inserted {
                previewPos.append(p)
                previewCol.append(col[index])
                if previewPos.count >= PREVIEW_BATCH_TARGET { break }
            }
        }

        if previewPos.count < PREVIEW_BATCH_TARGET {
            let needed = PREVIEW_BATCH_TARGET - previewPos.count
            let stride = max(1, pos.count / max(1, needed))
            var i = 0
            while i < pos.count && previewPos.count < PREVIEW_BATCH_TARGET {
                previewPos.append(pos[i])
                previewCol.append(col[i])
                i += stride
            }
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
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }

            self.totalPointsSent += pos.count    // main-thread-only counter
            self.accumulatedCloud.append(contentsOf: interleaved)

            let (previewPos, previewCol) = self.previewBatch(pos: pos, col: col)
            if !previewPos.isEmpty {
                let geo  = self.makePointGeo(pos: previewPos, col: previewCol)
                let node = SCNNode(geometry: geo)
                self.pointCloudNode.addChildNode(node)

                // Keep a rolling bounded preview so dots continue updating for long scans.
                while self.pointCloudNode.childNodes.count > self.PREVIEW_NODE_LIMIT {
                    guard let oldest = self.pointCloudNode.childNodes.first else { break }
                    oldest.geometry = nil
                    oldest.removeFromParentNode()
                }
            }

            let n = self.totalPointsSent
            let baseLabel = n >= 1_000_000
                ? String(format: "%.2fM pts", Double(n) / 1_000_000.0)
                : n >= 1000
                    ? String(format: "%.1fk pts", Double(n) / 1000.0)
                    : "\(n) pts"
            self.countLabel.text = baseLabel
        }
    }

    private func makePointGeo(
        pos: [Float3],
        col: [Float3],
        pointSize: CGFloat = 8,
        minimumPointScreenSpaceRadius: CGFloat = 2,
        maximumPointScreenSpaceRadius: CGFloat = 12
    ) -> SCNGeometry {
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
        element.pointSize                   = pointSize
        element.minimumPointScreenSpaceRadius = minimumPointScreenSpaceRadius
        element.maximumPointScreenSpaceRadius = maximumPointScreenSpaceRadius

        return SCNGeometry(sources: [posSrc, colSrc], elements: [element])
    }
}

// MARK: - ARSessionDelegate

extension ARScanViewController: ARSessionDelegate {

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let c = frame.camera.transform
        let curPos = SIMD3<Float>(c.columns.3.x, c.columns.3.y, c.columns.3.z)
        let curFwd = simd_normalize(SIMD3<Float>(-c.columns.2.x, -c.columns.2.y, -c.columns.2.z))

        var linearSpeed: Float = 0
        var angularSpeed: Float = 0
        if let lp = lastFramePos, let lf = lastFrameFwd, let ts = lastFrameTs {
            let dt = max(1e-3, Float(frame.timestamp - ts))
            linearSpeed = simd_length(curPos - lp) / dt
            let dotv = max(-1.0, min(1.0, simd_dot(curFwd, lf)))
            angularSpeed = acos(dotv) / dt
        }

        frameIndex += 1
        guard frameIndex % SAMPLE_EVERY == 0 else {
            lastFramePos = curPos
            lastFrameFwd = curFwd
            lastFrameTs = frame.timestamp
            return
        }

        // Ignore depth capture during very rapid motion where pose uncertainty spikes.
        if linearSpeed > CAPTURE_MAX_LINEAR_SPEED || angularSpeed > CAPTURE_MAX_ANGULAR_SPEED {
            lastFramePos = curPos
            lastFrameFwd = curFwd
            lastFrameTs = frame.timestamp
            return
        }

        // During the stabilization window, update the progress bar and skip depth capture.
        // The timer starts on the FIRST frame so the bar always starts moving immediately
        // (avoids the 0.5–2 s black "stuck at 0%" phase that occurred when the timer was
        // started before ARKit delivered its first frame).
        let now = Date()
        if !hasReceivedFirstFrame {
            hasReceivedFirstFrame = true
            stabilizeUntil = now.addingTimeInterval(STABILIZE_SECS)
        }

        if now < stabilizeUntil {
            let elapsed  = STABILIZE_SECS - stabilizeUntil.timeIntervalSince(now)
            let fraction = CGFloat(max(0, min(1, elapsed / STABILIZE_SECS)))
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                let trackW = self.stabilizeBar.bounds.width
                // Update the width constraint for the fill bar
                if let wc = self.stabilizeBarFill.constraints.first(where: { $0.firstAttribute == .width }) {
                    wc.constant = trackW * fraction
                    self.stabilizeBarFill.superview?.setNeedsLayout()
                }
            }
            lastFramePos = curPos; lastFrameFwd = curFwd; lastFrameTs = frame.timestamp
            return
        }

        // First frame after stabilization — dismiss the overlay and start warm-up
        if let overlay = view.viewWithTag(9001), !overlay.isHidden {
            warmupFramesRemaining = WARMUP_FRAMES
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                UIView.animate(withDuration: 0.35) {
                    self.view.viewWithTag(9001)?.alpha = 0
                } completion: { _ in
                    self.view.viewWithTag(9001)?.isHidden = true
                }
            }
        }

        // Warm-up: run processFrame during the first WARMUP_FRAMES so the Metal pipeline
        // (which is JIT-compiled on the first SceneKit geometry upload) is hot before the
        // real scan begins — this eliminates the GPU stutter/jumpiness at scan start.
        // The accumulated warm-up points are cleared when warm-up ends so the final cloud
        // contains only stable post-warm-up depth data.
        if warmupFramesRemaining > 0 {
            warmupFramesRemaining -= 1
            let wBudget = samplesForMotion(linearSpeed: linearSpeed, angularSpeed: angularSpeed)
            processFrame(frame, samplesPerFrame: wBudget)
            sampledFrames += 1
            if sampledFrames >= BATCH_FLUSH { sampledFrames = 0; flushBatchToScene() }

            if warmupFramesRemaining == 0 {
                // Flush any remaining batch, then discard all warm-up data on the main thread.
                // The DispatchQueue ordering guarantees the clear runs AFTER the last flush.
                flushBatchToScene(); sampledFrames = 0
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    self.accumulatedCloud.removeAll(keepingCapacity: true)
                    self.totalPointsSent = 0
                    for node in self.pointCloudNode.childNodes {
                        node.geometry = nil
                        node.removeFromParentNode()
                    }
                }
            }
            lastFramePos = curPos; lastFrameFwd = curFwd; lastFrameTs = frame.timestamp
            return
        }

        let sampleBudget = samplesForMotion(linearSpeed: linearSpeed, angularSpeed: angularSpeed)

        processFrame(frame, samplesPerFrame: sampleBudget)
        maybeCapture(frame: frame, currentPos: curPos)

        sampledFrames += 1
        if sampledFrames >= BATCH_FLUSH {
            sampledFrames = 0
            flushBatchToScene()
        }

        lastFramePos = curPos
        lastFrameFwd = curFwd
        lastFrameTs = frame.timestamp
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        let msg = error.localizedDescription
        print("[ARKit] session didFailWithError: \(msg)")
        DispatchQueue.main.async { [weak self] in
            self?.countLabel.text = "Error"
        }
        ARBridge.shared.sendError("ARKit session failed: \(msg)")
        dismiss(animated: true) { [weak self] in self?.onComplete?([]) }
    }

    /// ARSession interrupted (e.g. incoming call, system overlay).
    /// We do NOT dismiss — sessionInterruptionEnded will resume scanning.
    func sessionWasInterrupted(_ session: ARSession) {
        print("[ARKit] session interrupted")
        DispatchQueue.main.async { [weak self] in
            self?.countLabel.text = "Paused…"
        }
    }

    /// Resume the session without resetting tracking so the accumulated
    /// point cloud and world coordinate frame are preserved.
    func sessionInterruptionEnded(_ session: ARSession) {
        print("[ARKit] session interruption ended — resuming")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            let config = ARWorldTrackingConfiguration()
            if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
                config.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
            } else {
                config.frameSemantics = [.sceneDepth]
            }
            session.run(config)
        }
    }

    // MARK: - Snapshot capture

    /// Capture one JPEG snapshot + camera matrices if the camera has moved
    /// at least SNAPSHOT_MIN_MOVE metres since the last snapshot.
    ///
    /// Uses ARKit's frame.capturedImage (main wide-angle camera).
    /// fw/fh are the full sensor dimensions so the web formula K[0]/fw
    /// correctly normalises to [0,1] UV regardless of JPEG scale.
    /// JPEG encode runs on a background thread to avoid stalling the AR loop.
    private func maybeCapture(frame: ARFrame, currentPos: SIMD3<Float>) {
        guard !isFinishingScan else { return }
        guard snapshotIndex < SNAPSHOT_MAX_COUNT else { return }
        snapshotFrameCounter += 1
        guard snapshotFrameCounter >= SNAPSHOT_INTERVAL_FRAMES else { return }

        if let lastPos = lastSnapshotPos,
           simd_distance(currentPos, lastPos) < SNAPSHOT_MIN_MOVE { return }

        snapshotFrameCounter = 0
        lastSnapshotPos = currentPos

        let pixBuf   = frame.capturedImage
        let res      = frame.camera.imageResolution
        let fullW    = Int(res.width)
        let fullH    = Int(res.height)
        let kMatrix  = frame.camera.intrinsics
        let transform = frame.camera.transform
        pendingSnapshotEncodes += 1

        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self else { return }
            defer {
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    self.pendingSnapshotEncodes = max(0, self.pendingSnapshotEncodes - 1)
                    self.finishScanIfReady()
                }
            }

            // ── Snapshot encode ──────────────────────────────────────────────
            // Target 1024 px on the longest side — matches the web SNAP_TEX_MAX_PX cap
            // so no further downscaling is applied on upload.
            let targetMaxDim: CGFloat = 1024
            let srcMax   = CGFloat(max(fullW, fullH))
            let iosScale = srcMax > targetMaxDim ? Double(targetMaxDim / srcMax) : 1.0

            // Use the shared CIContext — creating a new one per snapshot allocated a
            // fresh Metal render pipeline each time; with up to 48 snapshots this
            // exhausted Metal resources and caused createCGImage to return nil,
            // leaving capturedSnapshots empty and producing a blank canvas.
            let ciImg = CIImage(cvPixelBuffer: pixBuf)
            let scaled = iosScale < 1.0
                ? ciImg.transformed(by: CGAffineTransform(scaleX: iosScale, y: iosScale))
                : ciImg
            guard let cgImg = snapshotCIContext.createCGImage(scaled, from: scaled.extent) else { return }
            let uiImg = UIImage(cgImage: cgImg)
            guard let jpegData = uiImg.jpegData(compressionQuality: 0.85) else { return }

            // Column-major 4×4 camera-to-world (ARKit storage order)
            let c2w: [Float] = [
                transform.columns.0.x, transform.columns.0.y, transform.columns.0.z, transform.columns.0.w,
                transform.columns.1.x, transform.columns.1.y, transform.columns.1.z, transform.columns.1.w,
                transform.columns.2.x, transform.columns.2.y, transform.columns.2.z, transform.columns.2.w,
                transform.columns.3.x, transform.columns.3.y, transform.columns.3.z, transform.columns.3.w,
            ]
            // Column-major 3×3 intrinsics — from UW sampleBuffer attachment or
            // ARKit frame.camera.intrinsics, whichever source provided the image.
            // Layout: [fx,0,0, 0,fy,0, cx,cy,1]  (K[0]=fx, K[4]=fy, K[6]=cx, K[7]=cy)
            let K: [Float] = [
                kMatrix.columns.0.x, kMatrix.columns.0.y, kMatrix.columns.0.z,
                kMatrix.columns.1.x, kMatrix.columns.1.y, kMatrix.columns.1.z,
                kMatrix.columns.2.x, kMatrix.columns.2.y, kMatrix.columns.2.z,
            ]

            // Upload snapshot directly to backend (fire-and-forget).
            ARBridge.shared.uploadSnapshotDirect(
                index: self.snapshotIndex,
                jpeg: jpegData, c2w: c2w, K: K, fw: fullW, fh: fullH
            )

            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.snapshotIndex += 1
                self.capturedSnapshots.append(PhotoProjector.SnapshotCapture(
                    jpegData: jpegData,
                    c2w: c2w,
                    K: K,
                    fw: fullW,
                    fh: fullH
                ))
            }
        }
    }
}

