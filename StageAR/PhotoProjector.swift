/**
 * PhotoProjector.swift
 *
 * On-device photo-to-point-cloud color projection.
 *
 * Replaces the low-res LiDAR depth-sensor colors in the raw point cloud with
 * accurate bilinear samples from the high-res iPhone camera snapshots.
 *
 * SCORING FORMULA  (the core idea)
 * ---------------------------------
 * For each raw point P and each snapshot i, the "pixel density score" is:
 *
 *       score = fx · cos(θ) / r²
 *
 * where θ = angle between the camera's optical axis and the camera→P ray,
 *       r = distance from camera position to P,
 *       fx = focal length in pixels.
 *
 * This equals the camera's sampling density at P in pixels per metre — the
 * physically correct optimum for "which photo has the most detail here."
 * Equivalently, it is the solid angle subtended by one pixel at point P,
 * which is what you maximise when selecting the best source frame.
 *
 * In camera space, cos(θ) = depth / r (ARKit's −Z convention), so:
 *
 *       score = fx · depth / r²    (computed entirely from camera-space coords)
 *
 * COORDINATE CONVENTIONS
 * ----------------------
 * ARKit camera space: X right, Y up, Z toward viewer (camera looks along −Z).
 * Image pixel space:  X right, Y downward (origin at top-left).
 * Therefore the correct projection from camera to image is:
 *
 *       u =  fx * x_cam / depth + cx
 *       v =  fy * (-y_cam) / depth + cy    ← Y must be negated
 *
 * ARKit capturedImage is ALWAYS in sensor-native landscape orientation
 * (e.g. 4032×3024 or 1008×756 after scaling). CIImage/CVPixelBuffer never
 * applies rotation. No orientation detection is needed or appropriate.
 *
 * VISIBILITY
 * ----------
 * A per-snapshot z-buffer built from the raw point cloud gates the projection:
 * a point's photo sample is only accepted if its reprojected depth is within
 * a tolerance of the shallowest observed depth at that pixel, preventing
 * back-wall bleeding through foreground geometry.
 */

import Foundation
import UIKit
import simd

enum PhotoProjector {

    // MARK: - Input type

    struct SnapshotCapture {
        let jpegData: Data
        /// Column-major 4×4 ARKit camera-to-world transform (16 floats).
        let c2w: [Float]
        /// Column-major 3×3 ARKit camera intrinsics in full-resolution pixel coords (9 floats).
        let K: [Float]
        let fw: Int   // original full-resolution capture width  (K reference frame)
        let fh: Int   // original full-resolution capture height
    }

    // MARK: - Internal prepared snapshot

    private struct PrepSnap {
        let pixels: [UInt8]          // RGBA, row-major, W×H×4 bytes
        let W: Int
        let H: Int
        let zW: Int                  // z-buffer width  = ⌈W / Z_SCALE⌉
        let zH: Int                  // z-buffer height = ⌈H / Z_SCALE⌉
        let w2c: simd_float4x4       // world-to-camera
        let fx, fy, cx, cy: Float    // intrinsics in full-res pixel coords
        // Precomputed scale from full-res intrinsic frame to decoded JPEG pixels.
        // ARKit capturedImage is always landscape-native; no rotation needed.
        let scaleX: Float            // = Float(W) / Float(fw)
        let scaleY: Float            // = Float(H) / Float(fh)
        var zbuf: [Float] = []       // shallowest depth per z-buffer cell, filled after build
    }

    // Depth tolerance for z-buffer gating: absolute floor + fraction of reference depth.
    private static let Z_SCALE: Float    = 2.0
    private static let TOL_ABS: Float    = 0.08   // metres absolute floor
    private static let TOL_REL: Float    = 0.03   // fraction of z_ref

    // MARK: - Public API

    /// Project snapshot colors onto the raw LiDAR point cloud, in place.
    ///
    /// Points not visible in any snapshot retain their original LiDAR color.
    ///
    /// - Parameters:
    ///   - cloud: interleaved [x,y,z,r,g,b …] Float32; length = pointCount × 6
    ///   - pointCount: number of 3-D points in `cloud`
    ///   - captures: snapshot bundles captured by ARScanViewController
    ///   - progress: optional 0→1 progress callback (called on the calling thread)
    static func project(
        cloud: inout [Float],
        pointCount: Int,
        captures: [SnapshotCapture],
        progress: ((Float) -> Void)? = nil
    ) {
        guard pointCount > 0, !captures.isEmpty,
              cloud.count >= pointCount * 6 else { return }

        // 1. Decode all snapshot JPEGs to RGBA pixel arrays + parse camera math.
        progress?(0.05)
        var snaps = captures.compactMap { decode($0) }
        guard !snaps.isEmpty else { return }
        progress?(0.15)

        // 2. Build a strided XYZ subset (≤ 350 K points) for z-buffer construction.
        let subCount = min(pointCount, 350_000)
        let stride   = max(1, pointCount / subCount)
        var subXYZ   = [SIMD3<Float>]()
        subXYZ.reserveCapacity(subCount)
        var idx = 0
        while idx < pointCount {
            let b = idx * 6
            subXYZ.append(SIMD3<Float>(cloud[b], cloud[b+1], cloud[b+2]))
            idx += stride
        }

        // 3. Per snapshot: build the z-buffer.
        let nSnaps = Float(snaps.count)
        for si in 0..<snaps.count {
            snaps[si].zbuf = buildZBuffer(snap: snaps[si], sub: subXYZ)
            progress?(0.15 + 0.30 * Float(si + 1) / nSnaps)
        }
        progress?(0.45)

        // 4. Main projection: outer loop over snapshots (keeps the z-buffer warm
        //    in cache), inner loop over all points.
        var bestScore = [Float](repeating: -.infinity, count: pointCount)

        for si in 0..<snaps.count {
            let snap = snaps[si]
            let Wf   = Float(snap.W)
            let Hf   = Float(snap.H)

            for ptIdx in 0..<pointCount {
                let b  = ptIdx * 6
                let pc = snap.w2c * SIMD4<Float>(cloud[b], cloud[b+1], cloud[b+2], 1.0)

                // ARKit camera looks along −Z: point must have negative z to be in front.
                let zc = pc.z
                guard zc < -0.05 else { continue }
                let depth = -zc    // positive depth along optical axis

                // Project to full-resolution camera pixel coordinates.
                // Y is negated because camera-space Y is up, image Y is down.
                let u =  snap.fx * pc.x         / depth + snap.cx
                let v =  snap.fy * (-pc.y)      / depth + snap.cy

                // Scale to decoded JPEG pixel coordinates.
                let pxf = u * snap.scaleX
                let pyf = v * snap.scaleY

                guard pxf >= 0.0, pxf < Wf - 1.0,
                      pyf >= 0.0, pyf < Hf - 1.0 else { continue }

                // Z-buffer occlusion gate.
                let zx   = max(0, min(snap.zW - 1, Int(pxf / Z_SCALE + 0.5)))
                let zy   = max(0, min(snap.zH - 1, Int(pyf / Z_SCALE + 0.5)))
                let zRef = snap.zbuf[zy * snap.zW + zx]
                guard zRef.isFinite,
                      depth <= zRef + max(TOL_ABS, TOL_REL * zRef) else { continue }

                // Pixel-density score: fx · cos(θ) / r²
                // cos(θ) = depth / r  →  score = fx · depth / r²
                let r2    = pc.x*pc.x + pc.y*pc.y + pc.z*pc.z
                let score = snap.fx * depth / (r2 + 1e-6)
                guard score > bestScore[ptIdx] else { continue }

                bestScore[ptIdx] = score
                let col = bilinear(snap: snap, px: pxf, py: pyf)
                cloud[b+3] = col.x
                cloud[b+4] = col.y
                cloud[b+5] = col.z
            }

            progress?(0.45 + 0.50 * Float(si + 1) / nSnaps)
        }

        progress?(1.0)
    }

    // MARK: - Snapshot decoding

    private static func decode(_ cap: SnapshotCapture) -> PrepSnap? {
        guard cap.K.count == 9, cap.c2w.count == 16,
              cap.fw > 1, cap.fh > 1 else { return nil }

        guard let uiImg = UIImage(data: cap.jpegData),
              let cgImg = uiImg.cgImage else { return nil }
        let W = cgImg.width
        let H = cgImg.height
        var pixels = [UInt8](repeating: 0, count: W * H * 4)
        guard let ctx = CGContext(
            data: &pixels, width: W, height: H,
            bitsPerComponent: 8, bytesPerRow: W * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        ctx.draw(cgImg, in: CGRect(x: 0, y: 0, width: W, height: H))

        // Rebuild simd_float4x4 from the column-major flat array ARKit produced.
        let c   = cap.c2w
        let c2w = simd_float4x4(columns: (
            SIMD4<Float>(c[0],  c[1],  c[2],  c[3] ),
            SIMD4<Float>(c[4],  c[5],  c[6],  c[7] ),
            SIMD4<Float>(c[8],  c[9],  c[10], c[11]),
            SIMD4<Float>(c[12], c[13], c[14], c[15])
        ))
        guard abs(c2w.determinant) > 1e-6 else { return nil }

        // ARKit K column-major 3×3: col0=(fx,0,0)  col1=(0,fy,0)  col2=(cx,cy,1)
        // Flat: [fx,0,0, 0,fy,0, cx,cy,1]  →  K[0]=fx  K[4]=fy  K[6]=cx  K[7]=cy
        let k  = cap.K
        let fx = k[0];  let fy = k[4]
        let cx = k[6];  let cy = k[7]

        let zW = max(1, Int(ceil(Float(W) / Z_SCALE)))
        let zH = max(1, Int(ceil(Float(H) / Z_SCALE)))

        return PrepSnap(pixels: pixels, W: W, H: H, zW: zW, zH: zH,
                        w2c: c2w.inverse,
                        fx: fx, fy: fy, cx: cx, cy: cy,
                        scaleX: Float(W) / Float(cap.fw),
                        scaleY: Float(H) / Float(cap.fh))
    }

    // MARK: - Z-buffer construction

    private static func buildZBuffer(snap: PrepSnap, sub: [SIMD3<Float>]) -> [Float] {
        let Wf = Float(snap.W);  let Hf = Float(snap.H)
        var zbuf = [Float](repeating: .infinity, count: snap.zW * snap.zH)
        for p in sub {
            let pc = snap.w2c * SIMD4<Float>(p.x, p.y, p.z, 1.0)
            guard pc.z < -0.05 else { continue }
            let d   = -pc.z
            let u   =  snap.fx * pc.x    / d + snap.cx
            let v   =  snap.fy * (-pc.y) / d + snap.cy
            let pxf = u * snap.scaleX
            let pyf = v * snap.scaleY
            guard pxf >= 0 && pxf < Wf && pyf >= 0 && pyf < Hf else { continue }
            let zx  = max(0, min(snap.zW - 1, Int(pxf / Z_SCALE + 0.5)))
            let zy  = max(0, min(snap.zH - 1, Int(pyf / Z_SCALE + 0.5)))
            let idx = zy * snap.zW + zx
            if d < zbuf[idx] { zbuf[idx] = d }
        }
        return zbuf
    }

    // MARK: - Bilinear sampling

    @inline(__always)
    private static func bilinear(snap: PrepSnap, px: Float, py: Float) -> SIMD3<Float> {
        let x0 = Int(px);  let y0 = Int(py)
        let x1 = min(x0 + 1, snap.W - 1)
        let y1 = min(y0 + 1, snap.H - 1)
        let dx = px - Float(x0);  let dy = py - Float(y0)

        @inline(__always)
        func px3(_ x: Int, _ y: Int) -> SIMD3<Float> {
            let i = (y * snap.W + x) * 4
            return SIMD3<Float>(Float(snap.pixels[i])   / 255.0,
                                Float(snap.pixels[i+1]) / 255.0,
                                Float(snap.pixels[i+2]) / 255.0)
        }

        let w00: Float = (1 - dx) * (1 - dy)
        let w10: Float = dx       * (1 - dy)
        let w01: Float = (1 - dx) * dy
        let w11: Float = dx       * dy
        return px3(x0, y0) * w00 + px3(x1, y0) * w10
             + px3(x0, y1) * w01 + px3(x1, y1) * w11
    }
}
