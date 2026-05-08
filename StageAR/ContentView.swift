import SwiftUI
import WebKit

struct ContentView: View {
    var body: some View {
        StageWebView()
            .ignoresSafeArea()
    }
}

// MARK: - WKWebView wrapper

struct StageWebView: UIViewRepresentable {

    static let appURL = URL(string: "https://mwegter95.github.io/gallery-wall-planner/")!

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()

        // Tell the web app it is running inside the native wrapper.
        // Checked via `window.__stageARNative` in LidarScanner.jsx.
        let flagScript = WKUserScript(
            source: "window.__stageARNative = true;",
            injectionTime: .atDocumentStart,
            forMainFrameOnly: true
        )
        config.userContentController.addUserScript(flagScript)

        // Register the bridge. The web app calls:
        //   window.webkit.messageHandlers.stageAR.postMessage({ action: '...' })
        config.userContentController.add(ARBridge.shared, name: "stageAR")

        config.allowsInlineMediaPlayback = true
        config.mediaTypesRequiringUserActionForPlayback = []

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.scrollView.bounces = false
        webView.scrollView.isScrollEnabled = true
        // Makes the page inspectable via Safari → Develop menu (great for debugging)
        if #available(iOS 16.4, *) {
            webView.isInspectable = true
        }

        ARBridge.shared.webView = webView

        webView.load(URLRequest(url: Self.appURL,
                               cachePolicy: .reloadRevalidatingCacheData,
                               timeoutInterval: 30))
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {}
}
