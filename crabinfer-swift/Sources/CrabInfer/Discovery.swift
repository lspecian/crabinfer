/// CrabInfer Discovery — Bonjour/mDNS service discovery for self-hosted providers.
///
/// Uses Apple's Network framework (NWBrowser) to discover CrabInfer servers
/// and Ollama instances on the local network.
///
/// Usage:
/// ```swift
/// let discovery = CrabInfer.Discovery()
/// discovery.onFound = { endpoint in
///     print("Found: \(endpoint.name) at \(endpoint.host):\(endpoint.port)")
/// }
/// discovery.start()
/// // later...
/// discovery.stop()
/// ```

#if canImport(Network)
import Network
import Foundation

extension CrabInfer {

    /// A discovered CrabInfer or Ollama service endpoint.
    public struct DiscoveredEndpoint: Sendable, Identifiable, Hashable {
        /// Unique identifier for this endpoint.
        public let id: String
        /// Human-readable service name (e.g., "Mac Studio - llama3").
        public let name: String
        /// Hostname or IP address.
        public let host: String
        /// Port number.
        public let port: UInt16
        /// Service type that was discovered.
        public let serviceType: String
    }

    /// Discovers CrabInfer servers and Ollama instances on the local network via Bonjour.
    ///
    /// Browses for `_crabinfer._tcp` and `_ollama._tcp` service types.
    /// Results are delivered via the `onFound` and `onLost` callbacks.
    @available(iOS 13.0, macOS 10.15, *)
    public class Discovery {
        /// Called when a new service is discovered.
        public var onFound: ((DiscoveredEndpoint) -> Void)?
        /// Called when a previously discovered service disappears.
        public var onLost: ((DiscoveredEndpoint) -> Void)?

        private var browsers: [NWBrowser] = []
        private let queue = DispatchQueue(label: "com.crabinfer.discovery")
        private var running = false

        /// Service types to browse for.
        public static let serviceTypes = ["_crabinfer._tcp", "_ollama._tcp"]

        public init() {}

        /// Start browsing for services on the local network.
        public func start() {
            guard !running else { return }
            running = true

            for serviceType in Self.serviceTypes {
                let params = NWParameters()
                params.includePeerToPeer = true
                let browser = NWBrowser(for: .bonjour(type: serviceType, domain: "local."), using: params)

                browser.browseResultsChangedHandler = { [weak self] results, changes in
                    self?.handleChanges(changes, serviceType: serviceType)
                }

                browser.stateUpdateHandler = { state in
                    switch state {
                    case .failed(let error):
                        print("[CrabInfer Discovery] Browser failed for \(serviceType): \(error)")
                    default:
                        break
                    }
                }

                browser.start(queue: queue)
                browsers.append(browser)
            }
        }

        /// Stop browsing.
        public func stop() {
            guard running else { return }
            running = false
            for browser in browsers {
                browser.cancel()
            }
            browsers.removeAll()
        }

        private func handleChanges(_ changes: Set<NWBrowser.Result.Change>, serviceType: String) {
            for change in changes {
                switch change {
                case .added(let result):
                    if let endpoint = makeEndpoint(from: result, serviceType: serviceType) {
                        onFound?(endpoint)
                    }
                case .removed(let result):
                    if let endpoint = makeEndpoint(from: result, serviceType: serviceType) {
                        onLost?(endpoint)
                    }
                default:
                    break
                }
            }
        }

        private func makeEndpoint(from result: NWBrowser.Result, serviceType: String) -> DiscoveredEndpoint? {
            guard case .service(let name, let type, let domain, _) = result.endpoint else {
                return nil
            }
            // NWBrowser provides the service name; actual host:port requires NWConnection resolution.
            // For now, provide the Bonjour name — callers can resolve via NWConnection.
            return DiscoveredEndpoint(
                id: "\(type).\(domain).\(name)",
                name: name,
                host: "\(name).\(domain)",
                port: 0, // Port is resolved when connecting
                serviceType: serviceType
            )
        }

        deinit {
            stop()
        }
    }
}
#endif
