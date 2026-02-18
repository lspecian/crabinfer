// InferenceThread.swift
// CrabInferDemo
//
// A dedicated OS thread with a persistent RunLoop for Metal inference.
//
// WHY THIS EXISTS:
//
// Candle's Metal backend compiles GPU shaders lazily on the first forward() call.
// On iOS, shader compilation crosses an XPC boundary to com.apple.MTLCompilerService.
// This can deadlock if the calling thread can't process Metal's internal callbacks.
//
// Neither Task.detached (cooperative thread pool) nor plain DispatchQueue (GCD worker
// threads with no persistent RunLoop) provide the right environment. We need:
//
//  1. A REAL OS thread (not recycled by GCD between blocks)
//  2. A LIVE RunLoop on that thread (Metal may use CFRunLoop for internal callbacks)
//  3. All Metal work pinned to this one thread (Metal device affinity)
//
// This class creates a Thread, adds a dummy port to keep its RunLoop alive,
// and uses CFRunLoopPerformBlock to schedule work on that thread's RunLoop.

import Foundation

final class InferenceThread: @unchecked Sendable {

    private var _runLoop: CFRunLoop!

    init() {
        let ready = DispatchSemaphore(value: 0)

        let thread = Thread {
            // Add a dummy port source so the RunLoop stays alive indefinitely.
            // Without this, RunLoop.current.run() returns immediately because
            // there are no input sources.
            RunLoop.current.add(NSMachPort(), forMode: .default)

            self._runLoop = CFRunLoopGetCurrent()
            ready.signal()

            // Block this thread forever, processing RunLoop events.
            // Metal callbacks and our CFRunLoopPerformBlock work items
            // will be processed here.
            RunLoop.current.run()
        }
        thread.name = "CrabInfer.MetalInference"
        thread.qualityOfService = .userInitiated
        thread.start()

        ready.wait()
    }

    // MARK: - Public API

    /// Await a throwing result produced on the inference thread.
    func perform<T: Sendable>(
        _ block: @escaping @Sendable () throws -> T
    ) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            CFRunLoopPerformBlock(_runLoop, CFRunLoopMode.defaultMode.rawValue) {
                do {
                    let result = try block()
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
            CFRunLoopWakeUp(_runLoop)
        }
    }

    /// Build an AsyncStream whose values are produced on the inference thread.
    func stream<T: Sendable>(
        _ producer: @escaping @Sendable (_ yield: (T) -> Bool) -> Void
    ) -> AsyncStream<T> {
        AsyncStream { continuation in
            CFRunLoopPerformBlock(self._runLoop, CFRunLoopMode.defaultMode.rawValue) {
                producer { value in
                    switch continuation.yield(value) {
                    case .enqueued, .dropped: return true
                    case .terminated:         return false
                    @unknown default:         return false
                    }
                }
                continuation.finish()
            }
            CFRunLoopWakeUp(self._runLoop)
        }
    }
}
