import SwiftUI
import UIKit

struct ContentView: View {
    @StateObject private var vm = InferenceViewModel()
    @State private var showModelSheet = false
    @State private var showSplash = true

    var body: some View {
        ZStack {
            // Main app UI
            VStack(spacing: 0) {
                DeviceInfoBar(vm: vm, showModelSheet: $showModelSheet)
                Divider()
                ChatView(vm: vm)
            }
            .sheet(isPresented: $showModelSheet) {
                ModelLoaderSheet(vm: vm)
                    .presentationDetents([.medium, .large])
            }

            // Splash overlay — covers the black launch screen transition
            if showSplash {
                splashOverlay
                    .transition(.opacity)
                    .zIndex(1)
            }
        }
        .preferredColorScheme(.dark)
        .onChange(of: vm.isReady) { ready in
            if ready {
                withAnimation(.easeOut(duration: 0.4)) {
                    showSplash = false
                }
                if !vm.isModelLoaded {
                    showModelSheet = true
                }
            }
        }
        .onAppear {
            prewarmKeyboard()
        }
    }

    // MARK: - Splash screen

    private var splashOverlay: some View {
        VStack(spacing: 16) {
            Spacer()

            Image(systemName: "cpu")
                .font(.system(size: 48, weight: .thin))
                .foregroundColor(.white.opacity(0.8))

            Text("CrabInfer")
                .font(.title2.weight(.semibold))
                .foregroundColor(.white)

            ProgressView()
                .tint(.white.opacity(0.7))
                .scaleEffect(1.1)

            Text("Initializing...")
                .font(.caption)
                .foregroundColor(.white.opacity(0.5))

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black)
    }

    // MARK: - Keyboard pre-warm

    private func prewarmKeyboard() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            guard let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                  let window = windowScene.windows.first else { return }

            let field = UITextField(frame: CGRect(x: -1000, y: -1000, width: 1, height: 1))
            field.autocorrectionType = .no
            window.addSubview(field)
            field.becomeFirstResponder()

            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                field.resignFirstResponder()
                field.removeFromSuperview()
            }
        }
    }
}
