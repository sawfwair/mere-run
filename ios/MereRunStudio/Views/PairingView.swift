import SwiftUI

/// First-run pairing: relay URL in, device-grant approval out. The
/// verification code is shown large so it can be typed on another device;
/// opening the link on this phone also works.
struct PairingView: View {
    @EnvironmentObject private var relay: RelayStore
    @State private var urlString = ""

    var body: some View {
        VStack(spacing: MereTheme.Spacing.xl) {
            Spacer()
            Text("Create anything.\nFrom anywhere.")
                .font(.system(.largeTitle, design: .serif))
                .multilineTextAlignment(.center)
                .foregroundStyle(MereTheme.textPrimary)
            Text("Pair with your relay to run work on your own machines.")
                .font(.callout)
                .foregroundStyle(MereTheme.textSecondary)
                .multilineTextAlignment(.center)

            switch relay.pairing {
            case .unpaired, .failed, .discovering:
                VStack(spacing: MereTheme.Spacing.m) {
                    TextField(
                        "",
                        text: $urlString,
                        prompt: Text("Your relay address").foregroundColor(MereTheme.textMuted)
                    )
                    .foregroundStyle(MereTheme.textPrimary)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                        .padding(MereTheme.Spacing.m)
                        .merePanel()
                    if case .failed(let message) = relay.pairing {
                        Text(message)
                            .font(.footnote)
                            .foregroundStyle(MereTheme.failure)
                            .multilineTextAlignment(.center)
                    }
                    Button {
                        let url = urlString
                        Task { await relay.pair(urlString: url) }
                    } label: {
                        if relay.pairing == .discovering {
                            ProgressView().frame(maxWidth: .infinity)
                        } else {
                            Text("Pair").frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(urlString.isEmpty || relay.pairing == .discovering)
                }
            case .awaitingApproval(let verificationURL, let userCode):
                VStack(spacing: MereTheme.Spacing.m) {
                    Text("Approve this phone")
                        .font(.headline)
                        .foregroundStyle(MereTheme.textPrimary)
                    Text(userCode)
                        .font(.system(.largeTitle, design: .monospaced))
                        .foregroundStyle(MereTheme.accent)
                        .textSelection(.enabled)
                    if let url = URL(string: verificationURL) {
                        Link("Open \(verificationURL)", destination: url)
                            .font(.callout)
                    }
                    ProgressView("Waiting for approval…")
                        .font(.footnote)
                        .foregroundStyle(MereTheme.textMuted)
                }
                .padding(MereTheme.Spacing.xl)
                .merePanel()
            case .paired:
                EmptyView()
            }
            Spacer()
            Text("Stays between your devices.")
                .font(.footnote)
                .foregroundStyle(MereTheme.textMuted)
        }
        .padding(MereTheme.Spacing.xxl)
        .background(MereTheme.background.ignoresSafeArea())
    }
}
