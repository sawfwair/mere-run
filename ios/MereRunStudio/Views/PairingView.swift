import SwiftUI

/// First-run pairing. One tap pairs with the public relay; the device-grant
/// approval handles the account, so nobody types infrastructure by hand. A
/// quiet disclosure reveals the address field for self-hosted or development
/// relays. The verification code is shown large so it can be typed on another
/// device; opening the link on this phone also works.
struct PairingView: View {
    static let defaultRelayURL = "https://relay.mere.run"

    @EnvironmentObject private var relay: RelayStore
    @State private var customRelay = false
    @State private var urlString = ""

    private var pairingURL: String {
        customRelay ? urlString : Self.defaultRelayURL
    }

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
                    if customRelay {
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
                    }
                    if case .failed(let message) = relay.pairing {
                        Text(message)
                            .font(.footnote)
                            .foregroundStyle(MereTheme.failure)
                            .multilineTextAlignment(.center)
                    }
                    Button {
                        let url = pairingURL
                        Task { await relay.pair(urlString: url) }
                    } label: {
                        if relay.pairing == .discovering {
                            ProgressView().frame(maxWidth: .infinity)
                        } else {
                            Text(customRelay ? "Pair" : "Pair with relay.mere.run")
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(pairingURL.isEmpty || relay.pairing == .discovering)
                    Button(customRelay ? "Use relay.mere.run" : "Use a different relay") {
                        customRelay.toggle()
                    }
                    .font(.footnote)
                    .foregroundStyle(MereTheme.textMuted)
                    .disabled(relay.pairing == .discovering)
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
