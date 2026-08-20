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
    @State private var directConnect = false
    @State private var urlString = ""
    @State private var pairCode = ""

    private var pairingURL: String {
        customRelay || directConnect ? urlString : Self.defaultRelayURL
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
                    if customRelay || directConnect {
                        TextField(
                            "",
                            text: $urlString,
                            prompt: Text(directConnect ? "Machine address, like lab.local:6373" : "Your relay address")
                                .foregroundColor(MereTheme.textMuted)
                        )
                        .foregroundStyle(MereTheme.textPrimary)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                        .padding(MereTheme.Spacing.m)
                        .merePanel()
                        .accessibilityIdentifier("pairing.address")
                    }
                    if directConnect {
                        TextField(
                            "",
                            text: $pairCode,
                            prompt: Text("Pairing code from the terminal").foregroundColor(MereTheme.textMuted)
                        )
                        .foregroundStyle(MereTheme.textPrimary)
                        .keyboardType(.numbersAndPunctuation)
                        .padding(MereTheme.Spacing.m)
                        .merePanel()
                        .accessibilityIdentifier("pairing.code")
                    }
                    if case .failed(let message) = relay.pairing {
                        Text(message)
                            .font(.footnote)
                            .foregroundStyle(MereTheme.failure)
                            .multilineTextAlignment(.center)
                    }
                    Button {
                        let url = pairingURL
                        if directConnect {
                            let code = pairCode
                            Task { await relay.pairDirect(urlString: url, code: code) }
                        } else {
                            Task { await relay.signIn(urlString: url) }
                        }
                    } label: {
                        if relay.pairing == .discovering {
                            ProgressView().frame(maxWidth: .infinity)
                        } else {
                            Text(directConnect
                                ? "Connect"
                                : (customRelay ? "Sign in" : "Sign in with mere.world"))
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .accessibilityIdentifier("pairing.primary")
                    .disabled(
                        pairingURL.isEmpty
                            || relay.pairing == .discovering
                            || (directConnect && pairCode.isEmpty)
                    )
                    HStack(spacing: MereTheme.Spacing.l) {
                        Button(customRelay ? "Use relay.mere.run" : "Use a different relay") {
                            customRelay.toggle()
                            directConnect = false
                        }
                        Button(directConnect ? "Use a hosted relay" : "Connect to a machine") {
                            directConnect.toggle()
                            customRelay = false
                        }
                        if !directConnect {
                            Button("Use a device code") {
                                let url = pairingURL
                                Task { await relay.pair(urlString: url) }
                            }
                        }
                    }
                    .font(.footnote)
                    .foregroundStyle(MereTheme.textMuted)
                    .disabled(relay.pairing == .discovering)
                    if directConnect {
                        Text("Run `mere.run relay serve` on your machine, then enter its address and the code it prints. Works over your network or Tailscale.")
                            .font(.caption)
                            .foregroundStyle(MereTheme.textMuted)
                            .multilineTextAlignment(.center)
                    }
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
            Text(directConnect
                ? "Connects directly to your machine over your network or tailnet."
                : "Hosted relay infrastructure carries prompts, status, and artifacts to your fleet.")
                .font(.footnote)
                .foregroundStyle(MereTheme.textMuted)
        }
        .padding(MereTheme.Spacing.xxl)
        .background(MereTheme.background.ignoresSafeArea())
    }
}
