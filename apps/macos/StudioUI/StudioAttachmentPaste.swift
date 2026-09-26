import AppKit
import StudioKit
import SwiftUI

// ⌘V into an attachment: the composer's prompt, a well, or the canvas. Every entry point reads the
// pasteboard through `StudioAttachmentPasteboard` and attaches through the drop path, so pasting
// a file behaves exactly like dropping it on the same place.

@MainActor
enum StudioAttachmentPaste {
    /// Pastes the general pasteboard into `slots` of `draft`: copied files, or a copied picture or
    /// sound written under the app's support folder, go to the slot a drop of them would pick.
    /// When `allowsText`, anything else that carries words (text, a Finder copy's file name) is
    /// left to the focused text field. What remains is refused with the system's beep, and
    /// nothing is written. Returns whether the paste was handled here; false means the text field
    /// pastes the words.
    @discardableResult
    static func paste<Draft: StudioAttachmentDraft>(
        into draft: inout Draft,
        slots: [StudioAttachmentSlot],
        allowsText: Bool
    ) -> Bool {
        let pasteboard = NSPasteboard.general
        let content = StudioAttachmentPasteboard.content(of: pasteboard)
        let intake: StudioPasteIntake
        do {
            intake = try StudioAttachmentPasteboard.intake(content) { url in slots.contains { $0.accepts(url) } }
        } catch {
            NSSound.beep()
            return true
        }
        switch intake {
        case .attach(let urls):
            draft.attach(dropped: urls, slots: slots)
            return true
        case .text, .refused:
            if allowsText, pasteboard.types?.contains(.string) == true { return false }
            NSSound.beep()
            return true
        }
    }
}

extension View {
    /// Runs `handler` for ⌘V while `isActive` — the composer's prompt has focus — before the
    /// text field pastes. A SwiftUI text field pastes through its own field editor, so a paste
    /// command handler on it never runs; the key monitor reaches the keystroke first. `handler`
    /// returns false to let the text field paste the words.
    func studioAttachmentPasteKey(isActive: Bool, handler: @escaping () -> Bool) -> some View {
        modifier(StudioAttachmentPasteKey(isActive: isActive, handler: handler))
    }
}

private struct StudioAttachmentPasteKey: ViewModifier {
    let isActive: Bool
    let handler: () -> Bool

    @State private var monitor = Monitor()

    /// The installed key monitor, the latest handler (so the draft it pastes into is current),
    /// and the window the view is in (so ⌘V in another Studio window is left alone).
    @MainActor
    final class Monitor {
        var handler: () -> Bool = { false }
        weak var window: NSWindow?
        /// Removes the installed monitor; nil while none is installed. AppKit's monitor token is
        /// opaque, so only this closure holds it.
        private var removeMonitor: (() -> Void)?

        func install() {
            guard removeMonitor == nil,
                  let token = NSEvent.addLocalMonitorForEvents(matching: .keyDown, handler: { [weak self] event in
                      guard let self,
                            event.modifierFlags.intersection([.command, .shift, .option, .control]) == .command,
                            Self.isPasteKey(event),
                            let window = self.window, event.window === window,
                            self.handler() else { return event }
                      return nil
                  }) else { return }
            removeMonitor = { NSEvent.removeMonitor(token) }
        }

        func remove() {
            removeMonitor?()
            removeMonitor = nil
        }

        /// The V key: by its character, or by its position when the layout types no Latin letter
        /// there (Cyrillic, Greek), which is how the Edit menu's ⌘V matches too.
        private static func isPasteKey(_ event: NSEvent) -> Bool {
            guard let key = event.charactersIgnoringModifiers?.lowercased() else { return false }
            return key == "v" || (!key.allSatisfy(\.isASCII) && event.keyCode == vKeyCode)
        }

        /// The V key's virtual key code on an ANSI keyboard, as `NSEvent.keyCode` reports it.
        private static let vKeyCode: UInt16 = 9
    }

    func body(content: Content) -> some View {
        monitor.handler = handler
        return content
            .background(StudioWindowReader { monitor.window = $0 })
            .onChange(of: isActive, initial: true) { _, active in
                if active { monitor.install() } else { monitor.remove() }
            }
            .onDisappear { monitor.remove() }
    }
}

/// Reports the window its view is placed in.
struct StudioWindowReader: NSViewRepresentable {
    let onWindow: (NSWindow?) -> Void

    func makeNSView(context: Context) -> WindowView {
        let view = WindowView()
        view.onWindow = onWindow
        return view
    }

    func updateNSView(_ nsView: WindowView, context: Context) {
        nsView.onWindow = onWindow
    }

    final class WindowView: NSView {
        var onWindow: ((NSWindow?) -> Void)?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            onWindow?(window)
        }
    }
}
