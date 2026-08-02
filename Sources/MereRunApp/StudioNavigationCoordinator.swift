import Combine
import Foundation

struct StudioLibraryNavigationRequest: Equatable {
    let token = UUID()
    let itemID: UUID
    let mode: StudioMode
}

@MainActor
final class StudioNavigationCoordinator: ObservableObject {
    @Published private(set) var libraryRequest: StudioLibraryNavigationRequest?

    func openLibraryItem(id: UUID, mode: StudioMode) {
        libraryRequest = StudioLibraryNavigationRequest(itemID: id, mode: mode)
    }
}
