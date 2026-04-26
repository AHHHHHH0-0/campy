import Foundation

/// Every user-facing string. Localization is a future lift, not a refactor.
enum UIStrings {
    static let appTitle = "Campy"
    static let welcomeWordmark = "Welcome to Campy"
    static let tapAndHoldToSpeak = "tap and hold to speak"
    static let noNetworkHoldMessage = "No network connection"
    static let poweredByPrefix = "Powered by "
    static let betaDisclaimer = "Product in Beta testing, models could make mistakes. Use at your own risk."

    // Tabs
    static let identifyTabLabel = "Identify"
    static let prepareTabLabel = "Prepare"

    // Permission required screens
    static let cameraRequiredCopy = "Camera access required"
    static let micRequiredCopy = "Microphone access required"

    // Detail sheet
    static let notInLocalDatabase = "not in local database"
    static let dismissHint = "tap anywhere to dismiss"
    static let plantUnsureRetry = "Not sure this is a plant. Try tapping again with a closer view."

    // Error / quota strings
    static let quotaReply = "Model limit reached"

    // Triple-tap clear flow
    static let clearConfirmTitle = "Delete conversation?"
    static let clearConfirmMessage = "This wipes your transcript, all region packs, and cached narratives."
    static let clearConfirmYes = "Yes"
    static let clearConfirmNo = "No"

    // Asset names
    static let transcriptDividerAsset = "TranscriptDivider"
    static let poweredByZeticAsset = "PoweredByZetic"
}
