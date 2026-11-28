import MereRunEnvironment
import XCTest

final class MereRunEnvironmentSwitchTests: XCTestCase {
    func testParsesTheSharedVocabularyIgnoringCaseAndWhitespace() {
        for raw in ["1", "true", "YES", " On\n"] {
            XCTAssertEqual(MereRunEnvironmentSwitch.parse(raw), true, raw)
        }
        for raw in ["0", "False", "no", "OFF "] {
            XCTAssertEqual(MereRunEnvironmentSwitch.parse(raw), false, raw)
        }
        for raw in [nil, "", "  ", "2", "enabled", "maybe"] {
            XCTAssertNil(MereRunEnvironmentSwitch.parse(raw), String(describing: raw))
        }
    }

    func testUnsetOrUnrecognizedValuesKeepTheDefault() {
        let environment = ["MERERUN_ON": "true", "MERERUN_OFF": "off", "MERERUN_BAD": "sometimes"]
        XCTAssertTrue(MereRunEnvironmentSwitch.isEnabled("MERERUN_ON", default: false, environment: environment))
        XCTAssertFalse(MereRunEnvironmentSwitch.isEnabled("MERERUN_OFF", default: true, environment: environment))
        XCTAssertTrue(MereRunEnvironmentSwitch.isEnabled("MERERUN_BAD", default: true, environment: environment))
        XCTAssertFalse(MereRunEnvironmentSwitch.isEnabled("MERERUN_UNSET", default: false, environment: environment))
        XCTAssertNil(MereRunEnvironmentSwitch.value("MERERUN_BAD", environment: environment))
        XCTAssertNil(MereRunEnvironmentSwitch.value("MERERUN_UNSET", environment: environment))
    }
}
