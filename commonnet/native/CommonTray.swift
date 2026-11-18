import AppKit
import SwiftUI
import Combine
import Darwin

private struct Limits: Codable, Equatable {
    var stageGib: Double?
    var cpuThreads: Int?
    var computePercent: Double?
    var bandwidthMbps: Double?
    enum CodingKeys: String, CodingKey {
        case stageGib = "stage_gib", cpuThreads = "cpu_threads"
        case computePercent = "compute_percent", bandwidthMbps = "bandwidth_mbps"
    }
    static let automatic = Limits()
    func equivalent(to other: Limits) -> Bool {
        stageGib == other.stageGib && cpuThreads == other.cpuThreads &&
        (computePercent ?? 100) == (other.computePercent ?? 100) && bandwidthMbps == other.bandwidthMbps
    }
    func validated() throws -> Limits {
        for value in [stageGib, computePercent, bandwidthMbps].compactMap({ $0 }) {
            if !value.isFinite || value <= 0 { throw TrayError.settings("Resource limits must be positive numbers.") }
        }
        if let value = cpuThreads, !(1...ProcessInfo.processInfo.activeProcessorCount).contains(value) {
            throw TrayError.settings("Choose a CPU thread count available on this computer.")
        }
        if let value = computePercent, !(1...100).contains(value) {
            throw TrayError.settings("Choose a compute share between 1 and 100 percent.")
        }
        return self
    }
}

private enum Preset: String, CaseIterable, Identifiable {
    case quiet, balanced, maximum, automatic
    var id: String { rawValue }
    var title: String { rawValue.capitalized }
    var detail: String {
        switch self {
        case .quiet: return "A lighter footprint while you use your Mac."
        case .balanced: return "Share spare power with room for your day."
        case .maximum: return "Use available capacity, keeping the memory reserve."
        case .automatic: return "Let Common size your contribution."
        }
    }
    func limits(memory: Double = Double(ProcessInfo.processInfo.physicalMemory) / pow(1024, 3),
                threads: Int = ProcessInfo.processInfo.activeProcessorCount) -> Limits {
        if self == .automatic { return .automatic }
        if self == .maximum { return Limits(cpuThreads: threads, computePercent: 100) }
        let quiet = self == .quiet
        return Limits(stageGib: max(0.5, floor(memory * (quiet ? 0.20 : 0.35) * 2) / 2),
                      cpuThreads: max(1, Int(floor(Double(threads) * (quiet ? 0.25 : 0.5)))),
                      computePercent: quiet ? 25 : 50, bandwidthMbps: quiet ? 5 : 20)
    }
    static func matching(_ limits: Limits) -> Preset? {
        allCases.first { $0.limits().equivalent(to: limits) }
    }
}

private struct ServiceConfig: Decodable {
    var argv: [String]
    var limits: Limits?
    static func load(_ path: URL) throws -> ServiceConfig {
        let config = try JSONDecoder().decode(ServiceConfig.self, from: Data(contentsOf: path))
        guard config.argv.count >= 2 else { throw TrayError.settings("Run common join again to restore the background service configuration.") }
        return config
    }
}

private enum TrayError: LocalizedError {
    case settings(String)
    var errorDescription: String? {
        switch self { case .settings(let text): return text }
    }
}

private func actionable(_ error: String) -> String {
    let clean = error.replacingOccurrences(of: "\u{001b}\\[[0-9;]*m", with: "", options: .regularExpression)
    if clean.contains("Traceback") || clean.contains("returned non-zero exit status") || clean.contains("CalledProcessError") {
        return "The background node could not restart. Try Apply again. If it still fails, check ~/.common-network/join.log."
    }
    let text = clean.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    return text.isEmpty ? "Check ~/.common-network/join.log, then try again." : String(text.prefix(500))
}

private func serviceCommand(_ config: ServiceConfig, limits: Limits, leave: Bool) throws -> [String] {
    guard config.argv.count >= 2 else { throw TrayError.settings("The node launch command is missing. Run common join again.") }
    if leave { return Array(config.argv.prefix(2)) + ["--no-update", "--remove-permanent"] }
    let limits = try limits.validated()
    let flags: Set<String> = ["--stage-gib", "--memory-gib", "--cpu-threads", "--compute-percent", "--bandwidth-mbps"]
    var result = Array(config.argv.prefix(2))
    var index = 2
    while index < config.argv.count {
        let argument = config.argv[index]
        let flag = String(argument.split(separator: "=", maxSplits: 1).first ?? "")
        if flags.contains(flag) {
            index += argument.contains("=") ? 1 : 2
            continue
        }
        if flag != "--permanent" && flag != "--foreground" { result.append(argument) }
        index += 1
    }
    if let value = limits.stageGib { result.append("--stage-gib=\(value)") }
    if let value = limits.cpuThreads { result.append("--cpu-threads=\(value)") }
    if let value = limits.computePercent { result.append("--compute-percent=\(value)") }
    if let value = limits.bandwidthMbps { result.append("--bandwidth-mbps=\(value)") }
    result.append("--permanent")
    return result
}

private func runCommand(_ command: [String]) -> (Bool, String) {
    let process = Process()
    let pipe = Pipe()
    process.executableURL = URL(fileURLWithPath: command[0])
    process.arguments = Array(command.dropFirst())
    process.standardOutput = pipe
    process.standardError = pipe
    do {
        try process.run()
        // Drain while the command runs; waiting before reading can deadlock on a full pipe.
        let timeout = DispatchWorkItem { if process.isRunning { process.terminate() } }
        DispatchQueue.global().asyncAfter(deadline: .now() + 120, execute: timeout)
        let output = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        timeout.cancel()
        return (process.terminationStatus == 0, actionable(String(data: output, encoding: .utf8) ?? ""))
    } catch { return (false, actionable(error.localizedDescription)) }
}

private struct NodeSummary {
    var badge = "Offline"
    var title = "Node not running"
    var detail = "Choose your contribution, then Join network to start."
    var activity = ""
    var color: Color = .secondary
}

private struct Activity: Decodable {
    var stageSessions: Int?
    var windowSeconds: Double?
    enum CodingKeys: String, CodingKey { case stageSessions = "stage_sessions", windowSeconds = "window_seconds" }
}

private struct NodeSnapshot: Decodable {
    var state: String?
    var pid: Int32?
    var ts: Double?
    var window: [Int]?
    var stages: [[Int]]?
    var coverage: [Int]?
    var error: String?
    var activity: Activity?
    var download: Download?
    struct Download: Decodable { var bytes: Double?; var total: Double? }
    func summary(now: Double = Date().timeIntervalSince1970) -> NodeSummary {
        var value = NodeSummary()
        if let error, !error.isEmpty { value.detail = actionable(error) }
        guard let pid, pid > 0, kill(pid, 0) == 0 || errno == EPERM,
              state != "stopped", state != "leaving" else { return value }
        guard let ts, ts.isFinite, (-5...15).contains(now - ts) else {
            value.title = "Node status is out of date"
            value.detail = "The process is running but has stopped reporting status. Wait for recovery or check ~/.common-network/join.log."
            return value
        }
        if state == "error" {
            value.title = "Node needs attention"
            return value
        }
        if ["starting", "loading", "downloading", "moving"].contains(state ?? "") {
            value.badge = "Preparing"
            value.color = .orange
            value.title = ["starting": "Starting your node", "loading": "Loading model layers", "downloading": "Downloading model layers", "moving": "Updating hosted layers"][state!]!
            value.detail = "Your node is preparing to contribute. Cached layers are reused."
            if state == "downloading", let done = download?.bytes, let total = download?.total,
               done.isFinite, total.isFinite, total > 0 {
                value.detail = "\(Int(min(100, max(0, done / total * 100))))% downloaded. Leave Common running in the background."
            }
            return value
        }
        guard state == "up" else {
            value.badge = "Preparing"; value.title = "Connecting to the network"; value.color = .orange
            return value
        }
        value.badge = "Online"; value.color = .green
        value.title = "Connected to Common"
        if let window, window.count == 2 { value.title = "Hosting layers \(window[0])–\(window[1])" }
        let covered = !(stages ?? []).isEmpty && coverage?.count == stages?.count && (coverage ?? []).allSatisfy { $0 > 0 }
        value.detail = covered ? "Peers cover every model layer. The network can attempt complete requests." : "Waiting for peers to cover the remaining model layers. Complete requests need every layer online."
        if let error, !error.isEmpty { value.detail += " " + actionable(error) }
        if let count = activity?.stageSessions, count >= 0 {
            let seconds = activity?.windowSeconds ?? 300
            let period = seconds.isFinite && seconds > 0 ? "last \(max(1, Int(seconds / 60))) min" : "recently"
            value.activity = count == 0 ? "No model sessions processed in the \(period)." : "\(count) model session\(count == 1 ? "" : "s") · \(period), including local work."
        }
        return value
    }
}

@MainActor
private final class TrayModel: ObservableObject {
    @Published var pending: Limits
    @Published var saved: Limits
    @Published var customExpanded = false
    @Published var busy = false
    @Published var summary = NodeSummary()
    @Published var message = ""
    @Published var errorMessage: String?
    let configPath: URL
    let statusPath: URL
    var didRefresh: ((NodeSummary) -> Void)?
    private var timer: Timer?

    init(configPath: URL, statusPath: URL) throws {
        let config = try ServiceConfig.load(configPath)
        self.configPath = configPath; self.statusPath = statusPath
        self.pending = config.limits ?? .automatic; self.saved = config.limits ?? .automatic
        refresh()
        timer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.refresh() }
        }
    }
    var dirty: Bool { !pending.equivalent(to: saved) }
    var selectedPreset: Preset? { Preset.matching(pending) }
    func choose(_ preset: Preset?) {
        if let preset { pending = preset.limits(); message = "" }
    }
    func refresh() {
        guard FileManager.default.fileExists(atPath: configPath.path) else {
            NSApplication.shared.terminate(nil); return
        }
        if let data = try? Data(contentsOf: statusPath), let node = try? JSONDecoder().decode(NodeSnapshot.self, from: data) {
            summary = node.summary()
        } else { summary = NodeSummary() }
        // A CLI reinstall can update saved limits while the panel is closed. Preserve unsaved choices.
        if !busy, let config = try? ServiceConfig.load(configPath) {
            let wasDirty = dirty
            saved = config.limits ?? .automatic
            if !wasDirty { pending = saved }
        }
        didRefresh?(summary)
    }
    func change(leave: Bool = false) {
        guard !busy, leave || dirty || summary.badge == "Offline" else { return }
        do {
            let config = try ServiceConfig.load(configPath)
            let selected = pending
            let command = try serviceCommand(config, limits: selected, leave: leave)
            busy = true
            message = leave ? "Leaving the network…" : "Saving changes and reconnecting…"
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                let result = runCommand(command)
                Task { @MainActor in
                    guard let self else { return }
                    self.busy = false
                    if result.0 {
                        self.saved = selected
                        self.message = leave ? "You have left the network." : "Saved. Your node is reconnecting."
                    } else {
                        self.message = "Could not update your node."
                        self.errorMessage = result.1
                    }
                    self.refresh()
                }
            }
        } catch { errorMessage = actionable(error.localizedDescription) }
    }
    func numberBinding(_ key: WritableKeyPath<Limits, Double?>) -> Binding<Double> {
        Binding(get: { self.pending[keyPath: key] ?? 0 }, set: { self.pending[keyPath: key] = $0 == 0 ? nil : $0; self.message = "" })
    }
    var threadBinding: Binding<Double> {
        Binding(get: { Double(self.pending.cpuThreads ?? 0) }, set: { self.pending.cpuThreads = $0 == 0 ? nil : Int($0); self.message = "" })
    }
}

@MainActor
private struct ResourceSlider: View {
    let title: String
    let unit: String
    @Binding var value: Double
    let maximum: Double
    var step: Double = 1
    var automaticLabel = "Automatic"
    // A stepped native slider draws a stop for every increment. Bandwidth's
    // 0...1000 range produced a solid strip of overlapping stops on macOS 26.
    // Keep the system slider continuous and quantize only the bound setting.
    private var continuousValue: Binding<Double> {
        Binding(get: { value }, set: { newValue in
            value = min(maximum, max(0, (newValue / step).rounded() * step))
        })
    }
    private var readout: String { value == 0 ? automaticLabel : "\(value.formatted(.number.precision(.fractionLength(0...1))))\(unit)" }
    var body: some View {
        VStack(spacing: 2) {
            HStack { Text(title); Spacer(); Text(readout).monospacedDigit().foregroundStyle(.secondary) }
                .font(.caption)
            Slider(value: continuousValue, in: 0...maximum) { Text(title) }
                .labelsHidden()
                .accessibilityLabel(title)
                .accessibilityValue(readout)
        }
    }
}

@MainActor
private struct TrayView: View {
    @ObservedObject var model: TrayModel
    private var presetBinding: Binding<Preset?> { Binding(get: { model.selectedPreset }, set: { model.choose($0) }) }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("COMMON.").font(.system(.title3, design: .default, weight: .bold))
                Spacer()
                HStack(spacing: 6) { Circle().fill(model.summary.color).frame(width: 7, height: 7); Text(model.summary.badge).font(.caption) }
                .accessibilityElement(children: .combine)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text(model.summary.title).font(.subheadline.weight(.medium))
                Text(model.summary.detail).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                if !model.summary.activity.isEmpty {
                    Text(model.summary.activity).font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
            }
            Divider()
            VStack(alignment: .leading, spacing: 6) {
                Text("Contribution").font(.subheadline.weight(.medium))
                Picker("Contribution preset", selection: presetBinding) {
                    ForEach(Preset.allCases) { preset in Text(preset.title).tag(Optional(preset)) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                Text(model.selectedPreset?.detail ?? "Your own resource limits.")
                    .font(.caption2).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            DisclosureGroup(isExpanded: $model.customExpanded) {
                VStack(spacing: 9) {
                    ResourceSlider(title: "Model memory", unit: " GiB", value: model.numberBinding(\.stageGib),
                                   maximum: max(Double(ProcessInfo.processInfo.physicalMemory) / pow(1024, 3), model.pending.stageGib ?? 0), step: 0.5)
                    ResourceSlider(title: "CPU threads", unit: " threads", value: model.threadBinding,
                                   maximum: Double(ProcessInfo.processInfo.activeProcessorCount))
                    ResourceSlider(title: "Compute share", unit: "%", value: model.numberBinding(\.computePercent), maximum: 100)
                    ResourceSlider(title: "Bandwidth", unit: " Mbps", value: model.numberBinding(\.bandwidthMbps),
                                   maximum: max(1000, model.pending.bandwidthMbps ?? 0), automaticLabel: "Unlimited")
                    Text("Memory budgets model layers. Compute and bandwidth pace work; they are not hard system caps.")
                        .font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                .padding(.top, 7)
            } label: {
                Text(model.selectedPreset == nil ? "Custom · selected" : "Custom").font(.subheadline)
            }
            .disabled(model.busy)
            Text(model.message.isEmpty ? (model.dirty ? "Apply to save your limits and reconnect." : "Changes apply when you are ready.") : model.message)
                .font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                .accessibilityAddTraits(.updatesFrequently)
            HStack {
                Button("Leave network") { model.change(leave: true) }
                    .buttonStyle(.plain).foregroundStyle(.secondary).disabled(model.busy)
                Spacer()
                applyButton
            }
        }
        .padding(18)
        .frame(width: 376)
        .disabled(model.busy)
        .alert("Common could not update your node", isPresented: Binding(get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } })) {
            Button("OK", role: .cancel) { model.errorMessage = nil }
        } message: { Text(model.errorMessage ?? "") }
    }
    @ViewBuilder private var applyButton: some View {
        if #available(macOS 26.0, *) {
            Button(model.summary.badge == "Offline" ? "Join network" : "Apply changes") { model.change() }
                .buttonStyle(.glassProminent)
                .disabled((!model.dirty && model.summary.badge != "Offline") || model.busy)
        } else {
            Button(model.summary.badge == "Offline" ? "Join network" : "Apply changes") { model.change() }
                .buttonStyle(.borderedProminent)
                .disabled((!model.dirty && model.summary.badge != "Offline") || model.busy)
        }
    }
}

@MainActor
private final class TrayDelegate: NSObject, NSApplicationDelegate {
    var item: NSStatusItem?
    var popover: NSPopover?
    var model: TrayModel?
    func applicationDidFinishLaunching(_ notification: Notification) {
        do {
            let options = arguments()
            let home = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".common-network")
            let config = URL(fileURLWithPath: options["--config"] ?? home.appendingPathComponent("service.json").path)
            let status = URL(fileURLWithPath: options["--status"] ?? home.appendingPathComponent("net/status.json").path)
            let assets = URL(fileURLWithPath: options["--assets"] ?? home.appendingPathComponent("commonnet-src/commonnet/assets").path)
            let model = try TrayModel(configPath: config, statusPath: status)
            self.model = model
            let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
            self.item = item
            if let button = item.button {
                let image = NSImage(contentsOf: assets.appendingPathComponent("tray-template.svg"))
                image?.isTemplate = true
                image?.size = NSSize(width: 19, height: 19)
                button.image = image
                if image == nil { button.title = "Common" }
                button.target = self; button.action = #selector(toggle(_:))
                button.setAccessibilityLabel("Common contribution controls")
            }
            let popover = NSPopover()
            popover.behavior = .transient
            let hosting = NSHostingController(rootView: TrayView(model: model))
            hosting.sizingOptions = [.preferredContentSize]
            popover.contentViewController = hosting
            self.popover = popover
            model.didRefresh = { [weak self] state in
                self?.item?.button?.toolTip = "Common — \(state.title). \(state.detail)"
            }
            model.refresh()
            if options["--show"] == "1" {
                DispatchQueue.main.async { [weak self] in self?.toggle(nil) }
            }
        } catch {
            let alert = NSAlert()
            alert.messageText = "Common controls could not open"
            alert.informativeText = actionable(error.localizedDescription)
            alert.runModal()
            NSApplication.shared.terminate(nil)
        }
    }
    @objc func toggle(_ sender: Any?) {
        guard let popover, let button = item?.button else { return }
        if popover.isShown { popover.performClose(sender) }
        else {
            model?.refresh()
            popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
            NSApplication.shared.activate(ignoringOtherApps: true)
        }
    }
}

private func arguments() -> [String: String] {
    var result: [String: String] = [:]
    let args = Array(CommandLine.arguments.dropFirst())
    var index = 0
    while index + 1 < args.count {
        result[args[index]] = args[index + 1]
        index += 2
    }
    return result
}

@MainActor
private func selfTest() throws {
    let automatic = Limits.automatic
    guard automatic.equivalent(to: Limits(computePercent: 100)),
          Preset.quiet.limits(memory: 16, threads: 8) == Limits(stageGib: 3, cpuThreads: 2, computePercent: 25, bandwidthMbps: 5),
          Preset.balanced.limits(memory: 16, threads: 8) == Limits(stageGib: 5.5, cpuThreads: 4, computePercent: 50, bandwidthMbps: 20)
    else { throw TrayError.settings("Preset semantics failed") }
    let config = ServiceConfig(argv: ["/python", "/join.py", "--foreground", "--stage-gib", "8", "--network=qwen3-8b", "--no-api"], limits: nil)
    let command = try serviceCommand(config, limits: Limits(stageGib: 4), leave: false)
    guard !command.contains("--foreground"), command.contains("--no-api"), command.contains("--stage-gib=4.0"), command.last == "--permanent" else { throw TrayError.settings("Service command failed") }
    let status = NodeSnapshot(state: "up", pid: getpid(), ts: Date().timeIntervalSince1970, window: [0, 16], stages: [[0,16],[16,32]], coverage: [1,0])
    guard status.summary().badge == "Online", status.summary().detail.contains("Waiting for peers") else { throw TrayError.settings("Coverage status failed") }
    _ = NSApplication.shared
    let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("common-tray-selftest-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: temporary) }
    let configURL = temporary.appendingPathComponent("service.json")
    try Data(#"{"argv":["/usr/bin/false","/tmp/test-only.py","--network"],"limits":{}}"#.utf8).write(to: configURL)
    let model = try TrayModel(configPath: configURL, statusPath: temporary.appendingPathComponent("status.json"))
    guard !model.dirty, !model.customExpanded else { throw TrayError.settings("Initial UI state failed") }
    model.choose(.quiet)
    guard model.dirty else { throw TrayError.settings("Preset dirty state failed") }
    model.choose(.automatic)
    guard !model.dirty else { throw TrayError.settings("Preset reset state failed") }
    let host = NSHostingView(rootView: TrayView(model: model))
    let compact = host.fittingSize
    model.customExpanded = true
    RunLoop.current.run(until: Date().addingTimeInterval(0.05))
    host.layoutSubtreeIfNeeded()
    let expanded = host.fittingSize
    guard compact.width <= 400, compact.height <= 450, expanded.height <= 650 else {
        throw TrayError.settings("Native panel exceeds its layout budget: compact \(compact), expanded \(expanded)")
    }
    print("PASS presets, resource validation, service argument preservation, honest coverage status, dirty-state behavior")
    print("Native SwiftUI layout: compact \(Int(compact.width))×\(Int(compact.height)), expanded \(Int(expanded.width))×\(Int(expanded.height))")
}

@main
private struct CommonTrayMain {
    @MainActor static func main() {
        if CommandLine.arguments.contains("--self-test") {
            do { try selfTest() } catch { fputs("\(error.localizedDescription)\n", stderr); exit(1) }
            return
        }
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        let delegate = TrayDelegate()
        app.delegate = delegate
        withExtendedLifetime(delegate) { app.run() }
    }
}
