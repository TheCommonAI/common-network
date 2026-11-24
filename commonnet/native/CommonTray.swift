import AppKit
import SwiftUI
import Combine
import Darwin
import Carbon

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
    var api: API?
    struct API: Decodable { var port: Int; var ready: Bool? }
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
    var openChat: (() -> Void)? = nil
    var shortcutAvailable = false
    private var presetBinding: Binding<Preset?> { Binding(get: { model.selectedPreset }, set: { model.choose($0) }) }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("COMMON.").font(.system(.title3, design: .default, weight: .bold))
                    Text(versionLabel).font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                }
                Spacer()
                HStack(spacing: 6) { Circle().fill(model.summary.color).frame(width: 7, height: 7); Text(model.summary.badge).font(.caption) }
                .accessibilityElement(children: .combine)
            }
            if let openChat {
                Button(action: openChat) {
                    HStack {
                        Label("Quick Chat", systemImage: "bubble.left")
                        Spacer()
                        if shortcutAvailable { Text("⇧ Space").foregroundStyle(.secondary) }
                    }
                }
                .help(shortcutAvailable ? "Open Quick Chat" : "Shift–Space is unavailable. Click to open Quick Chat.")
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

    private var versionLabel: String {
        guard let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String,
              let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String else { return "v0.5 Beta" }
        return "v\(version).\(build) Beta"
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

private struct ChatMessage: Codable, Identifiable {
    var id = UUID()
    let role: String
    let content: String
}

private struct Conversation: Codable, Identifiable {
    var id = UUID()
    var messages: [ChatMessage] = []
    var updatedAt: Date?
    var title: String { messages.first.map { String($0.content.prefix(45)) } ?? "New conversation" }
    var preview: String { messages.last(where: { $0.role == "assistant" })?.content ?? "" }
}

@MainActor
private final class QuickChat: ObservableObject {
    @Published var conversations: [Conversation]
    @Published var selection: UUID
    @Published var draft = ""
    @Published var busy = false
    @Published var reply = ""
    @Published var notice: String?
    @Published var focusRequest = 0
    @Published var showsHistory = false
    @Published var showsConversation = false
    private let statusPath: URL
    private let historyPath: URL

    init(statusPath: URL, historyPath: URL) {
        self.statusPath = statusPath
        self.historyPath = historyPath
        let saved = (try? JSONDecoder().decode([Conversation].self, from: Data(contentsOf: historyPath))) ?? []
        let initial = saved.isEmpty ? [Conversation()] : saved
        conversations = initial
        selection = initial[0].id
    }

    var messages: [ChatMessage] { conversations.first { $0.id == selection }?.messages ?? [] }
    var panelSize: NSSize {
        NSSize(width: 500, height: showsHistory ? 382 : showsConversation ? 330 : 88)
    }

    func openConversation(_ id: UUID) {
        guard !busy else { return }
        selection = id
        showsHistory = false
        showsConversation = !messages.isEmpty
        notice = nil
        focusRequest += 1
    }

    func newConversation() {
        guard !busy else { return }
        if !conversations.contains(where: { $0.messages.isEmpty }) {
            let conversation = Conversation()
            conversations.insert(conversation, at: 0)
        }
        draft = ""
        if let empty = conversations.first(where: { $0.messages.isEmpty }) { openConversation(empty.id) }
    }

    func send() {
        let prompt = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !busy, !prompt.isEmpty, let index = conversations.firstIndex(where: { $0.id == selection }) else { return }
        showsConversation = true
        let status = try? JSONDecoder().decode(NodeSnapshot.self, from: Data(contentsOf: statusPath))
        guard let api = status?.api, api.ready != false, (1...65535).contains(api.port),
              let url = URL(string: "http://127.0.0.1:\(api.port)/v1/chat/completions") else {
            notice = "Start the local chat API with common join --network --api."
            return
        }
        var request = URLRequest(url: url, timeoutInterval: 240)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let message = ChatMessage(role: "user", content: prompt)
        let context = (messages + [message]).map { ["role": $0.role, "content": $0.content] }
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: ["model": "auto", "messages": context, "stream": true])
        } catch {
            notice = error.localizedDescription
            return
        }
        conversations[index].messages.append(message)
        draft = ""
        notice = nil
        reply = ""
        busy = true
        Task {
            defer { busy = false; reply = "" }
            do {
                let (bytes, response) = try await URLSession.shared.bytes(for: request)
                guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                    var lines: [String] = []
                    for try await line in bytes.lines { lines.append(line) }
                    let body = Data(lines.joined().utf8)
                    let detail = (try? JSONDecoder().decode(ChatChunk.self, from: body))?.error?.message
                    throw TrayError.settings(detail ?? "The local chat API could not answer. Check common status.")
                }
                for try await line in bytes.lines where line.hasPrefix("data:") {
                    let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
                    if payload == "[DONE]" {
                        conversations[index].messages.append(ChatMessage(role: "assistant", content: reply))
                        conversations[index].updatedAt = Date()
                        do {
                            try JSONEncoder().encode(conversations.filter { !$0.messages.isEmpty }).write(to: historyPath, options: .atomic)
                        } catch {
                            notice = "Couldn’t save this conversation to disk."
                        }
                        return
                    }
                    guard let chunk = try? JSONDecoder().decode(ChatChunk.self, from: Data(payload.utf8)) else { continue }
                    if let error = chunk.error { throw TrayError.settings(error.message) }
                    reply += chunk.choices?.first?.delta.content ?? ""
                }
                throw TrayError.settings("The answer was interrupted. Try sending again.")
            } catch {
                conversations[index].messages.removeLast()
                draft = prompt
                notice = error.localizedDescription
            }
        }
    }

    private struct ChatChunk: Decodable {
        struct Choice: Decodable { var delta: Delta }
        struct Delta: Decodable { var content: String? }
        struct Failure: Decodable { var message: String }
        var choices: [Choice]?
        var error: Failure?
    }
}

@MainActor
private struct QuickChatView: View {
    @ObservedObject var chat: QuickChat
    let shortcutAvailable: Bool
    let close: () -> Void
    let resize: (NSSize) -> Void
    @FocusState private var focus: Field?
    @State private var search = ""
    @State private var copied: UUID?
    private enum Field { case message, search }
    private let panelColor = Color(white: 0.10).opacity(0.94)
    private let insetColor = Color.white.opacity(0.075)
    private let accent = Color(red: 0.27, green: 0.32, blue: 0.97)

    var body: some View {
        Group {
            if chat.showsHistory { history }
            else if chat.showsConversation { conversation }
            else { quick }
        }
        .frame(width: chat.panelSize.width, height: chat.panelSize.height, alignment: .top)
        .preferredColorScheme(.dark)
        .foregroundStyle(.white)
        .onAppear { focus = .message }
        .onChange(of: chat.focusRequest) { _, _ in focus = chat.showsHistory ? .search : .message }
        .onChange(of: chat.panelSize) { _, size in resize(size) }
        .onChange(of: chat.showsHistory) { _, visible in
            if visible { search = "" }
            focus = visible ? .search : .message
        }
        .onExitCommand {
            if chat.showsHistory { chat.showsHistory = false }
            else { close() }
        }
    }

    private var quick: some View {
        VStack(spacing: 8) {
            composer(compact: true)
                .background(panelColor, in: RoundedRectangle(cornerRadius: 17))
                .overlay { RoundedRectangle(cornerRadius: 17).strokeBorder(.white.opacity(0.09)) }
            if shortcutAvailable {
                Text("Shift + Space to open Common")
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.white.opacity(0.76))
                    .padding(.horizontal, 10).padding(.vertical, 4)
                    .background(panelColor, in: Capsule())
            }
        }
    }

    private var conversation: some View {
        VStack(spacing: 0) {
            header
            ScrollViewReader { scroll in
                ScrollView {
                    VStack(alignment: .leading, spacing: 15) {
                        ForEach(chat.messages) { message in
                            if message.role == "user" {
                                Text(message.content)
                                    .padding(.horizontal, 12).padding(.vertical, 8)
                                    .background(accent, in: RoundedRectangle(cornerRadius: 13))
                                    .frame(maxWidth: .infinity, alignment: .trailing)
                            } else {
                                VStack(alignment: .leading, spacing: 8) {
                                    Text(message.content).textSelection(.enabled)
                                    HStack {
                                        icon(copied == message.id ? "checkmark" : "document.on.document", "Copy reply") {
                                            NSPasteboard.general.clearContents()
                                            NSPasteboard.general.setString(message.content, forType: .string)
                                            copied = message.id
                                        }
                                        Spacer()
                                    }
                                }
                            }
                        }
                        if chat.busy {
                            Text(chat.reply.isEmpty ? "Thinking…" : chat.reply)
                                .foregroundStyle(chat.reply.isEmpty ? .secondary : .primary)
                                .textSelection(.enabled)
                        }
                        if let notice = chat.notice {
                            Text(notice).foregroundStyle(.secondary)
                        }
                        Color.clear.frame(height: 1).id("latest")
                    }
                    .font(.callout)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(15)
                }
                .onAppear { scroll.scrollTo("latest", anchor: .bottom) }
                .onChange(of: chat.messages.count) { _, _ in scroll.scrollTo("latest", anchor: .bottom) }
                .onChange(of: chat.reply) { _, _ in scroll.scrollTo("latest", anchor: .bottom) }
            }
            Divider()
            composer(compact: false)
                .background(insetColor, in: RoundedRectangle(cornerRadius: 12))
                .padding(9)
        }
        .background(panelColor, in: RoundedRectangle(cornerRadius: 17))
        .clipShape(RoundedRectangle(cornerRadius: 17))
        .overlay { RoundedRectangle(cornerRadius: 17).strokeBorder(.white.opacity(0.09)) }
    }

    private var history: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            VStack(alignment: .leading, spacing: 9) {
                Text("Your chats").font(.headline)
                HStack(spacing: 8) {
                    Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                    TextField("Search past chats…", text: $search)
                        .textFieldStyle(.plain).focused($focus, equals: .search)
                }
                .padding(9)
                .background(insetColor, in: RoundedRectangle(cornerRadius: 9))
                ScrollView {
                    VStack(alignment: .leading, spacing: 5) {
                        if matchingChats.isEmpty {
                            Text(search.isEmpty ? "No conversations yet." : "No matching conversations.")
                                .font(.callout).foregroundStyle(.secondary).padding(.vertical, 7)
                        }
                        historyGroup("Today", matchingChats.filter { $0.updatedAt.map(Calendar.current.isDateInToday) ?? false })
                        historyGroup("Earlier", matchingChats.filter { !($0.updatedAt.map(Calendar.current.isDateInToday) ?? false) })
                    }
                }
            }
            .padding(15)
        }
        .background(panelColor, in: RoundedRectangle(cornerRadius: 17))
        .clipShape(RoundedRectangle(cornerRadius: 17))
        .overlay { RoundedRectangle(cornerRadius: 17).strokeBorder(.white.opacity(0.09)) }
    }

    private var matchingChats: [Conversation] {
        chat.conversations.filter { conversation in
            !conversation.messages.isEmpty && (search.isEmpty || conversation.messages.contains {
                $0.content.localizedCaseInsensitiveContains(search)
            })
        }.sorted { ($0.updatedAt ?? .distantPast) > ($1.updatedAt ?? .distantPast) }
    }

    private func historyGroup(_ title: String, _ conversations: [Conversation]) -> some View {
        Group {
            if !conversations.isEmpty {
                Text(title).font(.caption2).foregroundStyle(.secondary).padding(.top, 4)
                ForEach(conversations) { item in
                    Button { chat.openConversation(item.id) } label: {
                        HStack(spacing: 10) {
                            Image(systemName: "bubble.left").foregroundStyle(.secondary)
                            VStack(alignment: .leading, spacing: 3) {
                                Text(item.title).font(.callout).lineLimit(1)
                                if !item.preview.isEmpty {
                                    Text(item.preview).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                                }
                            }
                            Spacer()
                            if let date = item.updatedAt {
                                Text(date, format: .dateTime.hour().minute())
                                    .font(.caption2).foregroundStyle(.secondary).fixedSize()
                            }
                            Image(systemName: "chevron.right").font(.caption2)
                        }
                        .padding(8)
                        .contentShape(Rectangle())
                        .background(item.id == chat.selection ? accent.opacity(0.19) : .clear,
                                    in: RoundedRectangle(cornerRadius: 8))
                    }
                    .buttonStyle(.plain).disabled(chat.busy)
                }
            }
        }
    }

    private var header: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Text("COMMON.").font(.caption.weight(.bold)).tracking(0.6)
                Spacer()
                if chat.showsHistory {
                    Button { chat.newConversation() } label: {
                        Label("New chat", systemImage: "plus")
                            .font(.caption)
                            .padding(.horizontal, 8).padding(.vertical, 5)
                            .background(insetColor, in: RoundedRectangle(cornerRadius: 6))
                    }.buttonStyle(.plain).disabled(chat.busy)
                } else {
                    icon("square.and.pencil", "New chat") { chat.newConversation() }.disabled(chat.busy)
                }
                icon("xmark", chat.showsHistory ? "Close history" : "Close Quick Chat") {
                    if chat.showsHistory { chat.showsHistory = false }
                    else { close() }
                }
            }
            .padding(.horizontal, 15).frame(height: 45)
            Divider()
        }
    }

    private func composer(compact: Bool) -> some View {
        HStack(spacing: 12) {
            if !compact { historyButton }
            TextField("Ask me something…", text: $chat.draft)
                .textFieldStyle(.plain)
                .font(.callout)
                .focused($focus, equals: .message)
                .onSubmit { chat.send() }
            if compact { historyButton }
            Button { chat.send() } label: {
                Image(systemName: "arrow.up")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.black.opacity(0.8))
                    .frame(width: 29, height: 29)
                    .background(.white, in: Circle())
            }
            .buttonStyle(.plain).accessibilityLabel("Send message")
            .disabled(chat.busy || chat.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
        .padding(.horizontal, compact ? 17 : 12)
        .frame(height: compact ? 58 : 42)
    }

    private var historyButton: some View {
        icon("clock", "Past chats") { chat.showsHistory = true }
    }

    private func icon(_ symbol: String, _ label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.white.opacity(0.70))
                .frame(width: 22, height: 24)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain).accessibilityLabel(label).help(label)
    }
}

private final class QuickChatPanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override func cancelOperation(_ sender: Any?) { orderOut(sender) }
}

@MainActor
private final class TrayDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    var item: NSStatusItem?
    var popover: NSPopover?
    var model: TrayModel?
    private var chat: QuickChat?
    private var chatPanel: QuickChatPanel?
    private var hotKey: EventHotKeyRef?
    private var hotKeyHandler: EventHandlerRef?
    private var shortcutAvailable = false

    private func registerShortcut() -> Bool {
        var event = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        let handlerStatus = InstallEventHandler(GetApplicationEventTarget(), { _, event, context in
            guard let event, let context else { return OSStatus(eventNotHandledErr) }
            var key = EventHotKeyID()
            guard GetEventParameter(event, OSType(kEventParamDirectObject), OSType(typeEventHotKeyID),
                                    nil, MemoryLayout<EventHotKeyID>.size, nil, &key) == noErr,
                  key.signature == 0x434F4D4D, key.id == 1 else { return OSStatus(eventNotHandledErr) }
            DispatchQueue.main.async {
                Unmanaged<TrayDelegate>.fromOpaque(context).takeUnretainedValue().toggleChat()
            }
            return noErr
        }, 1, &event, Unmanaged.passUnretained(self).toOpaque(), &hotKeyHandler)
        guard handlerStatus == noErr else { return false }
        let status = RegisterEventHotKey(UInt32(kVK_Space), UInt32(shiftKey),
                                         EventHotKeyID(signature: 0x434F4D4D, id: 1),
                                         GetApplicationEventTarget(), 0, &hotKey)
        if status != noErr { NSLog("Common could not register Shift–Space: %d", status) }
        return status == noErr
    }

    func applicationWillTerminate(_ notification: Notification) {
        if let hotKey { UnregisterEventHotKey(hotKey) }
        if let hotKeyHandler { RemoveEventHandler(hotKeyHandler) }
    }

    func windowDidResignKey(_ notification: Notification) { chatPanel?.orderOut(nil) }

    private func toggleChat() {
        if chatPanel?.isKeyWindow == true { chatPanel?.orderOut(nil) }
        else { showChat() }
    }

    private func showChat() {
        guard let chat else { return }
        popover?.performClose(nil)
        chat.showsHistory = false
        chat.showsConversation = false
        let panel: QuickChatPanel
        if let existing = chatPanel {
            panel = existing
        } else {
            panel = QuickChatPanel(contentRect: NSRect(origin: .zero, size: chat.panelSize),
                                   styleMask: [.borderless, .fullSizeContentView],
                                   backing: .buffered, defer: false)
            panel.title = "Common Quick Chat"
            panel.isFloatingPanel = true
            panel.level = .floating
            panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle]
            panel.isOpaque = false
            panel.backgroundColor = .clear
            panel.hasShadow = true
            panel.delegate = self
            let hosting = NSHostingController(rootView: QuickChatView(chat: chat, shortcutAvailable: shortcutAvailable,
                close: { [weak panel] in panel?.orderOut(nil) },
                resize: { [weak panel] size in
                    guard let panel else { return }
                    let frame = panel.frame
                    panel.setFrame(NSRect(x: frame.midX - size.width / 2, y: frame.midY - size.height / 2,
                                          width: size.width, height: size.height), display: true)
                }))
            hosting.sizingOptions = []
            panel.contentViewController = hosting
            chatPanel = panel
        }
        if let screen = NSScreen.screens.first(where: { $0.frame.contains(NSEvent.mouseLocation) }) ?? NSScreen.main {
            let frame = screen.visibleFrame
            panel.setFrame(NSRect(x: frame.midX - chat.panelSize.width / 2,
                                  y: frame.midY - chat.panelSize.height / 2,
                                  width: chat.panelSize.width, height: chat.panelSize.height), display: true)
        } else {
            panel.center()
        }
        NSApplication.shared.activate(ignoringOtherApps: true)
        panel.makeKeyAndOrderFront(nil)
        chat.focusRequest += 1
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        do {
            let options = arguments()
            let home = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".common-network")
            let config = URL(fileURLWithPath: options["--config"] ?? home.appendingPathComponent("service.json").path)
            let status = URL(fileURLWithPath: options["--status"] ?? home.appendingPathComponent("net/status.json").path)
            let assets = URL(fileURLWithPath: options["--assets"] ?? home.appendingPathComponent("commonnet-src/commonnet/assets").path)
            let model = try TrayModel(configPath: config, statusPath: status)
            self.model = model
            self.chat = QuickChat(statusPath: status,
                                  historyPath: config.deletingLastPathComponent().appendingPathComponent("tray-conversations.json"))
            shortcutAvailable = registerShortcut()
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
            let hosting = NSHostingController(rootView: TrayView(model: model, openChat: { [weak self] in self?.showChat() },
                                                                shortcutAvailable: shortcutAvailable))
            hosting.sizingOptions = [.preferredContentSize]
            popover.contentViewController = hosting
            self.popover = popover
            model.didRefresh = { [weak self] state in
                self?.item?.button?.toolTip = "Common — \(state.title). \(state.detail)"
            }
            model.refresh()
            if options["--show"] == "chat" {
                DispatchQueue.main.async { [weak self] in self?.showChat() }
            } else if options["--show"] == "1" {
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
    let historyURL = temporary.appendingPathComponent("tray-conversations.json")
    let saved = Conversation(messages: [ChatMessage(role: "user", content: "Hello")])
    try JSONEncoder().encode([saved]).write(to: historyURL)
    let chat = QuickChat(statusPath: temporary.appendingPathComponent("status.json"), historyPath: historyURL)
    guard chat.messages.first?.content == "Hello", chat.panelSize.height == 88 else {
        throw TrayError.settings("Quick Chat should open compact with its history intact")
    }
    chat.showsHistory = true
    guard chat.panelSize.height == 382 else { throw TrayError.settings("Chat history sizing failed") }
    chat.openConversation(saved.id)
    guard chat.panelSize.height == 330 else { throw TrayError.settings("Conversation sizing failed") }
    chat.newConversation()
    guard chat.messages.isEmpty, chat.panelSize.height == 88 else { throw TrayError.settings("New chat did not reset the composer") }
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
    print("PASS Quick Chat history and compact, conversation, and history sizing")
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
