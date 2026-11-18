"""Native preset-first macOS menu-bar controls.

Real AppKit controls sit inside NSGlassEffectView.contentView on macOS 26;
older systems use the system popover material. No web view is embedded here.
"""
from __future__ import annotations

import os
from pathlib import Path
import threading

from commonnet.controls import (SLIDERS, actionable_error, change_service, display_value,
                                limits_equal, load_config, matching_preset, memory_gib,
                                node_status, preset_limits)


def run_macos(path, status_path):
    import AppKit as A
    import objc
    from Foundation import NSObject, NSTimer, NSString

    path, status_path = Path(path), Path(status_path)
    width, compact_height, expanded_height = 392, 430, 680
    cards = (("quiet", "Quiet", "A lighter footprint", "moon"),
             ("balanced", "Balanced", "Room for your day", "leaf"),
             ("maximum", "Maximum", "More spare power", "bolt"),
             ("automatic", "Automatic", "Let Common decide", "sparkles"))

    class ContentView(A.NSView):
        def isFlipped(self):
            return True

    class StatusDot(A.NSView):
        def drawRect_(self, rect):
            color = getattr(self, "color", A.NSColor.systemGreenColor())
            color.setFill()
            A.NSBezierPath.bezierPathWithOvalInRect_(((1, 1), (7, 7))).fill()

    class PresetButton(A.NSButton):
        def isFlipped(self):
            return True

        def drawRect_(self, rect):
            selected = self.state() == A.NSControlStateValueOn
            bounds = self.bounds()
            box = ((1, 1), (bounds.size.width - 2, bounds.size.height - 2))
            shape = A.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(box, 12, 12)
            fill = (A.NSColor.controlAccentColor().colorWithAlphaComponent_(0.14) if selected
                    else A.NSColor.labelColor().colorWithAlphaComponent_(0.035))
            fill.setFill()
            shape.fill()
            (A.NSColor.controlAccentColor().colorWithAlphaComponent_(0.85) if selected
             else A.NSColor.separatorColor().colorWithAlphaComponent_(0.65)).setStroke()
            shape.setLineWidth_(1.4 if selected else 0.6)
            shape.stroke()
            foreground = A.NSColor.labelColor().colorWithAlphaComponent_(1 if self.isEnabled() else 0.5)
            attrs = {A.NSFontAttributeName: A.NSFont.systemFontOfSize_weight_(14, A.NSFontWeightSemibold),
                     A.NSForegroundColorAttributeName: foreground}
            NSString.stringWithString_(self.title()).drawAtPoint_withAttributes_((15, 12), attrs)
            attrs = {A.NSFontAttributeName: A.NSFont.systemFontOfSize_(10.5),
                     A.NSForegroundColorAttributeName: foreground.colorWithAlphaComponent_(0.72)}
            NSString.stringWithString_(self.detail).drawAtPoint_withAttributes_((15, 35), attrs)
            if selected:
                A.NSColor.controlAccentColor().setFill()
                A.NSBezierPath.bezierPathWithOvalInRect_(((bounds.size.width - 23, 15), (7, 7))).fill()

    class Controls(NSObject):
        def applicationDidFinishLaunching_(self, notification):
            self.config = load_config(path)
            self.saved = dict(self.config.get("limits", {}))
            self.pending = {name: self.saved.get(name) for name, _, _ in SLIDERS}
            self.expanded = False
            self.busy = False
            self.sliders, self.readouts, self.preset_buttons = {}, {}, {}
            self.item = A.NSStatusBar.systemStatusBar().statusItemWithLength_(A.NSVariableStatusItemLength)
            button = self.item.button()
            image = A.NSImage.alloc().initWithContentsOfFile_(str(Path(__file__).with_name("assets") / "tray-template.svg"))
            if image:
                image.setTemplate_(True)
                image.setSize_((19, 19))
                button.setImage_(image)
            else:
                button.setTitle_("Common")
            button.setTarget_(self)
            button.setAction_("toggle:")
            button.setAccessibilityLabel_("Common contribution controls")
            controller = A.NSViewController.alloc().init()
            self.view = ContentView.alloc().initWithFrame_(((0, 0), (width, compact_height)))
            if hasattr(A, "NSGlassEffectView"):
                self.surface = A.NSGlassEffectView.alloc().initWithFrame_(((0, 0), (width, compact_height)))
                self.surface.setCornerRadius_(24)
                self.surface.setContentView_(self.view)
            else:
                self.surface = A.NSVisualEffectView.alloc().initWithFrame_(((0, 0), (width, compact_height)))
                self.surface.setMaterial_(A.NSVisualEffectMaterialPopover)
                self.surface.setState_(A.NSVisualEffectStateActive)
                self.surface.addSubview_(self.view)
            controller.setView_(self.surface)
            self.popover = A.NSPopover.alloc().init()
            self.popover.setContentViewController_(controller)
            self.popover.setBehavior_(A.NSPopoverBehaviorTransient)

            self.label("COMMON.", 24, 20, 210, 28, 19, True)
            self.dot = StatusDot.alloc().initWithFrame_(((284, 28), (9, 9)))
            self.view.addSubview_(self.dot)
            self.badge = self.label("Connecting", 298, 22, 75, 20, 11, True)
            self.node_title = self.label("Checking your node…", 24, 58, 344, 21, 13, True)
            self.node_detail = self.label("", 24, 82, 344, 33, 11, secondary=True, lines=2)
            self.activity = self.label("", 24, 118, 344, 26, 10, secondary=True, lines=2)
            self.label("CONTRIBUTION", 24, 151, 344, 17, 10, True, secondary=True)
            for index, (name, title, detail, symbol) in enumerate(cards):
                card = PresetButton.alloc().initWithFrame_(((24 + index % 2 * 177, 176 + index // 2 * 72), (167, 62)))
                card.setTitle_(title)
                card.detail = detail
                card.setButtonType_(A.NSButtonTypePushOnPushOff)
                card.setBordered_(False)
                card.setTag_(index)
                card.setTarget_(self)
                card.setAction_("choosePreset:")
                card.setAccessibilityLabel_(title + " preset. " + detail)
                self.preset_buttons[name] = card
                self.view.addSubview_(card)
            self.custom = A.NSButton.buttonWithTitle_target_action_("Custom", self, "toggleCustom:")
            self.custom.setFrame_(((24, 326), (344, 24)))
            self.custom.setBezelStyle_(A.NSBezelStyleInline)
            self.custom.setAlignment_(A.NSTextAlignmentLeft)
            self.custom.setAccessibilityLabel_("Show custom resource sliders")
            self.view.addSubview_(self.custom)
            self.custom_views = []
            maximums = {"stage_gib": max(memory_gib(), self.pending.get("stage_gib") or 0),
                        "cpu_threads": os.cpu_count() or 1, "compute_percent": 100,
                        "bandwidth_mbps": max(1000, self.pending.get("bandwidth_mbps") or 0)}
            for index, (name, title, detail) in enumerate(SLIDERS):
                top = 358 + index * 50
                title_label = self.label(title, 24, top, 214, 18, 11, True)
                readout = self.label("", 238, top, 130, 18, 11, secondary=True)
                readout.setAlignment_(A.NSTextAlignmentRight)
                slider = A.NSSlider.alloc().initWithFrame_(((24, top + 19), (344, 25)))
                slider.setMinValue_(0)
                slider.setMaxValue_(maximums[name])
                slider.setNumberOfTickMarks_(0)
                slider.setAllowsTickMarkValuesOnly_(False)
                slider.setTickMarkPosition_(A.NSTickMarkBelow)
                slider.setContinuous_(True)
                slider.setTarget_(self)
                slider.setAction_("slide:")
                slider.setAccessibilityLabel_(title)
                slider.setToolTip_(detail + ". Leftmost position uses Automatic.")
                self.view.addSubview_(slider)
                self.sliders[name], self.readouts[name] = slider, readout
                self.custom_views.extend((title_label, readout, slider))
            self.note = self.label("Memory budgets model layers. Compute and bandwidth pace work; they are not hard system caps.", 24, 560, 344, 30, 10, secondary=True, lines=2)
            self.custom_views.append(self.note)
            self.message = self.label("Changes apply when you are ready.", 24, 354, 344, 23, 10, secondary=True, lines=2)
            self.leave = A.NSButton.buttonWithTitle_target_action_("Leave network", self, "leave:")
            self.leave.setFrame_(((20, 382), (154, 32)))
            self.leave.setBezelStyle_(A.NSBezelStyleRounded)
            self.leave.setAccessibilityLabel_("Leave the Common network and remove the background service")
            self.view.addSubview_(self.leave)
            self.apply = A.NSButton.buttonWithTitle_target_action_("Apply changes", self, "apply:")
            self.apply.setFrame_(((226, 382), (146, 32)))
            self.apply.setBezelStyle_(A.NSBezelStyleRounded)
            self.apply.setKeyEquivalent_("\r")
            self.apply.setAccessibilityLabel_("Apply resource changes and reconnect your node")
            self.view.addSubview_(self.apply)
            self.syncSliders()
            self.refreshSelection()
            self.layoutPanel()
            self.updateStatus()
            self.timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(2.0, self, "checkConfig:", None, True)

        @objc.python_method
        def label(self, text, x, y, w, h, size=12, bold=False, secondary=False, lines=1):
            item = A.NSTextField.labelWithString_(text)
            item.setFrame_(((x, y), (w, h)))
            item.setFont_((A.NSFont.boldSystemFontOfSize_ if bold else A.NSFont.systemFontOfSize_)(size))
            if secondary:
                item.setTextColor_(A.NSColor.labelColor().colorWithAlphaComponent_(0.72))
            if lines > 1:
                item.setLineBreakMode_(A.NSLineBreakByWordWrapping)
                item.setMaximumNumberOfLines_(lines)
            self.view.addSubview_(item)
            return item

        @objc.python_method
        def values(self):
            return dict(self.pending)

        @objc.python_method
        def syncSliders(self):
            for name, slider in self.sliders.items():
                value = self.pending.get(name)
                slider.setMaxValue_(max(slider.maxValue(), value or 0))
                slider.setDoubleValue_(value or 0)
                self.readouts[name].setStringValue_(display_value(name, value))

        @objc.python_method
        def refreshSelection(self):
            selected = matching_preset(self.pending)
            for name, button in self.preset_buttons.items():
                button.setState_(A.NSControlStateValueOn if name == selected else A.NSControlStateValueOff)
                button.setAccessibilityValue_("Selected" if name == selected else "Not selected")
                button.setEnabled_(not self.busy)
                button.setNeedsDisplay_(True)
            self.apply.setEnabled_(not self.busy and not limits_equal(self.pending, self.saved))
            self.leave.setEnabled_(not self.busy)
            self.custom.setTitle_(("▾" if self.expanded else "▸") + " Custom" + (" · selected" if selected == "custom" else ""))
            self.custom.setAccessibilityLabel_(("Hide" if self.expanded else "Show") + " custom resource sliders")

        @objc.python_method
        def layoutPanel(self):
            height = expanded_height if self.expanded else compact_height
            self.surface.setFrameSize_((width, height))
            self.view.setFrameSize_((width, height))
            self.popover.setContentSize_((width, height))
            for item in self.custom_views:
                item.setHidden_(not self.expanded)
            self.message.setFrame_(((24, 594 if self.expanded else 354), (344, 29 if self.expanded else 23)))
            self.leave.setFrameOrigin_((20, 632 if self.expanded else 382))
            self.apply.setFrameOrigin_((226, 632 if self.expanded else 382))
            self.refreshSelection()

        @objc.python_method
        def updateStatus(self):
            status = node_status(status_path)
            online = status["state"] in ("online", "waiting")
            preparing = status["state"] in ("starting", "loading", "downloading", "moving")
            self.badge.setStringValue_("Online" if online else "Preparing" if preparing else "Offline")
            self.dot.color = A.NSColor.systemGreenColor() if online else A.NSColor.systemOrangeColor() if preparing else A.NSColor.secondaryLabelColor()
            self.dot.setNeedsDisplay_(True)
            title = status["title"]
            self.node_title.setStringValue_(title.replace("Online · hosting", "Hosting"))
            self.node_detail.setStringValue_(status["detail"])
            self.node_detail.setToolTip_(status["detail"])
            self.activity.setStringValue_(status["activity"])
            self.item.button().setToolTip_("Common — " + title + ". " + status["detail"])

        def choosePreset_(self, sender):
            self.pending = preset_limits(cards[sender.tag()][0])
            self.syncSliders()
            self.message.setStringValue_("Apply to save these limits and reconnect your node.")
            self.refreshSelection()

        def toggleCustom_(self, sender):
            self.expanded = not self.expanded
            self.layoutPanel()

        def slide_(self, sender):
            for name, slider in self.sliders.items():
                value = slider.doubleValue()
                value = round(value * 2) / 2 if name == "stage_gib" else round(value)
                self.pending[name] = value or None
            self.syncSliders()
            self.message.setStringValue_("Apply to save these limits and reconnect your node.")
            self.refreshSelection()

        def checkConfig_(self, timer):
            if not path.exists():
                A.NSApplication.sharedApplication().terminate_(None)
            else:
                self.updateStatus()

        def toggle_(self, sender):
            if self.popover.isShown():
                self.popover.performClose_(sender)
            else:
                try:
                    config = load_config(path)
                    if not self.busy and limits_equal(self.pending, self.saved):
                        self.saved = dict(config.get("limits", {}))
                        self.pending = {name: self.saved.get(name) for name, _, _ in SLIDERS}
                        self.syncSliders()
                        self.refreshSelection()
                except (OSError, ValueError):
                    self.message.setStringValue_("Could not reload saved limits. Try reopening Common.")
                self.updateStatus()
                self.popover.showRelativeToRect_ofView_preferredEdge_(self.item.button().bounds(), self.item.button(), A.NSMinYEdge)
                A.NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

        def apply_(self, sender):
            if not limits_equal(self.pending, self.saved):
                self.begin(False)

        def leave_(self, sender):
            self.begin(True)

        @objc.python_method
        def begin(self, stop):
            self.busy = True
            self.refreshSelection()
            for slider in self.sliders.values():
                slider.setEnabled_(False)
            self.message.setStringValue_("Leaving the network…" if stop else "Saving limits and reconnecting…")
            chosen = dict(self.pending)
            def work():
                try:
                    change_service(path, chosen, stop=stop)
                    result = {"ok": True, "stop": stop, "limits": chosen}
                except Exception as error:
                    result = {"ok": False, "message": actionable_error(error)}
                self.performSelectorOnMainThread_withObject_waitUntilDone_("finished:", result, False)
            threading.Thread(target=work, daemon=True).start()

        def finished_(self, result):
            self.busy = False
            if result["ok"]:
                self.saved = dict(result["limits"])
                message = "You have left the network." if result["stop"] else "Saved. Your node is reconnecting."
            else:
                message = result["message"]
            for slider in self.sliders.values():
                slider.setEnabled_(True)
            self.message.setStringValue_(message)
            self.message.setToolTip_(message)
            self.refreshSelection()
            self.updateStatus()
            if not result["ok"]:
                alert = A.NSAlert.alloc().init()
                alert.setMessageText_("Common could not update your node")
                alert.setInformativeText_(message)
                alert.addButtonWithTitle_("OK")
                alert.runModal()

    app = A.NSApplication.sharedApplication()
    app.setActivationPolicy_(A.NSApplicationActivationPolicyAccessory)
    delegate = Controls.alloc().init()
    app.setDelegate_(delegate)
    app.run()
