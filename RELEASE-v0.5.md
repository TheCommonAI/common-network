# Common Network v0.5 Beta

**The alpha is over. Common Network v0.5 is officially in beta.**

The alpha was our first public test of AI running on machines owned by the people using it. With v0.5, we’re inviting more people to help run a shared model. Start a background node, choose what your machine can spare, and see what it’s contributing from the menu bar or tray.

Here’s what shipped with the beta.

- **A new default model.** Nanbeige4.3-33B spans 52 layers. Placement tests cover the model with two Apple Silicon peers offering about 10.5 GiB each, or three offering about 7.7 GiB each, using 4-bit MLX weights. These are model budgets. Runtime and request memory are additional.
- **Background contribution.** `common join` starts the node in the background. The service resumes at login and retries after a crash. Use `common join --foreground` for a terminal session.
- **Contribution controls.** Choose Quiet, Balanced, Maximum, or Automatic, or open Custom to set resource budgets. The macOS menu bar and Windows/Linux tray panels report hosted layers, missing peers, and recent model activity.
- **Native macOS tray.** The universal SwiftUI app uses standard system controls, with Liquid Glass button styling on macOS 26. The Windows launcher hides the background console and keeps startup logs.
- **Model choice.** The previous Qwen network is still available with `common join --network qwen3.5-35b-a3b`.

If you already run a node, update the CLI and run `common join` again to refresh the background service. The first Nanbeige join downloads the layers assigned to that machine. A complete response requires peers covering every layer.

Before announcing full-model readiness, run a live Nanbeige request across separate machines. Placement and small-checkpoint inference tests have passed. Live Windows desktop behaviour also needs a hands-on check.
