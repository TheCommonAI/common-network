# `common-chat`

A terminal client for asking the layer network questions. If you already ran
the [installer](../install.sh), you have this as `common-chat`. Prefer
`common ask` if you have the full CLI; this is the same local API with a
smaller surface.

Needs `common join` running on this machine. It talks to
`http://127.0.0.1:11440/v1`, not to a central gateway.

```bash
common-chat "What's a good way to learn recursion?"
common-chat                  # interactive, keeps conversation context
```

This folder is fetched by path from `main` by installed clients, so it stays
here.
