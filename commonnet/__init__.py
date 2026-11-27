"""Common Network: a simplified Petals-style split of a Qwen3 model across block servers.

A model is cut into contiguous decoder-layer ranges. Each range is served by a
block (commonnet.block.BlockRange) that keeps a KV cache per session. A client
(commonnet.client.Client) holds the embedding, the final norm and the head, and
chains the blocks for every token.
"""

from commonnet.block import BlockRange
from commonnet.client import BlockEndpoint, Client, LocalTransport, Transport

__version__ = "0.5.1"

__all__ = ["BlockRange", "BlockEndpoint", "Client", "LocalTransport", "Transport"]
