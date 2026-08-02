"""Model-slot to landmark adaptation.

This layer is deliberately replaceable. It holds the only knowledge of what a
particular checkpoint's slots mean, and that knowledge is currently
**provisional** — see ``maps/reloc2_18_provisional.json``. When a detector with
verified landmark semantics arrives, swapping this map is the whole migration.
"""
