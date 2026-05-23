# R111 — XCD locality note

MI355X 8 XCDs × 32 CUs = 256 CUs total. chiplet_transform_chunked spreads work across XCDs in `chunk_size` units.
R43 chunk_size=32 (Triton match) — each XCD processes 32 tiles before XCD rotation. L2 per XCD = ~4 MB.
For qwen_down B16 M2048 (2048 tiles): each XCD = 256 tiles, 8 chunks of 32 tiles each.
B tile size for qwen N=4096 K=1536: 256×1536/8XCD = 49KB → fits L2 per XCD ×8 chunks rotation. Good.
For dsv3 K=7168: B tile = 256×7168 = 1.75MB per group, *16 groups = 28MB > L2. Cross-XCD L2 invalidation per group rotation.
chunk_size doesn't control group rotation; that's via persistent kernel work-steal pattern.
