"""Full-game frame-processing pipeline.

Streams a long clip through the engine's plug boundary (court keypoints →
homography → detect → track → project) and writes durable, sharded per-frame
records for downstream analysis (shot charts, scouting). Nothing here loads the
whole video into memory; state is checkpointed per segment so a multi-hour run
is resumable.

See ``ENGINE.md`` for the frame-record schema and the downstream join contract.
"""
