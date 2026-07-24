"""The pluggable court-keypoint boundary.

``base.CourtModel`` is the seat; ``yolo_pose.YoloPoseCourtModel`` fills it for
any Ultralytics YOLO pose checkpoint, and ``adapters`` remap a model's native
keypoints into the canonical 22 (identity for the HS model, a lookup table for
the NBA model).
"""
