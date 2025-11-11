#!/usr/bin/env python3
"""
bone_vectors.pkl의 구조를 확인하는 스크립트
"""
import pickle
import numpy as np

seq_num = 11
with open(f'/home/kimyw/github/v2_SkillMimic/Parahome/s{seq_num}/bone_vectors.pkl', 'rb') as f:
    bone_vectors = pickle.load(f)

print("=== Bone Vectors Structure ===\n")

# Body
print("BODY:")
for parent, children in bone_vectors['body'].items():
    print(f"\n  {parent}:")
    if isinstance(children, dict):
        for child, vector in children.items():
            length = np.linalg.norm(vector)
            print(f"    → {child}: {vector} (length: {length:.4f}m)")
    else:
        print(f"    Value: {children}")

# Left hand
print("\n\nLEFT HAND:")
for parent, children in bone_vectors['lhand'].items():
    print(f"\n  {parent}:")
    if isinstance(children, dict):
        for child, vector in children.items():
            length = np.linalg.norm(vector)
            print(f"    → {child}: {vector} (length: {length:.4f}m)")

# Right hand
print("\n\nRIGHT HAND:")
for parent, children in bone_vectors['rhand'].items():
    print(f"\n  {parent}:")
    if isinstance(children, dict):
        for child, vector in children.items():
            length = np.linalg.norm(vector)
            print(f"    → {child}: {vector} (length: {length:.4f}m)")
