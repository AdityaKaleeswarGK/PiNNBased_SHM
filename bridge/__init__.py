"""Bridge package connecting the CV crack pipeline to the PINN backend.

Modules
-------
cv_features   : headless multi-crack geometry analysis -> machine-readable report
detectors     : pluggable detection backends (YOLO-seg/-box, torchvision, classical)
depth_sources : pluggable mm-per-pixel sources (manual, reference, standoff,
                depth camera file, monocular Depth Anything V2)
pinn_solver   : parameterised Mode-I fracture PINN + F(a/W) bank evaluation
build_bank    : offline trainer for the F(a/W) bank
"""
