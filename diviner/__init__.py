"""
diviner
=======
Diviner thermal integration module for the ISRO Hackathon lunar south-pole
ice-detection pipeline.

Loads, converts, aligns, analyses, and fuses three LRO Diviner thermal
datasets:

  • Mean Temperature  (Tmean)  — long-term mean surface temperature
  • Zero-Incidence Temperature (ZIT) — temperature at zero solar incidence
  • Pump parameter   (Pump)  — volatile cold-trapping efficiency proxy

These are fused with the existing DEM, Slope, PSR, DPSR, CPR, and DOP
layers to produce a 9-band feature stack and a physics-based Ice
Confidence Map.

Version : 1.0.0
Author  : ISRO Hackathon Team
"""

__version__ = "1.0.0"
__author__  = "ISRO Hackathon Team"
