# Wireframes & Mock Diagrams — Proposed Solution

**Project:** Multi-Sensor Detection of Water-Ice Stability Zones at the Lunar South Pole
**Pipeline:** Topographic Shadow (DPSR) + DFSAR Radar Polarimetry (CPR/DOP) + Diviner Thermal → **Ice Confidence Map**

> These diagrams are written in **Mermaid** (render natively on GitHub / VS Code / most Markdown viewers)
> plus **ASCII wireframes** for the presentation dashboard. Paste the Mermaid blocks into
> [mermaid.live](https://mermaid.live) to export high-resolution PNG/SVG for the slide deck.

---

## 1. System Architecture (High-Level Block Diagram)

```mermaid
flowchart TB
    subgraph INPUTS["📥 DATA INPUTS (NASA / ISRO)"]
        direction LR
        A1["LOLA DEM<br/>15168×15168 @ 20 m/px"]
        A2["LOLA PSR Shapefile<br/>80°S–90°S"]
        A3["Chandrayaan-2 DFSAR<br/>SLI / GRI / SRI"]
        A4["Official CPR Mosaic<br/>Putrevu et al. 2023"]
        A5["LRO Diviner<br/>Tmean · ZIT · Pump"]
    end

    subgraph PROC["⚙️ PHYSICS-BASED PROCESSING MODULES"]
        direction LR
        B1["DPSR Engine<br/>Solar Ray-Casting<br/>(Numba / CUDA)"]
        B2["CPR Engine<br/>SC/OC Polarimetry"]
        B3["DOP Engine<br/>Stokes / Covariance"]
        B4["Validation<br/>Computed vs Official"]
    end

    subgraph FUSION["🧮 FUSION ENGINE (diviner/)"]
        C1["Grid Alignment<br/>reproject → LOLA grid"]
        C2["Weighted Physics Score<br/>(no ML — explainable)"]
    end

    subgraph OUT["🗺️ OUTPUT"]
        D1["Ice Confidence Map<br/>+ Ranked Candidate Sites"]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B2
    A3 --> B3
    A4 --> B4
    B2 --> B4

    B1 --> C1
    B2 --> C1
    B3 --> C1
    A5 --> C1
    C1 --> C2 --> D1

    classDef inp fill:#e3f2fd,stroke:#1565c0,color:#0d47a1;
    classDef proc fill:#fff3e0,stroke:#e65100,color:#bf360c;
    classDef fus fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c;
    classDef out fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
    class A1,A2,A3,A4,A5 inp;
    class B1,B2,B3,B4 proc;
    class C1,C2 fus;
    class D1 out;
```

---

## 2. End-to-End Data Flow (Pipeline Sequence)

```mermaid
flowchart LR
    subgraph S1["1 · TOPOGRAPHY"]
        direction TB
        T1["Load LOLA DEM"] --> T2["Rasterize PSR mask"]
        T2 --> T3["Precompute Bresenham rays<br/>N_ANGLES=360, MAX_DIST=2500px"]
        T3 --> T4["Curvature-corrected<br/>visibility ray-cast"]
        T4 --> T5["Remove &lt;5px regions"]
        T5 --> T6[("DPSR.tif")]
    end

    subgraph S2["2 · RADAR POLARIMETRY"]
        direction TB
        R1["Read DFSAR HH/HV/VH/VV"] --> R2["Build scattering matrix"]
        R2 --> R3["σ_SC / σ_OC → CPR"]
        R2 --> R4["Stokes → DOP"]
        R3 --> R5[("CPR.tif")]
        R4 --> R6[("DOP.tif")]
    end

    subgraph S3["3 · THERMAL"]
        direction TB
        H1["Load Diviner grids"] --> H2[("Tmean · ZIT · Pump")]
    end

    subgraph S4["4 · FUSION"]
        direction TB
        F1["Align all 9 bands<br/>to LOLA grid"] --> F2["Percentile-clip + normalise"]
        F2 --> F3["Weighted sum<br/>Σ wᵢ·bandᵢ"]
        F3 --> F4[("Ice Confidence Map")]
    end

    T6 --> F1
    R5 --> F1
    R6 --> F1
    H2 --> F1

    classDef db fill:#c8e6c9,stroke:#2e7d32,color:#1b5e20,font-weight:bold;
    class T6,R5,R6,H2,F4 db;
```

---

## 3. Fusion Engine — Weighted Ice Confidence Score (Internal Detail)

```mermaid
flowchart LR
    subgraph IN["Aligned & Normalised Bands"]
        direction TB
        b1["CPR&nbsp;&nbsp;&nbsp;HIGH→ice"]
        b2["Tmean&nbsp;LOW→ice"]
        b3["ZIT&nbsp;&nbsp;&nbsp;LOW→ice"]
        b4["Pump&nbsp;&nbsp;HIGH→ice"]
        b5["DOP&nbsp;&nbsp;&nbsp;LOW→ice"]
        b6["PSR&nbsp;&nbsp;&nbsp;binary"]
        b7["DPSR&nbsp;&nbsp;binary"]
        b8["Slope&nbsp;LOW→ice"]
    end

    b1 -->|"×0.20"| SUM
    b2 -->|"×0.20"| SUM
    b3 -->|"×0.15"| SUM
    b4 -->|"×0.13"| SUM
    b5 -->|"×0.12"| SUM
    b6 -->|"×0.10"| SUM
    b7 -->|"×0.05"| SUM
    b8 -->|"×0.05"| SUM

    SUM["Σ (bandᵢ × wᵢ)<br/>÷ Σ wᵢ (valid only)"] --> SCORE["Ice Confidence<br/>0.0 ——— 1.0"]
    SCORE --> RANK["Ranked Candidate Sites"]

    classDef band fill:#fff3e0,stroke:#e65100;
    classDef calc fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c,font-weight:bold;
    class b1,b2,b3,b4,b5,b6,b7,b8 band;
    class SUM,SCORE,RANK calc;
```

Every weight and direction (HIGH/LOW → ice) is tied to a published reference — the fusion is
**fully explainable, no machine-learning black box.**

---

## 4. Results Dashboard — UI Wireframe (Mockup)

A proposed front-end to explore the output. This is a *concept mockup* of how a mission planner
would interact with the generated rasters.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  🌙  LUNAR SOUTH POLE — WATER-ICE STABILITY EXPLORER          [⚙ Settings] [?] ║
╠═══════════════════╦══════════════════════════════════════════════════════════╣
║  LAYERS  (toggle) ║                                                          ║
║  ─────────────────║           ┌───────────────────────────────┐            ║
║  ☑ Ice Confidence ║           │                               │  ┌───────┐ ║
║  ☐ DPSR mask      ║           │      ●  Shackleton            │  │ LEGEND│ ║
║  ☐ PSR mask       ║           │         ◍ Faustini  ★         │  │ ▓ 0.8+│ ║
║  ☐ CPR (radar)    ║           │    ◍ Haworth                  │  │ ▒ 0.5 │ ║
║  ☐ DOP            ║           │        ◍ Shoemaker    ★       │  │ ░ 0.2 │ ║
║  ☐ Tmean (thermal)║           │   ◍ Cabeus                    │  │ · 0.0 │ ║
║  ☐ DEM / hillshade║           │              ⊕ South Pole     │  └───────┘ ║
║                   ║           │                               │            ║
║  OPACITY          ║           │   ★ = top-ranked ice site     │  ┌───────┐ ║
║  [====●======] 60%║           └───────────────────────────────┘  │ ZOOM  │ ║
║                   ║                                                │ [+][-]│ ║
║  BASEMAP          ║   Lat/Lon: 87.18°S, 12.4°E   Confidence: 0.82│ [⌖ ]  │ ║
║  ( ) DEM  (●) Hill║                                                └───────┘ ║
╠═══════════════════╩══════════════════════════════════════════════════════════╣
║  📊 SELECTED SITE:  Faustini crater floor                                     ║
║  ┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐    ║
║  │ CPR    1.24 │ DOP    0.11 │ Tmean  62 K │ DPSR    ✓   │ Rank   #2   │    ║
║  │ ▲ ice-like  │ ▼ ice-like  │ ▼ cold trap │ dbl-shadow  │ of 148 PSR  │    ║
║  └─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘    ║
║  [ Export GeoTIFF ]  [ Export CSV of ranked sites ]  [ Generate Report ]     ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### Ranked Candidate Sites — Table View (Mockup)

```
┌──────┬──────────────┬──────────┬──────┬──────┬────────┬──────┬─────────────┐
│ Rank │ Site         │ Conf.    │ CPR  │ DOP  │ Tmean  │ DPSR │ Area (km²)  │
├──────┼──────────────┼──────────┼──────┼──────┼────────┼──────┼─────────────┤
│  1   │ Shoemaker-A  │  0.87 ▓▓ │ 1.31 │ 0.09 │  58 K  │  ✓   │   0.98      │
│  2   │ Faustini-fl  │  0.82 ▓▓ │ 1.24 │ 0.11 │  62 K  │  ✓   │   1.19      │
│  3   │ Haworth-NW   │  0.78 ▓  │ 1.18 │ 0.13 │  64 K  │  ✓   │   0.96      │
│  4   │ Cabeus-rim   │  0.71 ▓  │ 1.09 │ 0.14 │  71 K  │  ✓   │   0.44      │
│  …   │ …            │  …       │ …    │ …    │  …     │  …   │   …         │
└──────┴──────────────┴──────────┴──────┴──────┴────────┴──────┴─────────────┘
       [ Sort by: ●Confidence ○CPR ○Temp ○Area ]   [ Filter: DPSR-only ☑ ]
```

---

## 5. Processing Mode / Deployment View (Optional)

```mermaid
flowchart TB
    U["User / Mission Planner"] -->|"python main.py --annual --gpu"| CLI["CLI Orchestrator"]
    CLI --> GPU{"GPU<br/>available?"}
    GPU -->|Yes| CUDA["Numba CUDA kernel<br/>~30–90 min"]
    GPU -->|No| CPU["Numba parallel CPU<br/>~2–4 hrs"]
    CUDA --> R["results/*.tif"]
    CPU --> R
    R --> V["images-final/*.png<br/>figures + Ice Confidence Map"]
    V --> DASH["Dashboard / Report"]

    classDef node fill:#e3f2fd,stroke:#1565c0;
    class U,CLI,CUDA,CPU,R,V,DASH node;
```

---

### Notes for the slide deck
- Diagrams 1–3 & 5 are **Mermaid** → export SVG/PNG at [mermaid.live](https://mermaid.live) for crisp slides.
- Diagram 4 is the **UI wireframe** — the pipeline itself is headless (CLI + GeoTIFF/PNG outputs);
  the dashboard is a *proposed* presentation layer, appropriate to label "future work / concept".
- Color legend used throughout: 🟦 inputs · 🟧 processing · 🟪 fusion · 🟩 outputs.
