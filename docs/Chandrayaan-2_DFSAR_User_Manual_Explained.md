# Chandrayaan-2 Moon Radar (DFSAR) — Explained for Complete Beginners

> This is a friendly, no-jargon walkthrough of ISRO's official document `ch2_dfsar_user_manual_v1.0.pdf`. You do **not** need any space, engineering, or computer background to follow this. Every technical word is explained the first time it shows up, usually with an everyday comparison. Nothing from the original document is left out — it's just translated into plain English.

---

## 0. First, what is this thing called "DFSAR"? (Start here)

Chandrayaan-2 is an Indian spacecraft that orbits the Moon. It carries several instruments — think of them like different "eyes" or "tools" the spacecraft uses to study the Moon. One of those tools is called:

**DFSAR = Dual Frequency Synthetic Aperture Radar**

Let's break that name down piece by piece, like peeling an onion:

- **Radar** — You've probably heard "radar" in the context of airport flight tracking or weather forecasts. The idea is simple: you send out a radio wave (invisible to our eyes, like the signal from a TV remote but much stronger), it bounces off something (a plane, a raincloud, or in this case, the Moon's surface), and comes back to you. By measuring how the wave changed (timing, strength, etc.), you can figure out what the surface looks like — without needing any sunlight. That's why radar can "see" the Moon's surface even in permanently shadowed craters where cameras would just see black.
- **Synthetic Aperture** — A camera lens needs to be big to take a very sharp, detailed photo from far away. A radar antenna that's "big enough" to get sharp images from a spacecraft would be impractically huge. So engineers use a clever trick: as the spacecraft moves along its path, it keeps sending and receiving radar pulses, and then a computer combines all of those readings together cleverly, as if they came from one giant "synthetic" (artificially created) antenna covering the whole flight path. The end result is a much sharper image, made possible purely through clever math, not a giant physical antenna. "Aperture" just means "opening" — it's the technical word for the size/area of an antenna or lens.
- **Dual Frequency** — DFSAR doesn't use just one "channel" of radio wave, it uses **two**: an "L-band" and an "S-band" (these are just names for specific radio frequency ranges, like how FM radio and AM radio are different frequency ranges). Using two frequencies lets scientists learn different things about the Moon's surface — e.g., one frequency might penetrate slightly into the soil/dust, the other might bounce mostly off the very top surface. Combining both gives a richer picture, kind of like taking both a regular photo and an X-ray of the same object.

**So in one sentence:** DFSAR is a radar "camera" on Chandrayaan-2 that uses two different radio frequencies to take detailed images of the Moon's surface, working even in total darkness.

**Important:** This particular manual does **not** explain how the radar instrument itself works in detail. It only explains **how the data/images that come out of it are organized into files and folders**, so that a scientist downloading this data from the internet knows what each file is and which program to open it with. Think of it like the "read me" file that explains how a downloaded folder of photos and videos from a trip is organized — which folder has the raw camera files, which has the edited photos, which app to open them in, etc. That's exactly the role of this manual, just for Moon radar data.

---

## 1. The Document's "ID Card" (basic info about the manual itself)

Every official document has some bookkeeping info at the top. Here it is, explained:

| Field | Value | What this means |
|---|---|---|
| Report No. & Date | SAC/SIPG/MDPD/CH2/SAR/2020/12/23/v1.0 | This is just a unique tracking code + the date it was finalized (Dec 23, 2020), like a serial number on a product. |
| Title | Chandrayaan-2 Dual Frequency SAR Data Product User Manual | The official name of this document. |
| Type of Report | Technical | It's a factual/instructional document, not a research paper with new discoveries. |
| Authors | CH2 SAR Data Products Team | The group of people at ISRO who wrote it. |
| Originating Unit | Microwave Data Processing Division (MDPD), Signal & Image Processing Group (SIPG), Space Applications Centre (SAC) | The specific ISRO department responsible. "Microwave" here just refers to the type of radio wave radar uses — not the kitchen appliance! Radar microwaves and kitchen microwaves both use the same part of the electromagnetic spectrum, just for very different purposes. |
| Abstract | Gives users familiarity with the data products | A one-line summary: "this helps you understand the data files." |
| Security Classification | Unrestricted | Anyone is allowed to read this — it's public, not classified/secret. |
| Distribution | Users of DFSAR Data | Meant for scientists/researchers/engineers who plan to use this radar data. |

**Document History:** Only one version has ever been released (v1.0, Dec 23, 2020) — no corrections or updates have been made since.

**Acknowledgements:** The authors simply thank their director and colleagues, plus two other ISRO groups: **ISSDC** (the team that handles incoming spacecraft data) and the **PDS4 working group** (a team that helps standardize how planetary data gets archived — more on "PDS4" below).

---

## 2. The Goal of This Manual (Section "Objective")

In the document's own words, simplified: *"Help users understand the basic structure of Chandrayaan-2 DFSAR data: how it's organized, how many types of files exist, and what format each file is in."*

So really, this manual answers 3 simple questions:
1. **Where do I find the file I need?** (folder structure)
2. **What kinds of files exist, and what's the difference between them?** (data types)
3. **What format is each file in, and what program do I open it with?** (file formats + tools)

---

## 3. Where Is Everything Stored? (The Folder Structure)

Imagine ISRO has one giant shared hard drive containing data from *all* its science missions, not just Chandrayaan-2. This section explains how that giant hard drive is organized — like explaining the folder structure of a shared company drive: "Department X's folder is inside Region Y's folder, which is inside the Company folder," etc.

This filing system follows a standard called **PDS4** (more on this in Section 4). For now, just know: PDS4 is a globally-agreed-upon "filing cabinet template" for organizing space science data, used by NASA and other space agencies too, so that data is consistent and any scientist anywhere can understand the layout, not just ISRO staff.

### 3.1 Top-Level Folders (Mission Level)

```
isda                     ← "ISRO Science Data Archive" — the single root folder for ALL ISRO mission data
 └── ch2_bundle          ← "Chandrayaan-2" — everything specific to this one mission
      ├── cho_bundle     ← Data from the ORBITER part of the spacecraft (the part that stayed circling the Moon)
      ├── chl_bundle     ← Data from the LANDER part (the part that was meant to land on the surface)
      └── chr_bundle     ← Data from the ROVER part (the small vehicle meant to drive around after landing)
```

*(Side note: the Chandrayaan-2 lander/rover did not survive their landing attempt in 2019, but their data folders still exist in the archive structure as planned.)*

A **"bundle"** here just means "a packaged collection of related files" — similar to how a ZIP file bundles many files together under one name.

### 3.2 Mission Phases

Within the Orbiter/Lander/Rover folders, data is further split by **when** it was collected, called the "mission phase":

| Folder code | What it means | Plain English |
|---|---|---|
| `ebp` | Earth Bound Phase | Data collected while the spacecraft was still flying around Earth, *before* it left for the Moon (e.g., during initial instrument testing). |
| `nop` | Normal Operation Phase | Data collected once the spacecraft reached the Moon and started doing its real science work. |

### 3.3 Instrument Folders

Inside each phase folder, there's one folder per **instrument** (remember, "instruments" = the different science tools/eyes onboard). For the **Orbiter**, the instrument folders include:

| Code | Full name | What it studies (in plain terms) |
|---|---|---|
| `tmc` | Terrain Mapping Camera | A normal-light camera that maps the Moon's terrain (hills, craters) |
| `lir` | Imaging IR Spectrometer | A camera that sees infrared light (heat-related light invisible to our eyes) to study minerals |
| `ohr` | Orbiter High Resolution camera | A very sharp, zoomed-in regular camera |
| **`sar`** | **Dual Frequency Synthetic Aperture Radar** | **← This is our radar instrument — the subject of this whole manual** |
| `cla` | Chandrayaan-2 Large Area Soft X-ray Spectrometer | Detects X-rays to study the Moon's elemental composition (what chemicals make up the surface) |
| `xsm` | Solar X-ray Monitor | Watches X-rays coming from the Sun (used as a reference for the X-ray spectrometer above) |
| `cha` | CH2-Atmospheric Compositional Explorer-2 | Studies the Moon's extremely thin atmosphere |
| `frs` | Dual Frequency Radio Science Experiment | Uses radio signals to study things like the Moon's gravity field |

The Lander had `rip`, `chl` (an instrument with the same short-name as the lander bundle, just a coincidence in naming), and `ils`. The Rover had `lib` (a laser tool that vaporizes a tiny bit of rock to analyze its chemistry) and `apx` (a sensor that detects particles to determine rock/soil composition).

**Putting it together:** if you want radar data from the orbiter, you'd navigate:
`isda → ch2_bundle → cho_bundle → nop (or ebp) → sar_collection`

A **"collection"** is simply PDS4's word for "the folder containing all files for one particular instrument."

### 3.4 Inside the Radar's Folder (`sar_collection`)

Once you're inside the radar's own folder, the files are organized by **what kind of file it is**, then by **how processed it is**, then by **date**:

```
sar_collection
 │
 ├── browse        → quick "preview" images, like a thumbnail, so you can glance at data without
 │                    opening the full heavy file
 │     ├── calibrated → yyyymmdd/   (date folder)
 │     └── derived    → yyyymmdd/
 │
 ├── data          → the actual radar image files (the "real" content)
 │     ├── raw        → yyyymmdd/
 │     ├── calibrated  → yyyymmdd/
 │     └── derived    → yyyymmdd/
 │
 ├── geometry      → supporting "where is this exactly" location files matching the data
 │     ├── raw        → yyyymmdd/
 │     ├── calibrated  → yyyymmdd/
 │     └── derived    → yyyymmdd/
 │
 └── spice_kernels → files describing exactly where the spacecraft itself was pointed/located
       └── yyyymmdd/
```

Let's unpack each unfamiliar word here:

- **"raw" vs "calibrated" vs "derived"** — This describes how much processing/cleanup has been done to the data, similar to: raw phone camera file → color-corrected/edited photo → a collage made from multiple edited photos.
  - **raw** = straight from the instrument, basically untouched.
  - **calibrated** = cleaned up and corrected into an actual usable image.
  - **derived** = extra/advanced products built *from* the calibrated images (e.g., further scientific analysis results).
- **"geometry"** — Imagine you have a photo, but you also need a separate sheet of paper that tells you "pixel (5,10) in this photo corresponds to this exact spot on a map, at this exact angle." That's what geometry files are — they tell you precisely where on the Moon (which latitude/longitude) each part of the radar image corresponds to.
- **"spice_kernels"** — "SPICE" is a specific file format (originally created by NASA) used to record a spacecraft's exact position, speed, and orientation (which way it's tilted/pointed) at any given moment. Think of it like a flight data recorder ("black box") for the spacecraft's position and orientation — useful so scientists can later calculate exactly where the spacecraft was when each piece of radar data was captured.
- **"yyyymmdd"** — This is just a placeholder showing that the final folder is named using the date the data was collected, in Year-Month-Day format (e.g., `20191019` = October 19, 2019).

---

## 4. What Exactly Is a "PDS4 Data Product"? (The filing rule behind everything)

We mentioned "PDS4" a few times — let's properly explain it now.

**PDS4 (Planetary Data System, version 4)** is essentially a **rulebook for how to package and label space science data files**, originally developed by NASA, and adopted internationally (including by ISRO) so that scientists worldwide can pick up *any* agency's planetary data and understand it the same way — like a universal "nutrition label" standard that every food company in the world agreed to follow, regardless of country.

The core rule of PDS4 is: **every actual data file must be paired with a "label" file that describes it.**

- The **data file** = the real content (an image, a spreadsheet of numbers, etc.) — this is called a "digital object."
- The **label file** = always written in a format called **XML** (eXtensible Markup Language — basically a structured text format, a bit like a very strict, computer-readable form with labeled fields, e.g. `<FileSize>10MB</FileSize>`). This label tells you things like: what kind of data is in the file, how it's structured, how to interpret each value, units of measurement, etc. — without you having to guess.
- Every single product also gets a unique ID called an **LID** (Logical Identifier) — think of it as a permanent serial number/barcode for that specific file, so it can always be uniquely referred to, even years later, even if it gets copied elsewhere.

**Analogy:** Imagine every photo you take on your phone automatically came with a small text file next to it saying "this photo was taken on this date, with this camera setting, this many megapixels, in this format." That text file is the "label," the photo is the "data product," and PDS4 is the rule that says "always do this, every time, the same way."

### The Three "Processing Levels" You'll See Everywhere

Throughout this manual, data is described as **Level 0, Level 1, Level 2** (sometimes "Level 3"). This is simply a scale of **how much processing has been done**:

| Level | Common name | Plain meaning |
|---|---|---|
| **L0** | Raw | Practically untouched — straight off the spacecraft's transmission |
| **L1 / L2** | Calibrated | Cleaned up, corrected, and converted into an actual viewable image |
| **L3** | Derived | Extra, more advanced science products built on top of the calibrated images |

Think of L0→L1/L2→L3 like: **undeveloped camera film → a printed photograph → a photo album page with annotations and analysis added.**

---

## 5. Raw Data Explained (Level 0)

This is the "undeveloped film" stage — the data almost exactly as it came down from the spacecraft. The technical term for these files is **RDR (Raw Data Record)**.

### How raw data is created (step by step, explained)

1. **The radar sends down "frames" of data.** A "frame" is just one chunk/segment of the continuous data stream, like one page out of a long scroll of paper.
2. **The data gets "BAQ decompressed."**
   - **BAQ (Block Adaptive Quantization)** is a compression trick used *on board the spacecraft* before sending data to Earth — similar to how you might compress (zip) a file before emailing it, to save space/bandwidth, since sending data from the Moon back to Earth is slow and limited.
   - "Decompressing" it on the ground just means *undoing* that compression so we get back the full original data.
3. **The science data is organized as "I/Q" data and reassembled into continuous range lines.**
   - **I/Q (In-phase and Quadrature)** is the standard way radar signals are recorded. A radar wave, like any wave, has both a *strength* (how big the wave is) and a *timing/phase* (where exactly the wave is in its up-down cycle when it returns). Recording both pieces of information requires **two numbers per measurement** instead of just one — these two numbers are conventionally called "I" and "Q." Don't worry about the physics — just remember: **each radar measurement is stored as a pair of numbers, not a single number**, because that's the only way to fully capture what a radar wave is doing.
   - "Range lines" — As the spacecraft flies overhead, it sends out radar pulses and listens to the echoes (echoes = the returning signal). Each pulse's echo, once arranged in order, forms one line of an image. Stack enough lines together (like stacking rows of pixels) and you get a full 2D image. So "reassembling into continuous range lines" simply means properly ordering and stitching together each pulse's echo into rows that will eventually form an image.
4. **Timing information is added per "PRF."**
   - **PRF (Pulse Repetition Frequency)** = how often the radar fires a new pulse (e.g., 2,000 times per second). Each individual pulse gets a precise timestamp recorded.
5. **Each frame starts with a small "header."**
   - A "header" is just a small block of summary info placed at the very beginning of a file/frame — similar to the title and date written at the top of a printed report, before the main content starts.
6. **Missing frames are filled in with zeros.**
   - Sometimes, due to transmission issues, a chunk of data simply never arrives. Rather than leaving an awkward gap (which could confuse software expecting a fixed-size file), engineers fill that missing chunk with zeros — like leaving a blank, clearly-marked gap in a scrapbook page instead of just cutting the page shorter.

### Table: The Raw (Level 0) Files You'll Actually See

| File you'll see | What kind of file is it? | File extension | What's inside it (plain English) |
|---|---|---|---|
| Raw packets | A generic binary file (see note below) | `.dat` | The actual raw radar measurements: timestamps, some basic spacecraft info, and the science data itself, organized pulse-by-pulse |
| Raw SAR Label file | A structured text file (XML) | `.xml` | The "description card" explaining what's inside the `.dat` file above |
| Raw OAT | A spreadsheet-style file | `.csv` | **OAT = Orbit and Attitude.** This records exactly where the spacecraft was ("orbit" = its path/position) and which way it was tilted/pointed ("attitude" = orientation), at each moment |
| Raw OAT Label file | XML | `.xml` | The "description card" for the OAT file above |

> **What's a "binary" file?** Binary files store data as raw numbers/bytes that only make sense to a computer program that knows the exact recipe for reading them — unlike a `.txt` file, you can't just open a binary file in Notepad and read it sensibly. You need the right software (mentioned in Section 8 below) that knows how to interpret the bytes correctly.
>
> **What's a CSV file?** CSV = "Comma-Separated Values" — basically a very simple spreadsheet stored as plain text, where each line is a row and commas separate the columns. You can open these in Excel, Google Sheets, or even a plain text editor.

### Example: What an actual raw-data folder looks like

```
data
 └── raw
      └── 20191019/                                                     ← this means "Oct 19, 2019"
           ├── ch2_sar_nrxl_20191019t041710471_d_r0b_xx_fp_xx_gds.dat      ← the actual radar data
           └── ch2_sar_nrxl_20191019t041710471_d_r0b_xx_fp_xx_gds.xml      ← its description card
geometry
 └── raw
      └── 20191019/
           ├── ch2_sar_nrxl_20191019t041710471_g_oat_xx_fp_xx_gds.csv      ← orbit/position info
           └── ch2_sar_nrxl_20191019t041710471_g_oat_xx_fp_xx_gds.xml      ← its description card
```

**Key takeaway you'll notice everywhere in this archive:** every real data file (`.dat`, `.csv`, `.tif`) is *always* accompanied by a matching `.xml` "description card" file with almost the same name. That's the PDS4 rule from Section 4 in action.

---

## 6. Calibrated (Processed) Data Explained — Where Raw Signal Becomes an Actual Picture

This is the "raw radar numbers turn into something you can actually look at as an image" stage. There are three increasingly refined versions: **Level 1A, Level 1B, and Level 2.** Each one builds on/improves the previous one.

### 6.1 Level 1A — "SLC" Image (also called SLI)

- **SLC = Single Look Complex.** Let's unpack this odd-sounding name:
  - **"Single Look"** means the image is built from one single pass of data (as opposed to combining multiple passes together to reduce noise — that's a different, more averaged kind of product not covered as the main focus here).
  - **"Complex"** is the key word, and it does **not** mean "complicated" here — it's a math term. Remember the I/Q pair of numbers we discussed in Raw Data? Well, this image format keeps *both* of those numbers for every single pixel, instead of just one. In math, a pair of numbers like this is officially called a "complex number" (don't worry about why — just know it means **"this pixel secretly contains 2 numbers, not 1."**)
  - This image is also described as being in **"slant range."** Imagine standing in an airplane looking down and slightly forward at the ground through a window — the distance you're measuring to the ground is "slanted," not straight down. That's "slant range": distance measured directly along the radar's line of sight, which is at an angle, not corrected yet to be a true straight-down ground distance. We'll fix that in the next level (1B).
- These images are saved as **TIFF files** (`.tif`) — TIFF is simply a common, high-quality image file format (similar in spirit to JPEG or PNG, but typically used for higher-precision scientific/professional images rather than casual photos).
- If the radar was operating in a mode that captures multiple "polarizations" (explained in the box below), you get **one separate TIFF image per polarization**.

> **What is "polarization"?** Think of polarized sunglasses — they only let through light waves vibrating in a certain direction (that's literally what "polarized" means for light). Radio waves (which is what radar uses) can also be sent out vibrating in different directions — e.g., horizontally (H) or vertically (V), or in a circular spinning pattern, clockwise or counter-clockwise (these circular versions are called Right-Hand and Left-Hand, written as R and L). DFSAR can transmit in one direction and listen for the echo in either the same or a different direction. This is written as two letters: first letter = how it was *sent*, second letter = how the echo was *received*. So:
> - **HH** = sent Horizontal, received Horizontal
> - **HV** = sent Horizontal, received Vertical
> - **VH** = sent Vertical, received Horizontal
> - **VV** = sent Vertical, received Vertical
> - **LH / LV** = sent Left-circular, received Horizontal/Vertical-equivalent circular component
>
> Why bother with multiple polarizations? Different surface textures and materials reflect different polarizations differently — e.g., smooth ice might reflect very differently than rough rocky rubble. By capturing multiple polarization combinations, scientists get extra clues about what the surface is actually made of and how rough/smooth it is — similar to how looking at an object under different colored lighting can reveal different details.

#### Files you'll see for Level 1A (SLC)

| File | File type | Extension | What's inside (plain English) |
|---|---|---|---|
| Image | Binary | `.tif` | The actual radar image. Each pixel stores 2 numbers (the I and Q pair), each stored with high precision ("4-byte floating point" just means a fairly precise decimal number format used by computers). Each pixel also knows its Moon latitude/longitude ("Seleno-tagged" — "Seleno-" is a prefix meaning "Moon-related," from "Selene," the Greek Moon goddess — so "Seleno-tagged" just means "tagged with its Moon location"). |
| Image File Label | XML (structured text) | `.xml` | Description card for the image above |
| Grid | A spreadsheet-like file | `.txt` | A file listing the latitude/longitude for sample points across the image, plus the slant-range distance and "incidence angle" (the angle at which the radar beam hit the ground — like the angle of sunlight hitting a wall) at each of those points |
| Grid file Label | XML | `.xml` | Description card for the Grid file |
| OAT | Spreadsheet | `.csv` | Orbit and attitude info (same concept as before — spacecraft position & orientation) |
| OAT Label file | XML | `.xml` | Description card for the OAT file |

### 6.2 Level 1B — "GRI" Image (Ground Range Image)

- **GRI = Ground Range Image.** This is where the "slant range" distortion from Level 1A gets mathematically corrected, so the image now represents the *true, straight-down ground distance* — much closer to how a normal top-down map would look, rather than the angled view from Level 1A.
- Each pixel here is now a simpler single number (not the I/Q pair anymore) — specifically what's called an **"unsigned short integer,"** which just means "a whole, non-negative number, stored compactly" (think of it as just a brightness/strength value per pixel, similar to a black-and-white photo's pixel values, but representing radar signal strength instead of light brightness).
- Again, one TIFF file is produced per polarization channel if multiple were captured.

#### Files you'll see for Level 1B (GRI)

| File | File type | Extension | What's inside |
|---|---|---|---|
| Image | Binary | `.tif` | The ground-corrected radar image. Each pixel = one simple whole number (signal strength). Tagged with Moon lat/long. |
| Image File Label | XML | `.xml` | Description card for the image |
| Grid | Spreadsheet-like | `.txt` | Lat/long info per sample point, plus ground range and incidence angle |
| Grid file Label | XML | `.xml` | Description card for the Grid file |

### 6.3 Level 2 — "SRI" Image (Seleno Referenced Image)

- **SRI = Seleno Referenced Image** ("Seleno" again = "Moon-related," as explained above).
- This is the final, most "ready-to-use" stage: the image has now been **map-projected** — meaning it's been placed correctly onto an actual coordinate map of the Moon, properly aligned to true north, just like a satellite photo of Earth that's correctly aligned with a world map (north pointing up, correct scale, no tilt/skew). At Levels 1A and 1B, the image geometry still roughly followed the spacecraft's flight path; at Level 2, it's been "straightened out" to match a standard map grid.
- This is generally the most useful, plug-and-play version for most users who just want to view the data as a normal map-like image.

#### Files you'll see for Level 2 (SRI)

| File | File type | Extension | What's inside |
|---|---|---|---|
| Image | Binary | `.tif` | The final map-aligned image. Each pixel = a simple whole number representing signal strength ("amplitude"). |
| Image File Label | XML | `.xml` | Description card for the image |
| Grid | Spreadsheet-like | `.txt` | Lat/long for each pixel, aligned to true north |
| Grid file Label | XML | `.xml` | Description card for the Grid file |

### 6.4 Quick Side-by-Side Comparison

| Stage | Nickname | What's special about it | Pixel stores... |
|---|---|---|---|
| Level 1A | SLC / SLI | Closest to raw signal; geometry still "slanted"; keeps both I and Q numbers | 2 precise decimal numbers per pixel |
| Level 1B | GRI | Geometry corrected to true ground distance | 1 simple whole number per pixel |
| Level 2 | SRI | Fully aligned to a real Moon map, ready to view like a satellite photo | 1 simple whole number per pixel |

**Simple analogy for the whole pipeline:** Level 1A is like a photo taken at a tilted angle through an airplane window; Level 1B is like digitally "straightening" that photo so it looks like it was taken straight down; Level 2 is like then pasting that straightened photo onto exactly the right spot on a world map/atlas, correctly rotated and scaled.

---

## 7. Real Examples of What These Folders/Files Actually Look Like

### 7.1 Example: "Full Polarization" Dataset

"Full Polarization" (shortened to **"fp"** in filenames) means all four combinations were captured: HH, HV, VH, VV (explained in the polarization box in Section 6.1).

```
browse/calibrated/20191019/
   ch2_sar_ncxl_20191019t041710471_b_brw_xx_fp_xx_gds.png   ← a small quick-look preview picture
   ch2_sar_ncxl_20191019t041710471_b_brw_xx_fp_xx_gds.xml   ← its description card

data/calibrated/20191019/
   ..._d_gri_in_fp_xx_gds.tif        ← GRI image, combined/overall view
   ..._d_gri_xx_fp_hh_gds.tif        ← GRI image, HH polarization only
   ..._d_gri_xx_fp_hv_gds.tif        ← GRI image, HV polarization only
   ..._d_gri_xx_fp_vh_gds.tif        ← GRI image, VH polarization only
   ..._d_gri_xx_fp_vv_gds.tif        ← GRI image, VV polarization only
   ..._d_gri_xx_fp_xx_gds.xml        ← description card for the GRI images
   ..._d_sli_xx_fp_hh_gds.tif        ← SLC image, HH polarization
   ..._d_sli_xx_fp_hv_gds.tif        ← SLC image, HV polarization
   ..._d_sli_xx_fp_vh_gds.tif        ← SLC image, VH polarization
   ..._d_sli_xx_fp_vv_gds.tif        ← SLC image, VV polarization
   ..._d_sli_xx_fp_xx_gds.xml        ← description card for the SLC images
   ..._d_sri_in_fp_xx_gds.tif        ← SRI (map-aligned) image, combined view
   ..._d_sri_xx_fp_hh_gds.tif        ← SRI image, HH polarization
   ..._d_sri_xx_fp_hv_gds.tif        ← SRI image, HV polarization
   ..._d_sri_xx_fp_vh_gds.tif        ← SRI image, VH polarization
   ..._d_sri_xx_fp_vv_gds.tif        ← SRI image, VV polarization
   ..._d_sri_xx_fp_xx_gds.xml        ← description card for the SRI images

geometry/calibrated/20191019/
   ..._g_gri_xx_fp_xx_gds.csv  + .xml   ← location grid for the GRI image + its description card
   ..._g_oat_xx_fp_xx_gds.csv  + .xml   ← spacecraft orbit/orientation info + description card
   ..._g_sli_xx_fp_xx_gds.csv           ← location grid for the SLC image
   ..._g_sri_xx_fp_xx_gds.csv           ← location grid for the SRI image
   ..._g_xxx_xx_fp_xx_gds.xml           ← one combined description card covering geometry files
```

### 7.2 Example: "Circular Polarization" Dataset

"Circular Polarization" (shortened to **"cp"**) is the other capture mode, using LH/LV channels (left-hand circular combinations) instead of the HH/HV/VH/VV linear ones.

```
browse/calibrated/20190904/
   ch2_sar_ncxs_20190904t122209694_b_brw_xx_cp_xx_g26.png + .xml

data/calibrated/20190904/
   ..._d_gri_in_cp_xx_g26.tif      ← GRI, combined view
   ..._d_gri_xx_cp_lh_g26.tif      ← GRI, LH channel
   ..._d_gri_xx_cp_lv_g26.tif      ← GRI, LV channel
   ..._d_gri_xx_cp_xx_g26.xml      ← description card
   ..._d_sli_xx_cp_lh_g26.tif      ← SLC, LH channel
   ..._d_sli_xx_cp_lv_g26.tif      ← SLC, LV channel
   ..._d_sli_xx_cp_xx_g26.xml      ← description card
   ..._d_sri_in_cp_xx_g26.tif      ← SRI, combined view
   ..._d_sri_xx_cp_lh_g26.tif      ← SRI, LH channel
   ..._d_sri_xx_cp_lv_g26.tif      ← SRI, LV channel
   ..._d_sri_xx_cp_xx_g26.xml      ← description card

geometry/calibrated/20190904/
   ..._g_gri_xx_cp_xx_g26.csv
   ..._g_oat_xx_cp_xx_g26.csv + .xml
   ..._g_sli_xx_cp_xx_g26.csv
   ..._g_sri_xx_cp_xx_g26.csv
   ..._g_xxx_xx_cp_xx_g26.xml
```

### 7.3 How to "read" one of these long filenames

Take this example: `ch2_sar_ncxl_20191019t041710471_d_gri_xx_fp_hh_gds.tif`

Breaking it into chunks, left to right:

| Chunk | Meaning |
|---|---|
| `ch2` | Chandrayaan-2 (the mission) |
| `sar` | This is SAR/radar data |
| `ncxl` | An internal mode/processing code (not spelled out in detail in this manual) |
| `20191019t041710471` | The date and time the data was captured: Oct 19, 2019, at time `04:17:10.471` |
| `d` | This is a **d**ata file (you'll also see `g` for geometry files, `b` for browse/preview files) |
| `gri` | The product type — here, "Ground Range Image" (you'll also see `sli` for SLC, `sri` for SRI, `oat` for orbit/attitude, `brw` for browse) |
| `xx` | "Not applicable here" placeholder |
| `fp` | Full Polarization mode (you'll also see `cp` for Circular Polarization) |
| `hh` | The specific polarization channel of this file (HH in this case) |
| `gds` | An internal site/processing tag |
| `.tif` | The file format (TIFF image) |

You don't need to memorize this — just know that **every long filename is really just a list of short labels glued together with underscores**, each one telling you one fact about the file (what mission, what date, what product, what polarization, etc.) — similar to how a barcode or a product SKU encodes multiple bits of information in one string. For the *exact, official* definition of every single one of these filename segments, the manual itself says to check a separate, more detailed companion document (reference [1] in Section 11 below) — this particular manual only gives examples, not the full rulebook.

---

## 8. What Software Do I Need to Open These Files? (Section "Tools")

| Data product | File extension | What kind of numbers are stored inside | Recommended software |
|---|---|---|---|
| Level-0 (Raw) | `.dat` | Plain raw bytes | Any general-purpose binary file viewer/hex editor |
| Level-1A (SLC) | `.tif` | Pairs of precise decimal numbers per pixel | Any program that can open TIFF image files |
| Level-1B (GRI) | `.tif` | One simple whole number per pixel | **QGIS**, **SNAP**, **Midas**, or similar |
| Level-2 (SRI) | `.tif` | One simple whole number per pixel | **QGIS**, **SNAP**, **Midas**, or similar |

What are these programs, in plain terms?

- **QGIS** — A free, very popular mapping program (a "Geographic Information System," or GIS). It's used worldwide by everyone from city planners to scientists to view, overlay, and analyze any kind of map-based data, including radar images like these.
- **SNAP** — A free tool made specifically by the European Space Agency for working with satellite/radar imagery. If QGIS is a general-purpose map viewer, SNAP is more like a specialist tool built specifically for the kind of radar data this manual is about.
- **Midas** — Another image-analysis tool mentioned as an option; works similarly to the above for viewing/analyzing this kind of imagery.

You generally **cannot** just double-click these files and expect Windows Photo Viewer or a basic image app to show anything meaningful (especially for Level 1A's complex pixel format) — you need one of the specialized tools above, particularly for the GRI/SRI map-ready images.

---

## 9. The Conclusion, in Plain English (Section "Conclusion")

What the manual's conclusion actually says, de-jargonized:

- This manual gave you just the **overview** — a "big picture" map of how the data is organized. It is **not** the deep technical rulebook (that's a separate document, see Section 11).
- The radar started taking pictures of the Moon in **September 2019**.
- It worked continuously for about **3 months**, flying in what's called a **"Dawn-Dusk orbit."**
  - **What's a Dawn-Dusk orbit?** It's a specific type of orbit path chosen so the spacecraft always flies roughly along the boundary line between the Moon's "daytime" side and "nighttime" side (like always flying along sunrise/sunset). This keeps the lighting/imaging conditions very consistent for every single image taken, instead of varying randomly orbit to orbit.
- All the data beamed down from the spacecraft was first processed by **ISSDC** (ISRO Space Science Data Centre — basically ISRO's "data receiving and processing department").
- After processing, files were organized following the PDS4 rules we discussed and stored in what's called an **"active archive"** (a constantly updated, ready-to-add-more-to storage system).
- Every so often, a complete snapshot of everything collected so far was bundled up into something called a **"Long Term Archive" (LTA)** — think of it like periodically burning a complete backup DVD of everything collected up to that point, meant to be a stable, permanent, "final" copy.
- These LTA snapshots were then made available to the public/researchers through a **"Chandrayaan-2 Map-based browse application"** — essentially a website/tool where you can browse the Moon on a map and click to access the actual radar data for any area, hosted by **ISSDC** and **ISTRAC** (ISRO's satellite tracking and ground-station network).
- For the full nitty-gritty technical details (exact filename rules, byte-level formats, etc.), the manual points you to a separate, more detailed reference document.

---

## 10. Glossary — Every Abbreviation Used in This Document, Explained

| Term | What it stands for | What it means in plain English |
|---|---|---|
| **ISSDC** | ISRO Space Science Data Center | The ISRO department/facility that receives and processes data sent down from spacecraft |
| **SLC** | Single Look Complex Image | A radar image from one single pass, where each pixel keeps 2 numbers (I and Q) instead of 1 |
| **GRI** | Ground Reference Image | A radar image corrected so distances reflect actual ground distance, not radar-angle distance |
| **SRI** | Seleno Reference Image | A radar image properly aligned onto an actual Moon map ("Seleno-" = Moon-related) |
| **PDS** | Planetary Data System | The international standard rulebook for organizing/labeling space science data |
| **ebp** | Earth Bound Phase | The period when the spacecraft was still near Earth, before heading to the Moon |
| **nop** | Normal Operation Phase | The period once the spacecraft was actively doing its main science work at the Moon |
| **DFSAR** | Dual Frequency Synthetic Aperture Radar | The radar instrument this whole manual is about |
| **ISTRAC** | ISRO Telemetry Tracking and Command Network | ISRO's network of ground stations that talk to and track its spacecraft |
| **MDPD** | Microwave Data Processing Division | The specific ISRO team that processes this radar's data |
| **SIPG** | Signal and Image Processing Group | The broader ISRO group that MDPD belongs to |
| **LTA** | Long Term Archive | A stable, "final" backup snapshot of all collected data, released periodically |
| **XML** | eXtensible Markup Language | The structured text format used for all the "description card" label files |
| **OAT** | Orbit Attitude | Data describing the spacecraft's exact position (orbit) and orientation/tilt (attitude) |

### Extra codes seen in the figures (explained even though not in the manual's official glossary table)

| Code | Meaning |
|---|---|
| isda | ISRO Science Data Archive (the very top folder of everything) |
| cho | Chandrayaan-2 **O**rbiter |
| chl | Chandrayaan-2 **L**ander |
| chr | Chandrayaan-2 **R**over |
| tmc | Terrain Mapping Camera |
| lir | Imaging IR Spectrometer |
| ohr | Orbiter High Resolution camera |
| sar | Synthetic Aperture Radar (our instrument) |
| cla | Large Area Soft X-ray Spectrometer |
| xsm | Solar X-ray Monitor |
| cha | Atmospheric Compositional Explorer-2 |
| frs | Dual Frequency Radio Science Experiment |
| apx | Alpha Particle X-ray Spectrometer (on the Rover) |
| lib | Laser Induced Breakdown Spectroscope (on the Rover) |

### Other plain-English terms explained in this guide (quick lookup)

| Term | Quick plain-English meaning |
|---|---|
| Radar | Sends radio waves out, listens for the echo, to "see" things without needing light |
| Synthetic Aperture | A math trick that simulates a giant antenna by combining many readings taken while moving |
| L-band / S-band | Two different specific ranges of radio frequency the radar can use |
| Polarization (H/V/L/R) | The "direction" a radio wave vibrates in — different directions reveal different surface details |
| I/Q (In-phase/Quadrature) | The 2-number pair needed to fully describe a radar wave's strength and timing |
| Slant range | Distance measured at an angle, along the radar's direct line of sight |
| Ground range | The same distance, but mathematically corrected to be the true straight-down ground distance |
| Map-projected | An image correctly placed and aligned onto a real coordinate map |
| Binary file | A file whose raw bytes only make sense to specific software, not human-readable as plain text |
| CSV file | A simple spreadsheet-style text file, columns separated by commas |
| TIFF | A common, high-quality image file format |
| Bundle / Collection | PDS4's words for "a folder containing a group of related files" |
| Label file | The XML "description card" that always accompanies a PDS4 data file |
| LID (Logical Identifier) | A permanent unique ID/barcode for one specific data file |
| Frame | One chunk/segment of a continuous stream of incoming data |
| PRF (Pulse Repetition Frequency) | How frequently the radar fires new pulses |
| BAQ compression | A space-saving compression trick used before sending data from the Moon to Earth |
| Dawn-Dusk orbit | An orbit path that keeps the spacecraft flying along the day/night boundary line for consistent lighting |

---

## 11. References (for anyone who wants the deeper technical details)

1. **Data Product Software Interface Specification (DPSIS) and PDS-4 Archival, Chandrayaan-2 Dual Frequency SAR** — SAC/SIPG/MDPD/CH2/SAR/2018/04/03/v1.1
   *(This is the "full rulebook" companion document — go here if you need the exact, complete filename rules and byte-level technical specs, not just examples.)*
2. **"L- and S-band Polarimetric Synthetic Aperture Radar on Chandrayaan-2 mission"** — Space Applications Centre, Ahmedabad, published in *Current Science*, Vol. 118, No. 2, 25 January 2020, DOI: `10.18520/cs/v118/i2/226-233`
   *(This is the published scientific paper about the radar hardware itself — read this if you're curious about how the actual instrument was engineered, rather than just its data files.)*

---

## 12. One-Page Cheat Sheet (everything above, compressed)

```
WHAT IS DFSAR?
A radar "camera" on Chandrayaan-2 that images the Moon using two radio
frequencies, even in total darkness, using a math trick ("synthetic
aperture") to get sharp images without needing a giant antenna.

WHERE IS THE DATA STORED?
isda → ch2_bundle → cho_bundle (Orbiter) → ebp/nop (mission phase)
     → sar_collection
        → browse / data / geometry / spice_kernels
            → raw / calibrated / derived
                → yyyymmdd (the date folder)

WHAT ARE THE PROCESSING LEVELS?
L0  = Raw            → .dat → untouched, straight from the spacecraft
L1A = SLC / SLI       → .tif → signal converted to image, still at an angle ("slant range"), 2 numbers/pixel
L1B = GRI             → .tif → angle corrected to true ground distance, 1 number/pixel
L2  = SRI             → .tif → fully aligned to a real Moon map, ready to view, 1 number/pixel

RULE THAT NEVER CHANGES:
Every real data file always comes with a matching .xml "description card" file.

POLARIZATION MODES:
fp (Full Polarization)     → channels: HH, HV, VH, VV
cp (Circular Polarization) → channels: LH, LV

WHAT SOFTWARE OPENS THE IMAGES?
Raw (.dat)        → any generic binary viewer
SLC (.tif)        → any TIFF viewer
GRI / SRI (.tif)  → QGIS, SNAP, or Midas (specialized map/radar viewers)

MISSION TIMELINE:
Imaging began Sept 2019, ran ~3 months in a "Dawn-Dusk" orbit (consistent
lighting along the day/night line). Data processed at ISSDC, archived
under PDS4 rules, periodically snapshotted into a "Long Term Archive,"
and made publicly browsable via a map-based tool hosted at ISSDC/ISTRAC.
```
