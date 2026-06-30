# Chandrayaan-2 OHRC Data Products — Explained for Complete Beginners

> This is a friendly, no-jargon walkthrough of ISRO's official document `ch2_ohrc_data_products_user_guide.pdf`. You do **not** need any space, engineering, or computer background to follow this. Every technical word is explained the first time it appears, usually with an everyday comparison.
>
> **A note on completeness:** The source PDF provided to me has 18 pages, and its content runs from the cover page through the start of **Section 6 ("Local Data Dictionary")**, ending mid-table. However, the document's own Table of Contents lists further sections — a **Conclusion** (page 20), **Annexure I: File Naming Conventions** (page 21), **Annexure II: Visualizing data with the PDS4 Viewer** (page 23), and **Annexure III: Team Contact Information** (page 29) — none of which were included in the 18 pages I was given. So everything below faithfully covers everything that **was** provided; the annexures/conclusion are flagged as "not available" rather than guessed at.

---

## 0. First, what is "OHRC"? (Start here)

Chandrayaan-2's orbiter carries 13 different science instruments. One of them is:

**OHRC = Orbiter High Resolution Camera**

In plain terms: **OHRC is basically the sharpest, most zoomed-in regular camera onboard the spacecraft.** Unlike the radar instrument (DFSAR, covered in a separate guide) which uses radio waves and works even in darkness, OHRC works like a normal camera — it needs sunlight to take a picture, and it produces a standard-looking photograph (in grayscale, not color — explained below), just extremely detailed.

- **"High Resolution"** means it can pick out very small details — fine enough to spot things like potential boulders or safe/unsafe patches of ground, which was useful for planning where the Chandrayaan-2 lander might touch down.
- The camera sensor is described as **"panchromatic" (PAN)**. This just means it captures **brightness only, across the whole visible light spectrum, as a single black-and-white channel** — like an old black-and-white photograph, rather than a color photo with separate red/green/blue channels. Many high-resolution satellite cameras use panchromatic sensors because skipping color capture lets them pack in much finer detail for the same data size.

**So in one sentence:** OHRC is Chandrayaan-2's super-sharp black-and-white camera, and this guide explains how the photos it takes are organized into files, folders, and formats for scientists to use.

This particular manual, just like the radar one, does **not** explain how the camera itself works — it only explains **how its output data is organized, named, and formatted**, so anyone downloading it from the internet knows what they're looking at.

---

## 1. The Document's "ID Card"

| Field | Value | Plain meaning |
|---|---|---|
| Report No. & Date | Chandrayaan-2/DP/SAC/SIPG/HRDPD/TR-27/November 2019 | A unique tracking code + the date, like a serial number |
| Title | Chandrayaan-2 OHRC PDS4 Data Products User Guide | The official name of this document |
| Number of Pages (original) | 29 | The full original report has 29 pages (only the first 18 were available to us — see note above) |
| Number of References | 2 | It cites 2 other documents |
| Type of Report | Technical | A factual/instructional document |
| Authors | Data Processing, Payload, Application Team | The mixed group of ISRO teams who wrote it |
| Originating Unit | High Resolution Data Processing Division (HRDPD), Signal & Image Processing Group (SIPG), Space Applications Centre (SAC) | The specific ISRO department in charge of OHRC's data |
| Abstract | Gives users OHRC's PDS4 data product info | One-line summary |
| Key Words | Archive, PDS4, ISDA | The main topics covered |
| Security Classification | **Restricted** | Unlike the SAR manual (which was "Unrestricted"/public), this one is restricted — meant for a defined audience, not freely public |
| Distribution | To identified individuals in ISRO/DOS, and users on the internet | Still reaches public internet users, just with some controlled distribution alongside |

**Document Change History:**

| Version | Date | What changed |
|---|---|---|
| V1.0 | Nov 21, 2019 | Initial draft — everything written for the first time |
| V2.0 | Nov 3, 2020 | Updated the **file naming conventions table** in Annexure I (note: Annexure I itself wasn't included in our 18 pages, but we know from the file structure examples elsewhere in the doc roughly what these filenames look like — see Section 6.3 below) |

**Acknowledgements:** Thanks given to SAC's Director (Shri DK Das), the SIPG group director (Shri Debajyoti Dhar), the HRDPD team members, and the PDS4 Working Group leadership (the team that maintains the international data-archiving standard this whole system is built on).

**Technical Content Approvals:** Reviewed by Amitabh (Data Processing Division, Chandrayaan-2 optical payloads) and Aditya Dagar (the **Principal Investigator**, i.e., the lead scientist in charge, for OHRC). Approved by TP Srinivasan (Head of HRDPD).

---

## 2. The Goal of This Manual

In plain terms: *"Explain, from a regular user's point of view, what data products OHRC produces — how many types exist, and what file formats they come in — so a researcher downloading this data knows what to do with it."*

**Audience:** This guide is meant for three groups:
1. Staff at the **archiving authority** (ISSDC, explained below) who manage the actual data storage.
2. The **data processing team and Principal Investigator (PI) team** at ISRO who created the instrument and process its data.
3. **Scientists/researchers** (in India and abroad) who want to actually use this Moon imagery for research.

---

## 3. The Bigger Picture: Chandrayaan-2's Entire Data Archive

Before zooming into OHRC specifically, the manual explains where OHRC fits into ISRO's *overall* data archive for the whole Chandrayaan-2 mission.

### 3.1 Mission Background (in plain English)

- Chandrayaan-2 (India's second Moon mission) was launched in **July 2019** by a rocket called **GSLV-MK-III** (India's most powerful rocket at the time — "GSLV" = Geosynchronous Satellite Launch Vehicle; "MK-III" = "Mark 3," its third major design version).
- The orbiter carries **13 different scientific instruments** (cameras, spectrometers, radar, etc.) and circles the Moon in a **100 km × 100 km circular orbit** (meaning it stays at a near-constant height of about 100 kilometers above the Moon's surface, in a roughly circular — not oval — path).
- All the data these 13 instruments collect gets sent down to Earth, where it is received, processed, and permanently stored at a facility called **ISSDC**.

> **ISSDC** = **Indian Space Data Centre** — basically ISRO's central facility for receiving, processing, and archiving all the data sent down by its science spacecraft, so it can later be given out to researchers.

- The goal is to make this archived data **easy to access over the internet**, using simple, standardized methods — similar to how other countries' space agencies (like NASA) share their planetary data.

### 3.2 PDS4 — The Universal Filing Standard (recap)

**PDS4 (Planetary Data System, version 4)** is the internationally agreed-upon "filing rulebook" for organizing and labeling space science data, used not just by ISRO but by NASA, ESA (Europe), and JAXA (Japan) as well — so that data from any of these agencies' missions can be understood the same way by any scientist worldwide, regardless of which country produced it. It's described in this manual as the **"de facto international standard"** — meaning it's the standard that has become the natural common choice across the field, even without being a strict legal requirement.

### 3.3 Which Instruments Have Which Data Types?

Not every one of Chandrayaan-2's 13 instruments produces the same range of data products. Here's the breakdown given in the manual:

| Instrument | Raw data? | Calibrated data? | Derived data? |
|---|---|---|---|
| TMC2 (Terrain Mapping Camera 2) | Yes | Yes | Yes |
| IIRS (Imaging InfraRed Spectrometer) | Yes | Yes | Yes |
| **OHRC** (our camera) | **Yes** | **Yes** | **No** |
| SAR (the radar instrument, covered in a separate guide) | Yes | Yes | No |
| CLASS (Chandrayaan-2 Large Area Soft X-ray Spectrometer) | Yes | No | No |
| XSM (Solar X-ray Monitor) | Yes | Yes | No |
| CHACE-2 (Atmospheric Compositional Explorer) | Yes | No | No |

**What does "Raw / Calibrated / Derived" mean again?** (Quick recap, explained fully with analogies in Section 4 below)
- **Raw** = almost untouched, straight from the instrument (like undeveloped camera film).
- **Calibrated** = cleaned up, corrected, and converted into a properly usable image (like a printed photograph).
- **Derived** = extra advanced products built *from* the calibrated data, like a map or a measurement table extracted from the image (like an annotated photo album page).

**Key takeaway for OHRC users:** you can get OHRC images at the **Raw** and **Calibrated** stages, but ISRO does **not** currently produce a further "Derived" product (such as elevation maps) from OHRC images — at least not through this particular archive pipeline.

### 3.4 Three Different "Level Numbering" Systems (and how they map to each other)

Here's something that can be confusing: there isn't just *one* way that processing levels get labeled — there are actually **three different naming systems** used side-by-side, and this manual provides a translation table between them:

1. **PDS4 Level** — the descriptive names we've been using: "Raw," "Calibrated," "Derived."
2. **ISRO Level** — ISRO's own internal numbering: 0, 1, 2.
3. **CODMAC Level** — an older, broader international scale used across planetary science generally (CODMAC = **Co**mmittee **o**n **D**ata **Ma**nagement and **C**omputation, a standards body). CODMAC levels run roughly from 1 (rawest) to 5+ (most processed); ISRO maps onto levels 2, 3, and 4+.

| PDS4 Level | What it actually means (full description from the manual, in plain terms) | ISRO Level | CODMAC Level |
|---|---|---|---|
| **Raw** | The original data from the instrument, after only some very basic initial housekeeping steps — "decompression" (undoing space-saving compression used during transmission), "reformatting" (rearranging the data into a standard file structure), and "packetization" (organizing the data into the small chunks, or "packets," that it was actually transmitted in). The final stored file follows the official PDS-approved archive format. | 0 | 2 |
| **Calibrated** | Data that has been **converted into real physical units** — meaning the values now represent an actual measurable physical quantity, not just raw instrument-specific numbers, **regardless of which exact instrument/camera was used to capture it**. Two specific corrections are mentioned: "radiometrically corrected" (pixel brightness values adjusted to be physically accurate — see the box below) and "Seleno-tagged" (each pixel is tagged with its actual Moon location — "Seleno-" = Moon-related, see earlier guide). | 1 | 3 |
| **Derived** | Results that have been further distilled/extracted from one or more *calibrated* data products — for example: maps, gravity/magnetic field measurements, or particle-size distributions (for instruments that study things like rings or dust). It also includes supporting reference data — such as calibration tables or "viewing geometry" tables (info about the exact angle/position the data was captured from) — used to help interpret the main data, if that supporting data doesn't cleanly fit any of the other 3 categories. Higher-level finished products, like "ortho" images (perfectly map-corrected images) or "DEM" (Digital Elevation Models — 3D height maps of terrain), also fall under "Derived." | 2 | 4 and above |

> **What does "radiometrically corrected" actually mean, in plain English?** Every individual camera sensor pixel can be very slightly more or less sensitive to light than its neighbors, due to tiny manufacturing differences. "Radiometric correction" is the process of mathematically adjusting each pixel's recorded value to cancel out these small sensor quirks, so the final number genuinely reflects how bright that spot on the Moon's surface actually was — rather than partly reflecting "quirks of this particular camera chip." This is what makes the data "instrument independent" — meaning, in theory, two different cameras observing the same spot under the same conditions should report the same corrected value.

---

## 4. How OHRC's Own Folder Is Organized (Archive Structure)

Just like the radar instrument's data (covered in a separate guide), OHRC's files live inside ISRO's giant shared archive system, nested under an instrument-specific folder. The manual calls this the **"instrument collection,"** and for OHRC, it's named `ohr_collection` (sometimes shown simply as `ohr` in diagrams) — note the folder is abbreviated **"ohr"** even though the instrument is called OHRC; this is just the internal short-code ISRO uses.

```
ohr
 ├── browse        → raw        → yyyymmdd/   → dataproduct  (quick preview images)
 │                 → calibrated → yyyymmdd/   → dataproduct
 │
 ├── data          → raw        → yyyymmdd/   → dataproduct  (the actual camera images)
 │                 → calibrated → yyyymmdd/   → dataproduct
 │
 ├── geometry      → calibrated → yyyymmdd/   → dataproduct  (location/position info matching the images)
 │
 ├── calibration   (calibration-related reference files)
 ├── miscellaneous (supporting non-PDS files — explained below)
 └── document      (documentation files)
```

This should look familiar if you've seen the structure used for other Chandrayaan-2 instruments:
- **browse** = small, lightweight preview images so you can glance at the data without opening a full-size file.
- **data** = the actual real image content.
- **geometry** = files telling you exactly which Moon location each part of the image corresponds to.
- **calibration / miscellaneous / document** = supporting folders for reference material, non-standard files, and written documentation respectively.
- **yyyymmdd** = the date the image was taken, in Year-Month-Day format.

One small but important difference from the radar archive: **OHRC's `geometry` folder only has a "calibrated" branch, not a "raw" one** — meaning location/geometry information is only formally provided once the image has been calibrated, not at the very raw stage.

---

## 5. Raw OHRC Data Products — What You Actually Get

### 5.1 What "Raw" Means Here

Raw OHRC data is the camera's image **before any corrections are applied**, but it does already include some basic positioning info — specifically, the **system-level corner coordinates** (a rough estimate of which Moon locations the four corners of the image cover, calculated automatically by the system rather than precisely refined by scientists yet).

For every single photo the camera takes ("payload observation"), the system automatically:
1. Generates the proper PDS4 files (figuring out the correct file name based on rules covered in Annexure I — not included in our pages, but illustrated in the examples below).
2. **Zips them together** into one compressed download file.
3. Files that zip away under the correct year/month/day folder.

### 5.2 A Real Example, Unzipped

The manual gives a real example filename a user might download:

```
ch2_ohr_nrp_20190906T224128571429200_d_img_gds.zip
```

When you unzip this file, here's what's inside (exactly as shown in the manual's Figure 2):

```
data
 └── raw
      └── 20190906/                                                          ← Sept 6, 2019
           ├── ch2_ohr_nrp_20190906T224128571429200_d_img_gds.img              ← the actual raw image
           └── ch2_ohr_nrp_20190906T224128571429200_d_img_gds.xml              ← its description card (PDS4 label)
browse
 └── raw
      └── 20190906/
           ├── ch2_ohr_nrp_20190906T224128571429200_b_brw_gds.png              ← a small preview thumbnail
           └── ch2_ohr_nrp_20190906T224128571429200_b_brw_gds.xml              ← its description card
miscellaneous
 └── raw
      └── 20190906/
           ├── ch2_ohr_nrp_20190906T224128571429200_d_img_gds.lbr              ← libration angle info
           ├── ch2_ohr_nrp_20190906T224128571429200_d_img_gds.oat              ← orbit & attitude info
           ├── ch2_ohr_nrp_20190906T224128571429200_d_img_gds.oath             ← orbit & attitude header info
           └── ch2_ohr_nrp_20190906T224128571429200_d_img_gds.spm              ← Sun parameter info
```

### 5.3 Every File, Explained One by One

| File | Format | What's actually inside (plain English) | What software opens it |
|---|---|---|---|
| **Data Raw Image** (`.img`) | Binary (raw computer bytes, not human-readable as text) | The actual camera image, completely unprocessed, stored as a generic binary blob. | Any binary image viewer — e.g., **ImageJ**, **ENVI**, **ERDAS** (specialized scientific image-analysis tools). You'll need to manually tell the viewer the image's width, height, and data type — these details are listed in the matching `.xml` label file, since a raw binary image file has no built-in way to describe its own dimensions. |
| **Data Raw Label** (`.xml`) | XML (structured text) | The "description card" for the raw image — explains its size, format, and other details needed to read it correctly. | **PDS4 Viewer** (a dedicated tool made for reading PDS4 label files — see Section 8 below for more on tools) |
| **Browse Raw Image** (`.png`) | PNG (a common, everyday image format — the same kind used for regular web images) | A smaller, "sub-sampled" (lower-resolution) quick-look version of the raw image, just for getting a fast preview. | **Any normal image viewer** — since PNG is a standard format, you don't even need the label file to view it (unlike the raw `.img` file above). |
| **Browse Raw Label** (`.xml`) | XML | Description card for the browse/preview PNG image above. | Not essential to open the PNG, but useful if you want full metadata — use a PDS4 Viewer. |
| **Miscellaneous Data** (`.lbr`, `.oat`, `.oath`, `.spm`) | Plain text | Four separate supporting files (these are **not** official PDS4-labeled products — hence "non-PDS data"): <br>• **`.oat`** = **Orbit & Attitude** — where the spacecraft was, and which way it was tilted/pointed, at the moment of capture.<br>• **`.oath`** = **Orbit & Attitude header** — a header/summary accompanying the OAT data.<br>• **`.lbr`** = **Libration angle** — see explanation box below.<br>• **`.spm`** = **Sun parameter file** — info about the Sun's position/angle during capture (important since OHRC, unlike radar, depends entirely on sunlight to take a picture). | Any plain text viewer/editor (e.g., Notepad). |

> **What is "libration"?** Even though we always see roughly "the same side" of the Moon from Earth, the Moon actually wobbles very slightly back and forth as it orbits — this slow wobble is called **libration**. For a camera in lunar orbit, knowing the precise libration angle at the time of a photo helps scientists work out the *exact* viewing geometry (the precise angle the camera was looking at the surface from), which matters for stitching together accurate maps.

---

## 6. Calibrated OHRC Data Products — The Corrected, Ready-to-Use Version

### 6.1 What Changed from Raw

Calibrated data takes the raw image and applies the **radiometric correction** described earlier (fixing up brightness values so they reflect true surface conditions, not sensor quirks), plus provides **refined corner coordinates** — meaning the "which Moon location does this image cover" estimate is now more precise/accurate than the rough system-level estimate given at the raw stage.

### 6.2 A Real Example, Unzipped

Example filename: `ch2_ohr_ncp_20190906T224128571429200_d_img_gds.zip`

```
data
 └── calibrated
      └── 20190906/
           ├── ch2_ohr_ncp_20190906T224128571429200_d_img_gds.img      ← the corrected image
           └── ch2_ohr_ncp_20190906T224128571429200_d_img_gds.xml      ← its description card
browse
 └── calibrated
      └── 20190906/
           ├── ch2_ohr_ncp_20190906T224128571429200_b_brw_gds.png      ← preview thumbnail
           └── ch2_ohr_ncp_20190906T224128571429200_b_brw_gds.xml      ← its description card
geometry
 └── calibrated
      └── 20190906/
           ├── ch2_ohr_ncp_20190906T224128571429200_g_grd_gds.csv      ← location grid file
           └── ch2_ohr_ncp_20190906T224128571429200_g_grd_gds.xml      ← its description card
miscellaneous
 └── raw
      └── 20190906/
           ├── ch2_ohr_ncp_20190906T224128571429200_d_img_gds.lbr
           ├── ch2_ohr_ncp_20190906T224128571429200_d_img_gds.oat
           ├── ch2_ohr_ncp_20190906T224128571429200_d_img_gds.oath
           └── ch2_ohr_ncp_20190906T224128571429200_d_img_gds.spm
```

(Notice: the supporting "miscellaneous" files are still filed under a folder literally called `raw` even within the calibrated product — this appears to simply be how ISRO's system organizes these particular support files regardless of the main image's processing level; the manual doesn't elaborate further on why.)

### 6.3 Every File, Explained One by One

| File | Format | What's actually inside | What software opens it |
|---|---|---|---|
| **Data Calibrated Image** (`.img`) | Binary | The radiometrically corrected image, in generic binary format. | Any binary image viewer (ImageJ, ENVI, ERDAS) — again, you'll need the image width/height/data type from the label file. |
| **Data Calibrated Label** (`.xml`) | XML | Description card for the calibrated image. | PDS4 Viewer |
| **Browse Calibrated Image** (`.png`) | PNG | A smaller, quick-look preview of the calibrated image. | Any normal image viewer |
| **Browse Calibrated Label** (`.xml`) | XML | Description card for the preview image. | PDS4 Viewer (optional, since PNG works standalone) |
| **Geometric Calibrated Grid** (`.csv`) | CSV (simple spreadsheet-style text file) | A table with **four columns: longitude, latitude, scan, and pix** — for selected points, this tells you exactly which Moon coordinate (lon/lat) corresponds to which pixel position in the image ("scan" and "pix" are just the row and column position of that pixel within the image — "scan line" and "pixel" being the standard satellite-imaging terms for "row" and "column"). | Any text editor, or even **Microsoft Excel** — since it's a plain CSV spreadsheet file. |
| **Geometric Calibrated Grid Label** (`.xml`) | XML | Description card for the geometry grid file. | PDS4 Viewer |
| **Miscellaneous Data** (`.lbr`, `.oat`, `.oath`, `.spm`) | Text | Same four supporting files as in the raw product (libration angle, orbit/attitude, orbit/attitude header, Sun parameters). | Any text viewer |

### 6.4 Quick Side-by-Side: Raw vs. Calibrated

| | Raw | Calibrated |
|---|---|---|
| Corner coordinates | Rough, system-estimated | Refined/more accurate |
| Pixel values | Untouched sensor readings | Radiometrically corrected (true brightness) |
| Has a `geometry`/grid file? | No | Yes (the lon/lat/scan/pix CSV grid) |
| Folders involved | `data`, `browse`, `miscellaneous` | `data`, `browse`, `geometry`, `miscellaneous` |

---

## 7. File Formats At a Glance (Section 5: "Archive Products Formats")

The manual summarizes everything into one compact table, specifically for OHRC's single sensor type — the **"high resolution panchromatic sensor"** (i.e., OHRC's one and only camera sensor, the black-and-white high-detail one described back in Section 0):

| Processing Level | Product Type | File Format | Data Type |
|---|---|---|---|
| Raw | Image | Binary | UnsignedByte |
| Raw | Browse (preview) | PNG | UnsignedByte |
| Calibrated | Image | Binary | UnsignedByte |
| Calibrated | Browse (preview) | PNG | UnsignedByte |
| Calibrated | Geometry | CSV | ASCII Text |

> **What is "UnsignedByte"?** The manual defines this directly: it's a **"Byte"** — specifically an **unsigned byte of 8-bit length**. In plain terms: each individual pixel's brightness value is stored using just 8 binary digits ("bits"), giving it a possible range of values from 0 to 255 (0 = pure black, 255 = pure white, with shades of gray in between) — this is exactly the same simple pixel format used in ordinary black-and-white digital photos. "Unsigned" just means the value can never be negative (there's no such thing as "negative brightness"). "ASCII Text" simply means ordinary readable text characters (letters, numbers, punctuation) — i.e., the CSV file is just plain readable text, not a binary format.

---

## 8. The "Local Data Dictionary" (Section 6) — ISRO's Custom Extension to PDS4

This section explains a more behind-the-scenes, technical concept — but here's what it means in plain terms:

- The international **PDS4** standard comes with its own built-in "dictionary" of standard terms/fields that any mission from any space agency can use (think of it like a shared, universal vocabulary list).
- However, every individual space mission also has its **own unique quirks and mission-specific details** that don't exist in that generic shared vocabulary — for example, a setting specific to how *this particular* camera's sensor was configured.
- PDS4 allows for this by letting agencies define a **Local Data Dictionary (LDD)** — essentially **a custom "extra vocabulary list" specific to ISRO's planetary missions**, built *on top of* the standard PDS4 dictionary, rather than replacing it.
- This isn't just for Chandrayaan-2 — it's designed to be **reused for ISRO's future planetary missions too**, so this vocabulary list doesn't need to be reinvented every time.

**Background context the manual gives:** ISRO actually first set up its own science data archive (called **ISDA — ISRO Science Data Archive**) back in **2008**, during the **Chandrayaan-1** mission (India's *first* Moon mission). Everything hosted in ISDA follows the PDS standard. For Chandrayaan-2 specifically, ISRO adopted the newer **PDS4** version of that standard — which the manual notes is also the version that NASA, ESA (Europe's space agency), and JAXA (Japan's space agency) are all using or moving toward for their own current and future missions. This means Chandrayaan-2's data is filed in a way that's broadly consistent with how the world's other major space agencies file their own planetary data.

### The Custom Fields ISRO Defined (Table 7 — partially available)

The manual includes a table of custom fields under a category called **`Mission_Area → Product_Parameters`**. Here is everything listed in our available pages (the table appeared to continue beyond where our provided pages ended, so this list may not be complete):

| Field name | What it means (plain English) |
|---|---|
| `job_id` | A unique tracking number automatically assigned by ISRO's internal "DP Scheduler" (Data Processing Scheduler — the system that manages processing jobs) when this particular data product is first ingested into the system. |
| `level0_dir_name` | The internal folder name used for this product's Level-0 (raw) input data, on ISRO's own processing systems. |
| `imaging_orbit_number` | The orbit count (i.e., "this was the Nth time the spacecraft went around the Moon") *during which the image was actually captured* — this number is assigned automatically onboard the spacecraft itself. |
| `dumping_orbit_number` | The orbit count *during which the data was actually transmitted down ("dumped") to a ground station* — note this can be a **different** orbit number than `imaging_orbit_number`, since the spacecraft can store images onboard for a while before getting a chance to transmit them down when it's in range of a ground station. |
| `line_exposure_duration` | How long the camera "integrated" (collected light for) each line/row of the image — essentially the camera's exposure time per row, similar in concept to a regular camera's shutter speed. |
| `bits_selection` | Which specific bits of the camera sensor's raw output get kept/selected for storage (can be `lsb`, `msb`, or `mid` — short for "least significant bits," "most significant bits," or "middle" bits). This relates to the camera sensor producing more raw detail per pixel than is ultimately transmitted/stored, so the system has to choose which portion of that detail to keep — based on the **TDI** settings (explained below). |
| `tdi_stages` | The number of "**Time Delay Integration**" stages used. **TDI** is a clever camera-sensor trick used by many high-resolution satellite cameras: instead of capturing a scene in one single quick snapshot, the sensor captures the *same* passing strip of ground multiple times in rapid succession as the spacecraft flies over it, perfectly timed to the spacecraft's motion, and then **adds all those captures together**. This boosts the signal strength and image quality significantly — similar to how taking several quick photos of a dim scene and digitally combining them can produce one much clearer, less noisy final image. "Stages" = how many of these repeated captures get added together. |
| `detector_pixel_width` | The physical size of a single pixel element on the camera's sensor chip. |
| `focal_length` | A property of the camera's lens that determines how "zoomed in" the camera is — a longer focal length means a narrower but more magnified field of view, similar to a telephoto lens on a regular camera. |
| `spacecraft_altitude` | How high above the Moon's surface the spacecraft was at the moment of capture. |
| `orbit_limb_direction` | Whether the spacecraft was moving "**Ascending**" or "**Descending**" along its orbit path at that moment — i.e., generally moving from south-to-north (ascending) or north-to-south (descending) relative to the Moon, which affects the lighting/geometry of the captured image. |
| `spacecraft_yaw_direction` | Describes the spacecraft's **yaw** orientation mode at the time. "Yaw" is one of the three ways a spacecraft can rotate (the other two being "pitch" and "roll") — specifically, rotation around the vertical axis, like a person standing still and turning to face a different direction without leaning. *(Our provided pages cut off mid-sentence right at this entry, so the exact possible values for this field weren't available to us.)* |

> **Why does any of this matter to a regular data user?** Mostly, it doesn't — these are deep, behind-the-scenes technical metadata fields aimed at very advanced users (e.g., someone trying to precisely recompute the exact geometry of an image, or trace exactly when/how a specific photo was taken and transmitted). Casual users browsing OHRC imagery for general research purposes will mostly just care about the actual **image file** and its basic **label**, not this deeper dictionary of internal tracking fields.

---

## 9. Sections Referenced But Not Available In Our Pages

For transparency, here is exactly what the document's own Table of Contents says comes *after* where our available pages end, which we cannot summarize since the content wasn't provided to us:

| Section | Page (per original doc) | What it likely covers (based on its title only) |
|---|---|---|
| 7. Conclusion | 20 | Probably a short wrap-up summary, similar in spirit to the radar manual's conclusion (mission timeline, archive process summary) |
| Annexure I: File Naming Conventions & Formats | 21 | The detailed, official rulebook for exactly what each segment of a filename like `ch2_ohr_nrp_20190906T224128571429200_d_img_gds.img` means — we could only infer some of this from context and examples in this guide (see the breakdown attempt below) |
| Annexure II: Visualize PDS4 Data Products using PDS4 Viewer | 23 | Step-by-step instructions for actually using the "PDS4 Viewer" tool mentioned repeatedly above |
| Annexure III: Team Contact Information | 29 | Contact details for the ISRO team responsible, in case users have questions |

### Best-Effort Filename Decoding (based on patterns observed in this guide, not the official Annexure I table)

Taking the example `ch2_ohr_nrp_20190906T224128571429200_d_img_gds.img`:

| Chunk | Best-effort meaning based on context |
|---|---|
| `ch2` | Chandrayaan-2 |
| `ohr` | OHRC instrument |
| `nrp` | An internal product/processing mode code — seen as `nrp` for raw products and `ncp` for calibrated products in the two examples in this guide, suggesting the 2nd letter may distinguish raw ("r") vs calibrated ("c") |
| `20190906T224128571429200` | Date and time of capture: Sept 6, 2019, at time `22:41:28.571429200` |
| `d` | A **d**ata file (vs. `b` for browse/preview, `g` for geometry, as seen in the folder examples) |
| `img` | Product type: **img**age (vs. `brw` for browse, `grd` for the geometry grid) |
| `gds` | An internal site/processing tag (also seen in the radar manual's filenames) |
| `.img` / `.xml` / `.png` / `.csv` | The actual file format/extension |

As with the radar manual, **this is an educated best-effort reading based on visible examples, not the authoritative rulebook** — the real, complete definition lives in Annexure I, which wasn't part of the pages available to us.

---

## 10. Tools Mentioned in This Guide (recap)

| Tool | What it is, in plain terms |
|---|---|
| **ImageJ** | A free, widely-used scientific image analysis program, originally popular in biology/microscopy research but commonly used for any kind of raw scientific imagery. |
| **ENVI** | A specialized (commercial) software package built specifically for analyzing remote-sensing/satellite imagery. |
| **ERDAS** | Another specialized (commercial) remote-sensing/satellite image analysis software package, similar in purpose to ENVI. |
| **PDS4 Viewer** | A dedicated tool (built for the PDS4 standard specifically) used to properly read the `.xml` label/description-card files and view the data products they describe, in the way the standard intends. Step-by-step usage instructions for this exact tool would have been in Annexure II, which wasn't available to us. |
| Any normal image viewer | For the everyday `.png` preview/browse images — no special software needed, any standard photo viewer works. |
| Any text editor / Microsoft Excel | For the plain-text/CSV files (geometry grid, orbit/attitude info, etc.) |

---

## 11. Glossary — Every Term Used in This Guide

| Term | Plain-English meaning |
|---|---|
| **OHRC** | Orbiter High Resolution Camera — Chandrayaan-2's sharpest, most zoomed-in regular (visible-light) camera |
| **Panchromatic (PAN)** | A camera sensor that captures brightness only, across the whole visible spectrum, as one black-and-white channel (no color) |
| **PDS / PDS4** | Planetary Data System — the international rulebook for organizing/labeling space science data, used by ISRO, NASA, ESA, and JAXA |
| **ISRO** | Indian Space Research Organization |
| **ISSDC** | Indian Space Data Centre — ISRO's facility for receiving, processing, and archiving spacecraft data |
| **ISDA** | ISRO Science Data Archive — the overall archive system, first set up in 2008 during Chandrayaan-1 |
| **SAC** | Space Applications Centre — the specific ISRO center (in Ahmedabad) responsible for this data processing |
| **DP** | Data Processing |
| **PI** | Principal Investigator — the lead scientist responsible for a given instrument |
| **HRDPD** | High Resolution Data Processing Division — the specific ISRO team responsible for OHRC's data |
| **SIPG** | Signal and Image Processing Group — the broader ISRO group HRDPD belongs to |
| **GSLV-MK-III** | The rocket that launched Chandrayaan-2 — India's most powerful launch vehicle at the time |
| **CODMAC** | Committee on Data Management and Computation — an older/broader international standard for ranking how processed a piece of space science data is (Levels 1 to 5+) |
| **Radiometric correction** | Adjusting pixel brightness values to cancel out individual sensor quirks, so the value reflects true real-world brightness |
| **Seleno-tagged** | Tagged with an actual Moon (lunar) location — "Seleno-" = Moon-related |
| **Corner coordinates** | The Moon location (lat/long) of the four corners of an image, telling you roughly what area it covers |
| **Libration** | The Moon's slight natural wobble as it orbits, even though it always shows roughly the same face to an observer |
| **Orbit & Attitude (OAT)** | Data describing a spacecraft's position (orbit) and orientation/tilt (attitude) at a given moment |
| **TDI (Time Delay Integration)** | A camera-sensor trick where the same passing scene is captured multiple times in quick succession and added together, to boost image quality |
| **Yaw / Pitch / Roll** | The three basic ways any vehicle (spacecraft, plane, ship) can rotate — yaw = turning left/right, pitch = tilting nose up/down, roll = tilting side to side |
| **Ascending / Descending (orbit)** | Whether a spacecraft, at a given moment, is moving generally south-to-north (ascending) or north-to-south (descending) along its orbit |
| **Local Data Dictionary (LDD)** | ISRO's custom extension to the standard PDS4 vocabulary, covering fields specific to ISRO's own missions |
| **Binary file** | A file whose raw bytes only make sense to software that knows the exact format — not readable as plain text |
| **CSV file** | A simple spreadsheet-style plain text file, with columns separated by commas |
| **PNG / TIFF** | Common image file formats (PNG is the everyday format used for most web images; TIFF, mentioned in the radar guide, is a higher-precision format often used for scientific imagery) |
| **XML** | eXtensible Markup Language — the structured text format used for all PDS4 "label"/description-card files |
| **UnsignedByte** | A pixel value stored using 8 bits, giving a range of 0–255, and which can never be negative |
| **Zip file** | A single compressed file that bundles multiple files together, to make downloading easier |

---

## 12. References (as listed in the manual)

1. **Chandrayaan-2 Science Data Management and Archive Plan** — Chandrayaan-2/DP/SAC/SIPG/HRDPD/TR-06/July 2018
   *(The overarching plan for how all of Chandrayaan-2's science data — not just OHRC's — gets managed and archived.)*
2. **Chandrayaan-2 SAC DP – ISSDC Interface for Archival** — Chandrayaan-2/DP/SAC/SIPG/HRDPD/TR-13/Dec 2018
   *(The technical document describing exactly how SAC's data processing systems hand data over to ISSDC for long-term archiving.)*

---

## 13. One-Page Cheat Sheet (everything above, compressed)

```
WHAT IS OHRC?
Chandrayaan-2's sharpest, most zoomed-in regular (black-and-white,
"panchromatic") camera — works only with sunlight, unlike the radar
instrument which works even in darkness.

WHERE DOES OHRC FIT IN THE BIGGER ARCHIVE?
Chandrayaan-2 has 13 instruments. Only some (TMC2, IIRS, OHRC, SAR, CLASS,
XSM, CHACE-2) are listed here. OHRC has Raw + Calibrated data, but NOT a
further "Derived" product (unlike TMC2/IIRS, which have all three).

THREE LEVEL-NAMING SYSTEMS THAT ALL MEAN THE SAME THING:
PDS4 "Raw"        = ISRO Level 0 = CODMAC Level 2
PDS4 "Calibrated" = ISRO Level 1 = CODMAC Level 3
PDS4 "Derived"    = ISRO Level 2 = CODMAC Level 4+

WHERE IS OHRC'S DATA STORED?
ohr_collection
  → browse / data / geometry / calibration / miscellaneous / document
      → raw / calibrated
          → yyyymmdd (date folder) → actual files

WHAT FILES DO YOU GET?
Raw:        .img (image) + .xml (label) + .png (preview) + .xml (label)
            + .lbr/.oat/.oath/.spm (orbit, libration, sun-angle info)
Calibrated: same as raw, PLUS a .csv geometry grid file (lon/lat/scan/pix)

FILE FORMAT SUMMARY:
Images (.img)   → Binary, UnsignedByte (0-255 grayscale, like a B&W photo)
Previews (.png) → standard PNG, UnsignedByte
Geometry (.csv) → plain ASCII text spreadsheet

WHAT SOFTWARE OPENS THESE?
.img        → ImageJ, ENVI, ERDAS (need width/height/datatype from .xml label)
.xml        → PDS4 Viewer
.png        → any normal image viewer
.csv/.oat/etc → any text editor or Excel

NOT COVERED IN OUR PAGES (referenced in the doc's contents, but not provided):
- Section 7: Conclusion
- Annexure I: official file naming convention rules
- Annexure II: how to actually use the PDS4 Viewer tool
- Annexure III: team contact details
```
