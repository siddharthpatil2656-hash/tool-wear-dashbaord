# ⚙️ Machining Tool Wear & Life Forecasting Intelligence Dashboard

An interactive, high-precision engineering dashboard for forecasting cutting tool life, simulating 3-stage flank wear ($VB$) progression, estimating Remaining Useful Life (RUL), and providing mathematically optimized strategies to minimize tool wear while preserving production throughput (Material Removal Rate - MRR).

The dashboard includes bundled Taylor-model records and an optional calibration workflow using your measured machining trials. Bundled coefficients are model inputs, not proof of production accuracy; verify citation metadata against publisher records before relying on a reference.

---

## 🌟 Key Features

1. **Interactive Inputs**:
   - **Workpiece & Tool Pairings**: Titanium Ti-6Al-4V, Inconel 718, AISI 1045 Carbon Steel, AISI 4140 Alloy Steel, Hardened Bearing Steel (58-62 HRC), AISI 304 Stainless Steel, Aluminum 6061-T6, Grey Cast Iron GG25, Mild Steel 1018.
   - **Tool Materials & Coatings**: Submicron Tungsten Carbide, PVD TiAlN / AlCrN, Multi-layer CVD $Ti(C,N)/Al_2O_3$, Silicon Nitride & Whisker Ceramics, Polycrystalline Cubic Boron Nitride (PCBN), Polycrystalline Diamond (PCD), and M2 High-Speed Steel (HSS).
   - **Machine Dynamics**: 5-Axis High-Precision CNC, Heavy-Duty 3-Axis Box-Way VMC, CNC Slant-Bed Lathe, High-Speed Center (20k+ RPM), Standard VMC, and Conventional Manual Lathes/Mills.
   - **Cutting Parameters**: Cutting Speed ($V_c$), Feed Rate ($f$), Depth of Cut ($a_p$), and In-Service Elapsed Time.
   - **Cooling / Lubrication**: High-Pressure Coolant (70-100 bar), Flood Emulsion, MQL, Dry Air Blast, Cryogenic $CO_2/LN_2$.

2. **Forecasting & Graphical Analytics**:
   - **3-Stage Flank Wear ($VB$) Curve**: Simulates Stage I (Run-in), Stage II (Linear Steady-State), and Stage III (Tertiary Runaway) leading to the selected failure threshold. The chart has a time-range slider, clear current/limit markers, hover detail, and resettable zoom.
   - **Taylor $V-T$ Sensitivity Curve**: Demonstrates the exponential cliff where tool life collapses as speed rises ($V_c \cdot T^n = \text{const}$).
   - **2D / Contour Map**: Visualizes tool life response across simultaneous speed and feed changes.
   - **Real-Time Remaining Useful Life (RUL)**: Computes remaining minutes, remaining cutting distance (meters), and degradation status (Safe / Caution / Critical).

3. **Intelligent Wear Minimization Engine**:
   - Balanced, maximum-life, and high-efficiency model estimates plus a bounded search for maximum predicted life at 80%, 100%, and 120% of current MRR.
   - The search uses active Taylor constants and pairing parameter boundaries, respects the selected machine envelope, rechecks candidate calculations and target MRR, and reports local sensitivity to ±2% input changes. These checks find calculation/range issues; they do not prove prediction accuracy.
   - **Maximum predicted life**: Model-screened candidate to increase predicted life; this does not estimate breakage probability and is not a basis for unattended operation.
   - **Pareto Trade-Off Frontier**: Interactive curve of Tool Life vs. Volumetric Productivity with hover, zoom, pan, and chart toolbar.
   - **Interactive analytics**: Switch the parameter contour between predicted tool life and calculated MRR; hover for parameter values and zoom/pan to inspect regions. Charts use a softer, theme-aware palette in light and dark mode, with readable labels, subtle gridlines, responsive sizing, and simplified pan/zoom/reset controls.
   - **Setup-Specific Guidance**: Operation, cutter, tool material/coating, machine rigidity, holder/overhang, and coolant records inform actionable checks.
   - Candidate gains are model estimates—not verified production recipes. Validate in controlled trials and check manufacturer, machine, workholding, and quality limits before production.
   - **Interactive What-if**: Adjust cutting speed, feed, and depth in the optimization tab to recalculate predicted tool life and MRR alongside the unchanged current forecast. Controls follow the active model limits and machine envelope; candidates outside the model range are flagged as extrapolations. The selected candidate is also included in CSV/PDF scenario exports.
   - **Cost efficiency and unit economics**: Estimate cost per part and plot cost against cutting time over a bounded cutting-speed sweep. Inputs include removed part volume, tool-edge cost, machine/operator hourly rates, and tool-change time. Setup, scrap, energy, coolant, and non-cutting logistics are excluded, so calibrate the inputs to the shop before using the model estimate.
   - **Failure-mode inspection guidance**: Select flank wear (VBmax), notch wear (VN), or crater wear (KT) and enter a validated VN/KT limit. The bundled Taylor life model still predicts VB only; a selected VN/KT value is an inspection criterion, not a non-flank-wear life prediction.
   - **G-code S/F preview**: Preview replacement of S/F words in one metric G01 block using the what-if settings. Requires confirmation of G21 and user-entered machine RPM/feed limits; rejects G20, G96, rapid/arc/canned-cycle blocks, and conflicting feed modes. It does not generate toolpaths or validate a controller; simulate and review the program before any machine use.
   - **Measured-trial-backed options**: With a valid setup-matched trial CSV uploaded, the optimization tab ranks observed tool-life runs against calculated MRR targets. This uses the local CSV and remains available offline; without matching trials, this section does not label model-generated plans as verified. Internet article metadata is for citations, not numerical machining outcomes.

### How model-screened and measured results differ
Model-screened suggestions use the dashboard's bundled Taylor coefficients (or a Taylor fit to the uploaded matching trials) and are shown separately from measured trial records. The search checks machine/model bounds, target-MRR agreement, finite outputs, repeat calculation consistency, and local sensitivity. A low sensitivity means only that small input changes have a small effect in this equation; it is **not** a statistical accuracy score. When calibrated, the leave-one-out error describes only held-out rows from that uploaded dataset and can still fail to represent future production. Validate proposed changes with repeated, controlled cuts and the same wear/failure criterion.

Online Crossref results and the offline citation library contain bibliographic metadata, not experimental cutting tables. Internet access may refresh citations, but it does not automatically improve or verify numeric optimization. The same bundled model and local calculations work offline.

### Maintain or extend the database
For future edits, change the source files in the project folder, not only a copy of the ZIP:

- **`springer_database.py`** is the source for bundled material/tool pairings, Taylor coefficients and parameter ranges, machines, coolants, operations, holders, tooling, and citation metadata. Keep dictionary keys unique; parameter units and valid ranges explicit; and record traceable source/provenance notes. Verify every DOI/title/author/year against the publisher or Crossref. Do not label a coefficient “verified” from citation metadata alone.
- **Prefer adding measured, setup-matched tool-life trials** via the calibration CSV instead of guessing new Taylor constants. Use consistent failure criteria and vary speed, feed, and depth independently. Calibration requires at least 8 matching runs and reports fit/leave-one-out diagnostics. The fit is a local empirical model, not a universal material constant.
- **`springer_library.py` / `springer_references.json`** manage imported citations for offline viewing; these records do not affect model calculations. `springer_references.json` is local and ignored by git, so back it up/export it separately if needed.
- After a code/database change, run `python test_suite.py` and `python -m unittest test_calibration test_crossref_evidence test_springer_library test_cost_model test_gcode_optimizer -v`, then `python -m py_compile app.py optimizer.py calibration.py springer_database.py tool_physics.py cost_model.py gcode_optimizer.py`. Update this README when the schema or workflow changes and rebuild `offline_dashboard_update.zip` from the source project files for offline PCs.

4. **Research, standards, and model evidence**:
   - Evidence tab includes Crossref discovery across journal publishers, a separate IEEE-publisher metadata search, plus links to Sandvik Coromant machining formulas/tool guidance, ISO, ASME, ASTM, the PHM Society CNC cutter wear dataset, and the CRC Press *Handbook of Advanced Ceramics Machining* record.
   - Standards are identified as test methods/catalog records, not as open prediction datasets.
   - Bundled citation metadata is checked for known DOI mismatches; the model-range score is explicitly heuristic, not statistical confidence.

5. **Scenario Comparison & Export**:
   - Side-by-side comparison of current, model-screened, measured-trial-backed, and interactive what-if scenarios.
   - CSV data export, a portrait PDF forecast report, and a one-page operator setup sheet with model/machine boundaries and a configurable planning interval.
   - Each input dropdown includes a **Reset to default** option.
   - Optional Taylor-model calibration from at least 8 measured tool-life runs matching the current material, machine, coolant, operation, tooling, holder, overhang, and roughing setup.

### Calibrate with measured trials
In the dashboard sidebar, expand **Calibrate with measured tool-life trials** and download the CSV template. Fill one row per completed trial, keeping its setup columns unchanged while varying cutting speed, feed, and depth independently. Use consistent tool-life failure criteria across runs, then upload the CSV. Only matching setup rows are fitted; the app reports both in-sample and leave-one-out fit statistics and limits the model's supported parameter range to the uploaded trials. A publisher citation or test standard alone does not recalibrate predictions.

### Live references and handbooks
The **Research, standards & evidence** tab searches Crossref's public journal metadata across publishers, including a separate publisher-filtered IEEE search, for records related to the selected workpiece/tool/operation. It requires an internet connection but no API key; responses are cached briefly. The tab also links to public Sandvik Coromant machining formulas and manufacturer tool guidance, the PHM Society's CNC cutter-wear sensor dataset, the CRC Press *Handbook of Advanced Ceramics Machining* book record, and the applicable ISO/ASME/ASTM catalogs. Some handbooks and standards require purchase or institutional access. These are discovery and engineering references, not all freely downloadable handbooks.

Citation metadata and manufacturer formulas generally do not expose comparable experimental rows for the exact tool grade, geometry, coolant, machine, wear criterion, and cut. Therefore the app does not infer Taylor coefficients or claim improved prediction accuracy from reference titles/abstracts, including IEEE records. Crossref references are not full text, and the PHM dataset is a specific sensor-based milling experiment rather than a universal machining table. Live references and the local offline citation library add traceable context only; they do not automatically alter the numeric model.

To improve setup-specific numeric accuracy, upload repeated measured tool-life trials with matching setup and consistent failure criteria. Keep a copy of the original records and validate fitted models on held-out/repeat runs. Do not paste proprietary handbook tables or article full text into the bundled database without permission; record bibliographic provenance and licensing, and manually validate any permitted values before modeling.

### Keep citations available offline
Export relevant citations from supported publishers or your library as RIS, BibTeX, or CSV metadata (follow licensing terms; do not bulk-download or redistribute article full text). In the dashboard's **Research, standards & evidence** tab, use **Choose a citation export** and select **Import and save references offline**. Imported citation records are stored locally in the ignored `springer_references.json` file beside `app.py` and displayed when offline; saved offline citations are not included in the forecast PDF. Transfer the citation export to the offline PC by USB before importing it there. The reference library is computer-local; export a CSV backup if moving/reinstalling the dashboard.

---

## 🚀 Quick Start (Windows)

### Option 1: Double-Click Launcher
From the updated project folder, double-click the included batch file. It starts
the `app.py` beside the launcher and prints the project folder it is using:
```cmd
run_dashboard.bat
```

To use recent GitHub changes in a local clone, check out the branch containing
those changes before starting the launcher. The dashboard sidebar shows the
full path of the `app.py` currently running.

### Run without internet
The dashboard calculations and reference data run locally. While online for
the initial setup, install Python and the required packages once:
```powershell
python -m pip install -r requirements.txt
```
After installation, disconnect from the internet and double-click
`run_dashboard.bat`, then open `http://localhost:8501` or the
`Dashboard (offline).url` shortcut. Keep the laptop on and the launcher
terminal open while using the dashboard. DOI links to external publications
require internet, but the dashboard, CSV export, and PDF report work offline.

### Option 2: Command Line
Open PowerShell or Command Prompt in this folder:
```powershell
python -m streamlit run app.py
```
Then open your browser to **`http://localhost:8501`**.

---

## 📁 Project Architecture

```
tool_wear_dashboard/
├── app.py                  # Main interactive Streamlit application with Plotly visual analytics
├── springer_database.py    # Springer IJAMT empirical database, Taylor constants, and paper metadata
├── tool_physics.py         # Physics engine: Extended Taylor equations, 3-stage wear, machine stiffness
├── optimizer.py            # Physics-guided strategies, Pareto curve, and bounded MRR search
├── presets.py              # 1-Click industry case studies (Aerospace, Auto, Energy, Die/Mold, etc.)
├── calibration.py          # Setup-matched Taylor model fit from measured tool-life trials
├── crossref_evidence.py    # Keyless Crossref search across journal publishers
├── springer_library.py     # Import and persist citation exports for offline use
├── cost_model.py           # Cost-per-part and speed-sweep unit economics
├── gcode_optimizer.py      # Guarded metric G01 S/F preview calculations
├── test_calibration.py     # Calibration validation tests
├── test_springer_library.py # Offline citation import/persistence tests
├── test_crossref_evidence.py # Live reference search client tests
├── test_cost_model.py      # Unit-economics model tests
├── test_gcode_optimizer.py # G-code S/F conversion and guard tests
├── test_suite.py           # Automated prediction, boundary, and optimizer checks
├── run_dashboard.bat       # Windows one-click launcher
└── README.md               # Documentation and engineering guide
```

---

## 🔬 Mathematical Formulations

### 1. Extended Taylor's Tool Life Equation
$$T_{\text{nominal}} = \left( \frac{C}{V_c \cdot f^x \cdot a_p^y} \right)^{1/n}$$

$$T_{\text{effective}} = T_{\text{nominal}} \cdot k_{\text{machine}} \cdot k_{\text{coolant}} \cdot k_{\text{mode}}$$

Where:
- $V_c$: Cutting speed ($\text{m/min}$)
- $f$: Feed rate ($\text{mm/rev}$ or $\text{mm/tooth}$)
- $a_p$: Axial depth of cut ($\text{mm}$)
- $n$: Taylor's exponent ($0.125$ for HSS up to $0.55$ for PCD)
- $x, y$: Feed and depth exponents ($x \approx 0.35-0.58$, $y \approx 0.15-0.28$)
- $k_{\text{machine}}$: Dynamic rigidity multiplier ($1.18$ for 5-Axis CNC down to $0.74$ for Manual Lathes)
- $k_{\text{coolant}}$: Heat dissipation multiplier ($1.35$ for High-Pressure Coolant, $1.45$ for Cryogenic)

### 2. Flank Wear ($VB$) Progression Model
$$VB(t) = VB_0 \cdot \left(1 - e^{-t/\tau_{\text{run-in}}}\right) + \alpha \cdot t + \beta \cdot \frac{e^{(t - t_3)/\tau_3} - 1}{e^{(T - t_3)/\tau_3} - 1}$$

---

## 📚 Primary Springer Literature References

1. **Ti-6Al-4V**: Pereira, O., Rodríguez, A., Barreiro, J., et al. (2022). *Comprehensive analysis of tool wear, tool life, surface roughness, costing and carbon emissions in turning Ti–6Al–4V titanium alloy: Cryogenic versus wet machining*. **The International Journal of Advanced Manufacturing Technology**, Vol. 121, pp. 4519–4537. DOI: 10.1007/s00170-022-09415-z.
2. **Inconel 718**: Altin, A., Nalbant, M., Taskesen, A. (2017). *Wear mechanisms during dry and wet turning of Inconel 718 with ceramic tools*. **The International Journal of Advanced Manufacturing Technology**, Vol. 91, pp. 2687–2698. DOI: 10.1007/s00170-016-9812-7.
3. **AISI 1045**: Suleiman, M., Abubakar, A., Bello, K. (2021). *Response surface methodology for tool wear and surface roughness modeling in turning AISI 1045 steel*. **The International Journal of Advanced Manufacturing Technology**, Vol. 115, pp. 3121–3135. DOI: 10.1007/s00170-021-07312-9.
4. **AISI 4140**: Kuntoğlu, M., Aslan, A., Pimenov, D. Y., et al. (2022). *Automated flank wear assessment and remaining tool life prediction in CNC turning of AISI 4140 steel*. **The International Journal of Advanced Manufacturing Technology**, Vol. 120, pp. 3779–3794. DOI: 10.1007/s00170-022-09012-0.
5. **Hardened Steel (PCBN)**: Kishore, K., Kumar, S., Patel, R. (2020). *Investigation on tool life and surface integrity in hard turning of hardened alloy steels with CBN and ceramic tools*. **The International Journal of Advanced Manufacturing Technology**, Vol. 109, pp. 1641–1655. DOI: 10.1007/s00170-020-05891-3.
6. **Stainless 304**: Singh, G., Aggarwal, V., Sharma, S. (2018). *Optimization of cutting parameters on flank wear and surface integrity in turning AISI 304 austenitic stainless steel*. **The International Journal of Advanced Manufacturing Technology**, Vol. 98, pp. 1953–1966. DOI: 10.1007/s00170-018-2194-2.
7. **Aluminum Alloys**: Gómez, M., Miguélez, M. H., Muñoz-Sánchez, A. (2019). *Tool wear mechanisms in high-speed machining of aluminum alloys with uncoated and PCD diamond tools*. **The International Journal of Advanced Manufacturing Technology**, Vol. 104, pp. 3201–3215. DOI: 10.1007/s00170-019-04105-z.
8. **Grey Cast Iron**: Ferreira, J. R., Coppini, N. L., Miranda, G. W. (2019). *Tool life and wear mechanisms in high-speed turning and milling of pearlitic gray cast iron with silicon nitride ceramics*. **The International Journal of Advanced Manufacturing Technology**, Vol. 102, pp. 1109–1121. DOI: 10.1007/s00170-019-03719-7.
