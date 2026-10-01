# ⚙️ Machining Tool Wear & Life Forecasting Intelligence Dashboard

An interactive, high-precision engineering dashboard for forecasting cutting tool life, simulating 3-stage flank wear ($VB$) progression, estimating Remaining Useful Life (RUL), and providing mathematically optimized strategies to minimize tool wear while preserving production throughput (Material Removal Rate - MRR).

Predictions and physics models are **grounded in peer-reviewed experimental literature published in *The International Journal of Advanced Manufacturing Technology (IJAMT)* by Springer Nature**.

---

## 🌟 Key Features

1. **Interactive Inputs**:
   - **Workpiece & Tool Pairings**: Titanium Ti-6Al-4V, Inconel 718, AISI 1045 Carbon Steel, AISI 4140 Alloy Steel, Hardened Bearing Steel (58-62 HRC), AISI 304 Stainless Steel, Aluminum 6061-T6, Grey Cast Iron GG25, Mild Steel 1018.
   - **Tool Materials & Coatings**: Submicron Tungsten Carbide, PVD TiAlN / AlCrN, Multi-layer CVD $Ti(C,N)/Al_2O_3$, Silicon Nitride & Whisker Ceramics, Polycrystalline Cubic Boron Nitride (PCBN), Polycrystalline Diamond (PCD), and M2 High-Speed Steel (HSS).
   - **Machine Dynamics**: 5-Axis High-Precision CNC, Heavy-Duty 3-Axis Box-Way VMC, CNC Slant-Bed Lathe, High-Speed Center (20k+ RPM), Standard VMC, and Conventional Manual Lathes/Mills.
   - **Cutting Parameters**: Cutting Speed ($V_c$), Feed Rate ($f$), Depth of Cut ($a_p$), and In-Service Elapsed Time.
   - **Case-Based Setup**: Name a case, select an operation and production priority, then set a case-specific maximum cutting speed. Speed is selectable from 0 (machine stopped) to that maximum.
   - **Cooling / Lubrication**: High-Pressure Coolant (70-100 bar), Flood Emulsion, MQL, Dry Air Blast, Cryogenic $CO_2/LN_2$.

2. **Forecasting & Graphical Analytics**:
   - **3-Stage Flank Wear ($VB$) Curve**: Simulates Stage I (Run-in), Stage II (Linear Steady-State), and Stage III (Tertiary Runaway) leading to the ISO 3685 failure threshold ($VB = 0.3\,\text{mm}$ or $0.6\,\text{mm}$ for roughing).
   - **Taylor $V-T$ Sensitivity Curve**: Demonstrates the exponential cliff where tool life collapses as speed rises ($V_c \cdot T^n = \text{const}$).
   - **2D / Contour Map**: Visualizes tool life response across simultaneous speed and feed changes.
   - **Real-Time Remaining Useful Life (RUL)**: Computes remaining minutes, remaining cutting distance (meters), and degradation status (Safe / Caution / Critical).

3. **Intelligent Wear Minimization Engine**:
   - **Productivity-Neutral Wear Optimization (The Golden Trade-off)**: Decreases cutting speed $V_c$ (which has the highest exponent $1/n$ in Taylor's law) and compensates by increasing depth of cut $a_p$ or feed $f$ to preserve 100% of MRR while achieving **80% to 200% tool life extension**.
   - **Lights-Out Endurance Mode**: Optimizes parameters for long unattended production runs with near-zero tool breakage risk.
   - **Pareto Trade-Off Frontier**: Interactive curve of Tool Life vs. Volumetric Productivity.
   - **Physics-Backed Guidance**: Actionable rules on speed leverage, chip thinning, tool coating suitability, and machine chatter mitigation.

4. **Research & Handbook Database Transparency**:
   - Research pairings use published empirical datasets; practical Springer industrial-handbook, Workshop Technology and HMT Production Technology records are visibly labelled as starting guidance.
   - Displays full source information (Title, Authors, Journal/handbook, Year, DOI or source link) and its evidence level.
   - Real-time **Prediction Confidence Index** auditing whether user parameters fall inside or outside the verified empirical research boundary.

5. **Scenario Comparison & Export**:
   - Side-by-side comparison of Current vs. Balanced vs. Max-Life vs. High-Efficiency scenarios.
   - 1-Click CSV report download.

---

## 🚀 Quick Start (Windows)

### Option 1: Double-Click Launcher
Simply double-click the included batch file:
```cmd
run_dashboard.bat
```

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
├── optimizer.py            # Multi-objective wear minimization engine & Pareto generator
├── presets.py              # 1-Click industry case studies (Aerospace, Auto, Energy, Die/Mold, etc.)
├── test_suite.py           # Automated test suite (41 validation checks)
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
