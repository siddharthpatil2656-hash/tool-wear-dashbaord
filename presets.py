"""
Industry Case Study Presets Calibrated from Springer IJAMT Research Papers.
Allows 1-click loading of realistic industrial machining scenarios into the dashboard.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class IndustryPreset:
    preset_name: str
    description: str
    pairing_key: str
    machine_name: str
    coolant_name: str
    vc: float
    feed: float
    ap: float
    current_time_min: float
    is_roughing: bool
    target_industry: str
    operation_name: str = "Turning (OD/ID Continuous Cut)"
    milling_tooling_name: Optional[str] = None


INDUSTRY_PRESETS: Dict[str, IndustryPreset] = {
    "Aerospace: Ti-6Al-4V Titanium Aircraft Bracket": IndustryPreset(
        preset_name="Aerospace: Ti-6Al-4V Titanium Aircraft Bracket",
        description="Finish milling of critical aircraft structural spar in Ti-6Al-4V alloy on a 5-Axis CNC center using high-pressure coolant and PVD (Al,Ti)N submicron carbide.",
        pairing_key="Ti-6Al-4V | PVD TiAlN Carbide",
        machine_name="5-Axis High-Precision CNC Machining Center",
        coolant_name="High-Pressure Coolant (70-100 bar)",
        vc=75.0,
        feed=0.12,
        ap=1.2,
        current_time_min=18.0,
        is_roughing=False,
        target_industry="Aerospace & Defense",
        operation_name="End Milling / Shoulder Milling",
        milling_tooling_name="Solid Carbide Square End Mill (4-Flute)"
    ),

    "Energy & Turbines: Inconel 718 High-Speed Gas Turbine Disk": IndustryPreset(
        preset_name="Energy & Turbines: Inconel 718 High-Speed Gas Turbine Disk",
        description="High-speed continuous profiling of aged Inconel 718 turbine disk using Silicon Nitride / Al2O3 whisker ceramic inserts under dry compressed air blast.",
        pairing_key="Inconel 718 | Ceramic (Si3N4 / Al2O3)",
        machine_name="CNC Slant-Bed Turning Center",
        coolant_name="Dry Machining (Compressed Air Blast)",
        vc=240.0,
        feed=0.14,
        ap=1.0,
        current_time_min=10.0,
        is_roughing=False,
        target_industry="Power Generation & Jet Propulsion"
    ),

    "Automotive: Al 6061-T6 High-Speed Transmission Casing": IndustryPreset(
        preset_name="Automotive: Al 6061-T6 High-Speed Transmission Casing",
        description="Ultra-high-speed face milling and pocketing of aluminum transmission housings using PCD tipped tooling on a 20,000+ RPM spindle center with MQL.",
        pairing_key="Al 6061-T6 | PCD Diamond",
        machine_name="High-Speed Spindle Center (HSM / 20k+ RPM)",
        coolant_name="Minimum Quantity Lubrication (MQL)",
        vc=950.0,
        feed=0.18,
        ap=1.5,
        current_time_min=45.0,
        is_roughing=False,
        target_industry="Automotive Powertrain",
        operation_name="Face Milling (Indexable 45°)",
        milling_tooling_name="PCD-Tipped End Mill (Diamond, 2-Flute)"
    ),

    "Heavy Machinery: AISI 1045 Drive Shaft Roughing": IndustryPreset(
        preset_name="Heavy Machinery: AISI 1045 Drive Shaft Roughing",
        description="Heavy rough turning of medium carbon steel drive shafts using multi-layer CVD Ti(C,N)/Al2O3 coated carbide on a heavy box-way CNC lathe.",
        pairing_key="AISI 1045 Steel | CVD Coated Carbide",
        machine_name="Heavy-Duty 3-Axis CNC VMC (Box Way)",
        coolant_name="Standard Flood Emulsion (7-10% oil)",
        vc=220.0,
        feed=0.28,
        ap=2.5,
        current_time_min=25.0,
        is_roughing=True,
        target_industry="Heavy Industrial Equipment"
    ),

    "Die & Mold: Hardened AISI 4140 Core Cavity Machining": IndustryPreset(
        preset_name="Die & Mold: Hardened AISI 4140 Core Cavity Machining",
        description="Finishing of pre-hardened injection mold core cavity in AISI 4140 (40 HRC) using micro-grain PVD TiAlN carbide on a standard VMC.",
        pairing_key="AISI 4140 Steel | PVD TiAlN Carbide",
        machine_name="Standard 3-Axis CNC VMC (Linear Guide)",
        coolant_name="Standard Flood Emulsion (7-10% oil)",
        vc=150.0,
        feed=0.14,
        ap=0.8,
        current_time_min=30.0,
        is_roughing=False,
        target_industry="Tool & Die / Injection Molding"
    ),

    "Educational Toolroom: Mild Steel 1018 Turning on Manual Lathe": IndustryPreset(
        preset_name="Educational Toolroom: Mild Steel 1018 Turning on Manual Lathe",
        description="General turning of low carbon mild steel using High-Speed Steel (HSS) toolbit on an engine lathe with flood coolant.",
        pairing_key="AISI 1018 Steel | High-Speed Steel (HSS)",
        machine_name="Conventional Manual Lathe / Knee Mill",
        coolant_name="Standard Flood Emulsion (7-10% oil)",
        vc=30.0,
        feed=0.15,
        ap=1.5,
        current_time_min=12.0,
        is_roughing=False,
        target_industry="General Workshop & Maintenance"
    ),

}
