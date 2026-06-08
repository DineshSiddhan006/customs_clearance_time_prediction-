# Use Case Documentation: Customs Clearance Time Prediction Engine (CR-001)

## 🏛️ Operational Scope & Domain Architecture
- **Operational Domain:** Customs, Regulatory Compliance, & Air Terminal Logistics
- **Prediction ID:** CR-001
- **Prediction Name:** Customs Clearance Time Prediction Engine (Air Cargo)
- **Deployment Hub:** King Abdulaziz International Airport (JED) – Jeddah, Saudi Arabia
- **Target Audience:** Operational Clearance Managers, Inbound Air Freight Dispatchers, and Customs Brokers
- **Business Purpose:** This engine runs real-time prescriptive risk audits on inbound cargo manifests before flight arrival. By identifying complex documentation mismatches, physical terminal congestion patterns, and compliance exceptions, it flags high-friction shipments early. This allows brokers to execute proactive document corrections, optimize warehouse labor schedules, and eliminate costly multi-day terminal demurrage fees.

---

## 📅 Dataset Chronological Timeline & Split Matrix
The dataset contains a total of **40,000 unique air cargo manifest records** distributed across a continuous 12-month window (March 2025 to February 2026 inclusive). To match real-world production deployment testing, the data is divided using a **Strict Temporal (Time-Based) Split Matrix**. This preserves the chronological sequence of supply chain events and prevents future target leakage from bleeding into the model.

- **`customs_clearance_train.csv` (28,000 rows):** Covers March 1, 2025 – November 30, 2025. Used to establish core baseline patterns, seasonal summer heat trends, and early lunar holiday spikes.
- **`customs_clearance_validation.csv` (6,000 rows):** Covers December 1, 2025 – December 31, 2025. Used for tuning hyper-parameters, checking early model drift, and validating regularization settings.
- **`customs_clearance_test.csv` (6,000 rows):** Covers January 1, 2026 – February 28, 2026. Used for final out-of-sample benchmarking, simulated model staging audits, and calculating real production metrics.

---

## 🎯 Target Variables Definition

### 1. `clearance_duration_hours`
- **Data Type:** Continuous Float
- **Operational Meaning:** The exact elapsed time delta calculated from the initial local FASAH system data submission timestamp to the final ZATCA electronic customs release order (Gate Pass) generation timestamp.
- **Strict Real-World Bounds:** Minimum: `0.5 Hours` (A perfect, fully automated Green Track server release). Maximum: `360.0 Hours` (The absolute 15-day statutory abandonment limit enforced across Saudi entry hubs. At 360 hours, unclaimed or blocked cargo is legally seized by the port authority and moved to the public customs auction yard).
- **Target Contribution Mechanics:** Acts as the primary continuous label. It scales dynamically based on a compound series of operational penalties (e.g., weekend cross-overs, late data submissions, regulatory document holds, and physical tarmac bottlenecks).

### 2. `inspection_track`
- **Data Type:** Categorical String (`"GREEN"`, `"YELLOW"`, `"RED"`)
- **Operational Meaning:** The formal risk tier assigned to the cargo manifest by ZATCA's automated risk engine upon entry routing.
  - `GREEN` (Automated Fast-Track Release): 0% human interaction. Target processing time: **0.5 to 3.0 hours**.
  - `YELLOW` (Manual Document Audit Loop): Requires a customs supervisor to manually open PDF scans, evaluate commercial invoice structures, and cross-reference value declarations. Target processing time: **12.0 to 48.0 hours**.
  - `RED` (Physical Yard Verification): Requires ground crews to tow containers across the tarmac to heavy X-ray scanning lanes, de-stuff pallets, and physically count cartons against the manifest. Target processing time: **48.0 to 240.0+ hours**.

---

## 📦 Feature Dictionary & Rigid Boundary Controls

### `shipment_id`
- **Data Type:** String (Unique Identifier / Non-Feature Element)
- **Format Shape:** `AAA-BBBBBBBB` (Strict IATA Master Air Waybill standard)
- **Description:** Stamped directly onto airframes and cargo sheets. `AAA` represents the 3-digit IATA airline operating prefix (e.g., `065` for Saudia Cargo, `160` for Cathay Pacific), followed by an 8-digit unique shipment serial number. 

### `submission_timestamp`
- **Data Type:** String (ISO 8601 Explicit Offset Format)
- **Format Shape:** `YYYY-MM-DDTHH:MM:SS+03:00`
- **Description:** The exact timestamp when the customs broker transmitted the cargo manifest to the FASAH gateway. Stored strictly in local Saudi Arabia Standard Time (AST / UTC+3) using the explicit offset format to prevent structural chronological misalignment. Minutes and seconds are cleanly bounded between `00` and `59`.

### `hs_code`
- **Data Type:** String (Categorical Identifier)
- **Format Shape:** 12-digit string sequence matching the Saudi Integrated Tariff System.
- **Description:** The legal Harmonized System commodity classification code. While it looks numerical, it functions as a high-cardinality categorical feature where the first 2 digits represent the broad industrial **Chapter** and the first 4 digits signify the specific product **Heading**.

### `shc_code`
- **Data Type:** Categorical String
- **Valid Dictionary Tiers:** `["GEN", "PER", "COL", "CRT", "VAL", "DG", "ELI"]`
- **Description:** The 3-letter IATA Special Handling Code assigned at the origin airport. Dictates terminal staging rules: `GEN` (General Cargo), `PER` (Perishables), `COL` (Cold Chain Freezer Storage), `CRT` (Controlled Room Temperature Pharmaceuticals), `VAL` (Valuable Cargo requiring armed escorts), `DG` (Dangerous Goods), and `ELI` (Lithium Batteries).

### `origin_country`
- **Data Type:** String (Categorical)
- **Format Shape:** 3-letter ISO Alpha-3 Code (e.g., `IND`, `DEU`, `NLD`, `FRA`, `TWN`, `CHN`, `ARE`).
- **Description:** The country of legal origin where the cargo was manufactured, grown, or certified. Dictates import tariffs, trade bans, and specialized partner government agency inspection protocols.

### `port_of_loading`
- **Data Type:** String (Categorical)
- **Format Shape:** 3-letter IATA Airport Code (e.g., `BOM`, `FRA`, `AMS`, `CDG`, `TPE`, `PVG`, `DXB`, `SIN`).
- **Description:** The specific global airport hub where the pallets were physically loaded onto the aircraft. Used by customs risk units to identify transit routing breaks or potential environmental cold-chain compromises.

### `importer_cr_id`
- **Data Type:** String (Categorical Reference Key)
- **Format Shape:** 10-digit text block.
- **Description:** The official Saudi Commercial Registration (CR) number of the importing entity. Adheres strictly to Saudi Ministry of Commerce guidelines by beginning exclusively with the legal operational sector prefixes: `1`, `2`, or `4`. The dataset contains exactly 30 unique enterprise values representing high-volume core trading entities.

### `is_aeo_certified`
- **Data Type:** Binary Integer (`0` or `1`)
- **Description:** Flag indicating if the importer holds Authorized Economic Operator trusted status under ZATCA’s Gold Tier framework. Certified entities (`1`) bypass standard documentation bottlenecks, while non-certified traders (`0`) default to baseline vetting loops.

### `historical_avg_clearance_hours`
- **Data Type:** Continuous Float
- **Strict Bounds:** Minimum: `1.5 Hours` | Maximum: `168.0 Hours` (1 week)
- **Description:** A rolling 90-day base rate calculating the mean clearance time for this specific importer's previous cargo entries. Serves as a vital anchor feature for the model to establish baseline trader behavior.

### `fatoorah_validation_passed`
- **Data Type:** Binary Integer (`0` or `1`)
- **Description:** Indicates whether the broker submitted a cryptographically signed Phase 2 ZATCA XML e-invoice payload (`1`). A value of `0` denotes legacy PDF scans or manual data entries that require human intervention.

### `weight_value_discrepancy`
- **Data Type:** Continuous Float (Percentage Ratio)
- **Strict Bounds:** Minimum: `0.00` (Perfect Match) | Maximum: `0.10` (10.0% structural cap)
- **Description:** The percentage delta between the airline’s physical weight scales at takeoff and the weight stated on the commercial invoice. A discrepancy above 5% (`0.05`) triggers immediate anti-smuggling and valuation fraud checks.

### `pre_arrival_filing_hours`
- **Data Type:** Continuous Float
- **Strict Bounds:** Minimum: `-48.0` Hours (Late, post-landing filing) | Maximum: `+72.0` Hours (Advance filing ceiling)
- **Description:** The advance time window between manifest data submission and the aircraft landing at JED. A negative value represents an operational violation where the flight landed before paperwork was submitted.

### `ambient_temperature_celsius`
- **Data Type:** Continuous Float
- **Strict Bounds:** Minimum: `12.0°C` (Winter night) | Max: `50.0°C` (Peak summer afternoon)
- **Description:** The actual ground temperature recorded on the JED cargo tarmac at the time of aircraft arrival.

### `visibility_meters`
- **Data Type:** Continuous Float
- **Strict Bounds:** Minimum: `100.0m` (Severe sandstorm or coastal fog) | Maximum: `10,000.0m+` (Clear sky baseline)
- **Description:** The horizontal runway visibility recorded by airport RVR automated sensors upon landing.

---

## 🏛️ Injected Real-World Operational Rules & Feature Co-dependencies

To ensure the model learns true aviation logistics logic rather than superficial mathematical relationships, the dataset enforces the following multi-variable dependency matrices:

### 1. Geographic Sourcing & Trade Route Co-dependencies
Cargo movements follow authentic global supply chain lanes. Features work in coordinated combinations based on the origin port:
- **The Cool Chain / Agro Lane:** If `port_of_loading` is `AMS` (Amsterdam) or `CDG` (Paris), `origin_country` must align to `NLD` or `FRA`. The `hs_code` must start with Chapter `0406` (Cheese/Dairy) or Heading `0602` (Live plants), and the `shc_code` is forced to `PER` or `COL`.
- **The Critical Pharma Lane:** If `port_of_loading` is `BOM` (Mumbai) or `FRA` (Frankfurt), `origin_country` must align to `IND` or `DEU`. The `hs_code` must start with Chapter `3004` (Medicaments), and the `shc_code` is forced to `CRT` or `COL`.
- **The Tech / Electronics Lane:** If `port_of_loading` is `PVG` (Shanghai) or `TPE` (Taipei), `origin_country` must align to `CHN` or `TWN`. The `hs_code` must start with Chapter `8517` (Smartphones/Telecom equipment), and the `shc_code` is forced to `GEN` or `VAL`.

### 2. Document Mismatch & Safety Enforcement Penalty
If a shipment's legal `hs_code` represents perishable food/medicine (Chapters `0406` or `3004`) but its `shc_code` is misdeclared as `GEN` (General), the system flags this as a documentation anomaly. The `inspection_track` is automatically forced to `RED` (Physical Yard Verification) and an automatic baseline penalty of **+72.0 hours** is added to `clearance_duration_hours`. This simulates the physical isolation of the cargo and secondary laboratory testing by the **Saudi Food & Drug Authority (SFDA)**.

### 3. Saudi Public Holiday Operational Shutdowns (2025-2026 Calendar)
If a manifest's `submission_timestamp` falls on an official Saudi holiday, the `clearance_duration_hours` scales up by a **6x to 10x multiplier**, and the `inspection_track` shifts toward manual `YELLOW` or `RED` paths. This accurately models the processing drops caused by regulatory skeleton staffing across the following exact dates:
- **Fixed Solar Holidays:** Saudi Founding Day (February 22) and Saudi National Day (September 23).
- **Moving Islamic Lunar Holidays:** Eid Al-Fitr window (March 30, 2025 – April 3, 2025) and Eid Al-Adha / Hajj Peak window (June 5, 2025 – June 9, 2025).

### 4. Weekday vs. Weekend Labor Shift Rules
The official Saudi public sector and customs administrative workweek runs strictly Sunday through Thursday; Friday and Saturday constitute the official weekend.
- If `submission_timestamp` falls on a Thursday after 14:00 AST (Pre-Weekend Drop Zone) or anywhere on Friday/Saturday, an automatic **+36.0 to +48.0 hour padding** is added to `clearance_duration_hours` for any cargo routed to manual `YELLOW` or `RED` tracks.
- Automated `GREEN` track entries bypass this penalty entirely, clearing via server automation within their standard 0.5 to 3.0-hour window.

### 5. ZATCA Advance Cargo Declaration (ACD) Enforcement
- **Early Compliance:** If `pre_arrival_filing_hours` >= 48.0, the shipment has an 85% probability of routing to the `GREEN` track, yielding a 0.5 to 2.0-hour clearance window (provided `is_aeo_certified == 1` and `fatoorah_validation_passed == 1`).
- **Post-Arrival Penalty:** If `pre_arrival_filing_hours` is negative (< 0.0), indicating post-landing filing, `GREEN` track probability drops to 0%. The shipment is forced into `YELLOW` or `RED` tracks and receives an automatic **+24.0 hour penalty** for manifest isolation in secondary holding zones.

### 6. Phase 2 Fatoorah E-Invoicing Compliance Mandate
If `fatoorah_validation_passed == 0` (Failed cryptographic check), the shipment is blocked from the automated `GREEN` track. It is forced into a `YELLOW` manual documentary review loop to verify its pricing structures, setting an absolute minimum processing floor of **18.0 hours**.

### 7. Manifest Weight Integrity Controls
If `weight_value_discrepancy` > 0.05 (Greater than a 5% delta between manifest scale metrics and invoice documentation), the system triggers valuation fraud alerts. The `inspection_track` is forced to `RED`, adding an automatic baseline processing penalty of **+48.0 hours** to account for physical container de-stuffing, manual box tallies, and terminal re-verification.

### 8. JED Runway Weather Physical Ground Halts
- **Dust/Sandstorm Ground Halt:** If `visibility_meters` < 1500.0, a flat **+12.0 hour operational delay** is applied across ALL tracks (including GREEN). This simulates airfield safety tower orders that halt tarmac towing equipment, container dollies, and forklift fleets, freezing cargo movement from aircraft to terminal gates.
- **Tarmac Thermal Shock Audits:** If `ambient_temperature_celsius` > 43.0°C and `shc_code` is `COL` or `PER`, `clearance_duration_hours` is increased by a **1.5x multiplier**. This represents mandatory physical inspections where SFDA officers pull and audit temperature-logging USB keys from containers to check for tarmac thermal spoilage.

---
