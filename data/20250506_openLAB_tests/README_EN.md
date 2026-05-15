# Monitoring Data of the openLAB Research Bridge – Load Test on PE 2.1

This data set is available:
- Authors: Max Herbers, Bertram Richter, Maria Walker, Steffen Marx (TU Dresden)
- URL: https://opara.zih.tu-dresden.de/handle/123456789/1485
- DOI: https://doi.org/10.25532/OPARA-852
- License: CC-BY-SA 4.0

All position references relate to support axis 10 (south-west), x = 0.

The load was introduced into precast element (PE) 2.1 using two hydraulic jacks positioned centrally in the web (x = 24.3 m, corresponding to the region of maximum bending moment due to dead load).

## Measurement data

Overview of measurement files: (# - file name - maximum displacement u)

1.	MD_2025_05_06_09_08_25.txt		u_max = 5 mm
2.	MD_2025_05_06_10_43_20.txt		u_max = 10 mm
3.	MD_2025_05_06_12_05_10.txt		u_max = 20 mm
4.	MD_2025_05_06_13_43_17.txt		u_max = 30 mm
5.	MD_2025_05_06_16_07_15.txt		u_max = 40 mm
6.	MD_2025_05_06_17_39_40.txt		u_max = 50 mm
7.	MD_2025_05_06_18_30_51.txt		u_max = 60 mm

The tests were carried out on May 06, 2025. All times in UTC+00:00.

Overview of measurement channels

| Channel | Quantity and unit        | Sensor type                                      | Location / Comment                                                      |
|---------|--------------------------|--------------------------------------------------|-------------------------------------------------------------------------|
| 1       | Time [s]                 | —                                                | —                                                                       |
| 2       | Strain [µm/m]            | Strain gauge Althen, L = 120 mm                  | Next to prestressing tendon opening (Damage No. 3 on Day 3)             |
| 3       | Time [s]                 | —                                                | —                                                                       |
| 4       | Force [kN]               | Load cell HBK C6A, max. 500 kN                   | North, toward axis 30                                                   |
| 5       | Force [kN]               | Load cell HBK C6A, max. 500 kN                   | South, toward axis 10                                                   |
| 6       | Displacement [mm]        | Inductive displacement transducer WETA 1/10, HBK | Horizontal, next to strain gauge (channel 2); measurement length 160 mm |
| 7       | Temperature [�C]         | Thermocouple Type K                              | Inside web PE 2.3 (approx. midspan)                                     |
| 8       | Ambient temperature [°C] | Thermocouple Type K                              | 50 cm above pavement, suspended from FT 2.3 (approx. midspan)           |
| 9       | Time [s]                 | —                                                | —                                                                       |
| 10      | Displacement [mm]        | Laser distance sensor Baumer OM30-L0350.HV.YUN   | Web bottom center, PE 2.1, x = 7.79 m (midspan of Field 1)              |
| 11      | Displacement [mm]        | Laser distance sensor Baumer OM30-L0350.HV.YUN   | Web bottom center, PE 2.1, x = 15.00 m (axis 20)                        |
| 12      | Displacement [mm]        | Laser distance sensor Baumer OM30-L0350.HV.YUN   | Web bottom center, PE 2.1, x = 19.68 m                                  |
| 13      | Time [s]                 | —                                                | —                                                                       |
| 14      | Displacement [mm]        | Laser distance sensor Baumer OM30-L0350.HV.YUN   | Web bottom center, PE 2.1, x = 24.30 m (below load application)         |
| 15      | Displacement [mm]        | Laser distance sensor Baumer OM30-L0350.HV.YUN   | Web bottom center, PE 2.1, x = 30.00 m (axis 30)                        |
| 16      | —                        | —                                                | —                                                                       |
| 17      | Force [kN]               | Calculated sum of channels 4 and 5               | —                                                                       |

## Additional notes

- For the first five loading steps (up to u_max = 40 mm), the data were zeroed (tared) before the start of measurement.
- Subsequently, the target displacement (starting from zero) was applied.  
- For loading steps 6 and 7, no taring was performed.  
- The tare values from the initial loading steps are included in the measurement files (lines 28 and 29) and enable the calculation of absolute displacements.


## Python script

The included script is structured as follows:

1. Data import
2. Post-processing:
    During the load plateaus (constant displacement, u = 5�60 mm), additional measurements were carried out on the bridge. In some cases, the laser beam path was obstructed (e.g., in the load application area), resulting in implausible readings. These outliers were filtered as follows:
    - A measurement point was removed if
        1. the length change to the previous measurement exceeded 1 mm, and
        2. the total displacement exceeded 65 mm.
    - Additionally, a moving average filter with a window size of 5 was applied.
3. Plots
