## Monitoring Data of the openLAB Research Bridge (2024-02-01 to 2024-10-31) and building information

### Abstract

This dataset presents Structural Health Monitoring (SHM) data from the openLAB research bridge, a large-scale laboratory structure located in Bautzen, Germany. Following an initial one-year reference phase, the bridge will undergo a series of load tests designed to induce significant structural damage. This open-access dataset provides researchers with a rare opportunity to validate SHM methodologies under near-real-world conditions. The current publication includes data from the undamaged bridge, covering the period from 2024-02-01 to 2024-10-31. Additional repositories will be published periodically as new data become available.

The bridge is equipped with a comprehensive monitoring system featuring fiber optic and electrical sensors that capture both structural behavior and environmental conditions (e.g., air temperature, humidity, and solar radiation). In this initial release, data is sourced from an electrical Gantner Instruments measurement system (Q.station 101T, with various Q.bloxx modules). The dataset includes measurements of acceleration, tilt, air temperature, humidity, and solar radiation. Data is recorded continuously at 10-minute intervals, with additional triggered measurements during non-damaging load tests conducted with a test vehicle or in response to increased vibration activity. The repository provides the data in Comma Separated Values (CSV) format. Each file includes a header specifying the names of the data columns. Additional details, such as units and sampling frequency, are provided in this README file. Each CSV file contains a Timestamp column that records the time of each sample as a datetime string in ISO 8601 format, without time zone information. All timestamps are in Coordinated Universal Time (UTC). Sensor data is represented as decimal numbers.

The data is accompanied with structural plans of the bridge and the installed monitoring system.
This dataset is available at [10.25532/OPARA-660](https://doi.org/10.25532/OPARA-660).

### Data column naming convention

To ensure consistency across the various measurement systems and sensor types, a standardized naming convention has been established for the resulting data columns. Each 20-character column name encodes key information, including the measurement system, measurement type, structural component, and sensor installation location. Certain sensor products offer multiple measurement capabilities, such as simultaneously recording tilt and temperature. Consequently, data from these sensors is represented in multiple data columns. The full naming convention is illustrated below using the example of the data column G_ACCZ_PE11_CB_0750_0.

```
G_ACCZ_PE11_CB0750_0
~ ~~~~ ~~~~ ~~~~~~ ~
| |    |    | |    |
| |    |    | |    |
| |    |    | |    |- Block 6 (1-digit): Sensor replacement,
| |    |    | |    |   e.g., 0 = original sensor, 1 = first replacement
| |    |    | |
| |    |    | |- Block 5 (4-digits): Distance from component origin,
| |    |    | |   e.g., 0750 = 7.5 m
| |    |    |
| |    |    |- Block 4 (2-digits): Installation Location,
| |    |    |   e.g., BU = concrete, bottom edge
| |    |
| |    |- Block 3 (4-digits): Structural component,
| |        e.g., PE11 = precast element, span 1, girder 1
| |
| |- Block 2 (4-digits): Sensor type,
| |   e.g., ACCZ = acceleration in vertical direction
|
|- Block 1 (1-digit): Measurement system, e.g., G = Gantner Instruments
```

The options for each block in the naming scheme are listed in the following tables. Only the relevant options for the presented dataset are included. For example, since the dataset does not contain data from fiber optic sensors and exclusively uses the electrical Gantner Instruments measurement system, the first block of each column name is always represented by G. This block has been omitted from the tables for brevity.

| Block 2 | Measurement type                                |
| :------ | :---------------------------------------------- |
| ACCZ    | Acceleration in vertical direction              |
| HTST    | Hygro-thermo sensor/temperature                 |
| HTSH    | Hygro-thermo sensor/humidity                    |
| PYRS    | Pyranometer                                     |
| TILY    | Tiltmeter/tilt in longitudinal bridge direction |
| TILX    | Tiltmeter/tilt in transverse bridge direction   |
| TILT    | Tiltmeter/temperature                           |
| TILH    | Tiltmeter/humidity                              |

| Block 3 | Structural component            |
| :------ | :------------------------------ |
| PE11    | Precast element, span 1, girder 1 |
| PE12    | Precast element, span 1, girder 2 |
| PE13    | Precast element, span 1, girder 3 |
| PE21    | Precast element, span 2, girder 1 |
| PE22    | Precast element, span 2, girder 2 |
| PE23    | Precast element, span 2, girder 3 |
| ENVR    | Environment                     |

| Block 4 | Installation location |
| :------ | :-------------------- |
| CB      | Concrete, bottom edge |
| EN      | Environment           |

### Structure of the repository

The repository is organized into four data directories, along with an additional directory containing supplementary material. The contents of each directory are described in detail in the following sections.

---

#### Directory: 01_acceleration_trigger

The directory `01_acceleration_trigger` contains triggered acceleration measurements. As the openLAB bridge exhibits very low acceleration response to ambient excitation, a 70-second sample is collected whenever the bridge experiences increased vibration activity. Specifically, a measurement is triggered when the range of the signal from sensor G_ACCZ_PE11_CB0750_0 exceeds 2 × 10⁻⁴ m/s² within the considered time window. The trigger system was activated on 2024-05-01, and data is available for the period from 2024-05-01 to 2024-10-31.

**Files:** The directory contains 523 CSV files. Each file name includes the prefix _acc_ followed by the timestamp marking the start of the triggered measurement. The file naming convention is as follows:

```
acc_{year}_{month}_{day}_T{hour}_{minute}_{second}.csv
```

**Sensors:** PCB 393A03

**Sample rate:** 500 Hz

**Schema:**

| Column name          | Data type      | Description           |
| :------------------- | :------------- | :-------------------- |
| Timestamp            | Datetime       | ISO 8601 format (UTC) |
| G_ACCZ_PE11_CB0750_0 | Decimal Number | Acceleration in m/s²  |
| G_ACCZ_PE12_CB0750_0 | Decimal Number | Acceleration in m/s²  |
| G_ACCZ_PE13_CB0750_0 | Decimal Number | Acceleration in m/s²  |
| G_ACCZ_PE21_CB0750_0 | Decimal Number | Acceleration in m/s²  |
| G_ACCZ_PE22_CB0750_0 | Decimal Number | Acceleration in m/s²  |
| G_ACCZ_PE23_CB0750_0 | Decimal Number | Acceleration in m/s²  |

**Preprocessing:** The median of each measurement has been subtracted from the signals to correct for non-zero offsets from the piezoelectric sensors. Additionally, a fourth-order Butterworth filter with a low-frequency cutoff of 0.5 Hz and a high-frequency cutoff of 100 Hz has been applied. The low-frequency cutoff was selected based on the measurement range specified by the manufacturer, while the high-frequency cutoff serves as an anti-aliasing filter.

---

#### Directory: 02_environment

The `02_environment` directory contains measurements from sensors for air temperature, humidity, and solar radiation. These sensors are part of a climate station mounted on top of the bridge. Samples are collected in 10-minute intervals, with data available from 2024-02-01 to 2024-10-31.

**Files:** The directory contains 9 CSV files each containing the data for one month. The file naming convention is as follows:

```
environment_{year}_{month}.csv
```

**Sensors:** Thies Clima 1.1005.54.773, Kipp & Zonen SP Lite2

**Sample rate:** 1/600 Hz

**Schema:**

| Column name          | Data type      | Description             |
| :------------------- | :------------- | :---------------------- |
| Timestamp            | Datetime       | ISO 8601 format (UTC)   |
| G_HTST_ENVR_EN0000_0 | Decimal Number | Air temperature in °C   |
| G_HTSH_ENVR_EN0000_0 | Decimal Number | Relative humidity in %  |
| G_PYRS_ENVR_EN0000_0 | Decimal Number | Solar radiation in W/m² |

**Preprocessing:** No preprocessing steps have been applied.

---

#### Directory: 03_tiltmeter

Tiltmeter data is recorded every 10 minutes and can be found in the `03_tiltmeter` directory. Tilt is measured along the transverse bridge direction (TILY) and the longitudinal bridge direction (TILX). Additionally, each tiltmeter records air temperature and relative humidity. Data is available from 2024-02-01 to 2024-10-31.

**Files:** The directory contains 9 CSV files each containing the data for one month. The file naming convention is as follows:

```
tiltmeter_{year}_{month}.csv
```

**Sensors:** Sisgeo 0S542HD0502

**Sample rate:** 1/600 Hz

**Schema:**

| Column name          | Data type      | Description                                   |
| :------------------- | :------------- | :-------------------------------------------- |
| Timestamp            | Datetime       | ISO 8601 format (UTC)                         |
| G_TILY_PE11_CB1100_0 | Decimal Number | Tilt in mm/m along the transverse direction   |
| G_TILX_PE11_CB1100_0 | Decimal Number | Tilt in mm/m along the longitudinal direction |
| G_TILT_PE11_CB1100_0 | Decimal Number | Air temperature in °C                         |
| G_TILH_PE11_CB1100_0 | Decimal Number | Relative humidity in %                        |
| G_TILY_PE12_CB1100_0 | Decimal Number | Tilt in mm/m along the transverse direction   |
| G_TILX_PE12_CB1100_0 | Decimal Number | Tilt in mm/m along the longitudinal direction |
| G_TILT_PE12_CB1100_0 | Decimal Number | Air temperature in °C                         |
| G_TILH_PE12_CB1100_0 | Decimal Number | Relative humidity in %                        |
| G_TILY_PE13_CB1100_0 | Decimal Number | Tilt in mm/m along the transverse direction   |
| G_TILX_PE13_CB1100_0 | Decimal Number | Tilt in mm/m along the longitudinal direction |
| G_TILT_PE13_CB1100_0 | Decimal Number | Air temperature in °C                         |
| G_TILH_PE13_CB1100_0 | Decimal Number | Relative humidity in %                        |
| G_TILY_PE21_CB0400_0 | Decimal Number | Tilt in mm/m along the transverse direction   |
| G_TILX_PE21_CB0400_0 | Decimal Number | Tilt in mm/m along the longitudinal direction |
| G_TILT_PE21_CB0400_0 | Decimal Number | Air temperature in °C                         |
| G_TILH_PE21_CB0400_0 | Decimal Number | Relative humidity in %                        |
| G_TILY_PE22_CB0400_0 | Decimal Number | Tilt in mm/m along the transverse direction   |
| G_TILX_PE22_CB0400_0 | Decimal Number | Tilt in mm/m along the longitudinal direction |
| G_TILT_PE22_CB0400_0 | Decimal Number | Air temperature in °C                         |
| G_TILH_PE22_CB0400_0 | Decimal Number | Relative humidity in %                        |
| G_TILY_PE23_CB0400_0 | Decimal Number | Tilt in mm/m along the transverse direction   |
| G_TILX_PE23_CB0400_0 | Decimal Number | Tilt in mm/m along the longitudinal direction |
| G_TILT_PE23_CB0400_0 | Decimal Number | Air temperature in °C                         |
| G_TILH_PE23_CB0400_0 | Decimal Number | Relative humidity in %                        |

**Preprocessing:** No preprocessing steps have been applied.

---

#### Directory: 04_tiltmeter_trigger

In addition to continuous tilt measurements, a 90-second sample is recorded each time a load test is conducted with the test vehicle. The data can be found in the directory `04_tiltmeter_trigger`. Each test follows the same procedure: the vehicle begins in bridge span 3 near the abutment, drives to span 1, stops near the abutment, and then returns to its starting position. Throughout the test, the vehicle moves at a maximum speed of approximately 4 km/h and remains on the bridge for the entire time. Each recording captures a complete test cycle, including crossings in both directions. A threshold trigger is used to start the measurement. The tests are scheduled monthly. However, additional tests may be conducted occasionally, such as during measurement campaigns organized by external research teams.

The first load test was conducted on 2024-06-05, and data from all load tests up to 2024-10-31 is included. Only the channels measuring tilt along the bridge's longitudinal direction are recorded during the tests.

**Files:** The directory contains 151 CSV files. Each file name includes the prefix _tiltmeter_ followed by the timestamp marking the start of the triggered measurement. The file naming convention is as follows:

```
tiltmeter_{year}_{month}_{day}_T{hour}_{minute}_{second}.csv
```

**Sensors:** Sisgeo 0S542HD0502

**Sample rate:** 5 Hz

**Schema:**

| Column name          | Data type      | Description                                   |
| :------------------- | :------------- | :-------------------------------------------- |
| Timestamp            | Datetime       | ISO 8601 format (UTC)                         |
| G_TILX_PE11_CB1100_0 | Decimal Number | Tilt in mm/m along the longitudinal direction |
| G_TILX_PE12_CB1100_0 | Decimal Number | Tilt in mm/m along the longitudinal direction |
| G_TILX_PE13_CB1100_0 | Decimal Number | Tilt in mm/m along the longitudinal direction |
| G_TILX_PE21_CB0400_0 | Decimal Number | Tilt in mm/m along the longitudinal direction |
| G_TILX_PE22_CB0400_0 | Decimal Number | Tilt in mm/m along the longitudinal direction |
| G_TILX_PE23_CB0400_0 | Decimal Number | Tilt in mm/m along the longitudinal direction |

**Preprocessing:** To isolate signal components related specifically to the load test, the median of the initial 4 s is subtracted from each channel, removing unrelated influences, such as temperature effects. Additionally, to filter out falsely triggered measurements, each sample is compared to a reference crossing using cross-correlation to calculate a time-dependent Pearson coefficient. Only samples with a maximum coefficient above 0.85 are considered valid.

#### Directory: 05_supplementary_material
The dataset is accompanied by supplementary materials that provide essential context regarding the bridge structure, the measurement system, and the load testing campaign. This additional information is organized in the `05_supplementary_material` directory.
The content is structured into the following five subdirectories:

* 01_plans: Contains the construction drawings of the openLAB bridge in PDF format. Please note that all annotations in the plans are in German.
* 02_bim: Includes detailed 3D models of the openLAB bridge in Industry Foundation Classes (IFC) format - an open, widely supported standard for Building Information Modeling (BIM). These files can be viewed with any standard IFC viewer (many are freely available). Sub-models are included for the bridge geometry, the monitoring system, the reinforcement layout, the scaffolding, and the test vehicle. Together, these models supply the technical detail needed to support advanced numerical simulations, such as reproducing sensor signals from the load testing campaign.
* 03_sensor_datasheets: Contains the technical datasheets of all sensors used in the monitoring system, provided as PDF files.
* 04_sensor_installation: Includes photographs documenting the sensor installation process. Images are provided in JPG format, with brief descriptions available in the accompanying `captions.md` file.
* 05_website: A broader overview of the openLAB bridge research project is available on the official [project website](https://tu-dresden.de/bu/bauingenieurwesen/imb/forschung/ida-ki-infrastrukturdatenauswertung-mit-kuenstlicher-intelligenz?set_language=en). To ensure long-term accessibility, an offline copy is included in this directory. The copy is provided in HTML format and can be viewed in any web browser. The website also contains a short video tutorial (with English subtitles) explaining how to navigate and use the IFC models.

#### Contributors

- Authors:
    - Andreas Jansen (Data Manager)
    - Bertram Richter (Editor)
    - Robert Röder (Data Collector)
    - Max Herbers (Project Manager)
    - Steffen Marx (Project Leader)
- Date of data extraction: 2024-11
- License: CC-BY-SA
