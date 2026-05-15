# Monitoring Data of the openLAB Research Bridge – Load Test on PE 2.1

Dieser Datensatz ist verfügbar:
- Autoren: Max Herbers, Bertram Richter, Maria Walker, Steffen Marx (TU Dresden)
- URL: https://opara.zih.tu-dresden.de/handle/123456789/1485
- DOI: https://doi.org/10.25532/OPARA-852
- Lizenz: CC-BY-SA 4.0

Alle Ortsangaben beziehen sich auf die Auflagerachse 10 (Süd-West), x = 0.

Die Krafteinleitung erfolgte im FT 2.1 mit zwei hydraulischen Zylindern mittig in den Steg (x = 24,3 m, im Bereich des maximalen Biegemoments infolge ständiger Einwirkungen).

## Messdaten

Übersicht der Messdateien: (# - file name - max. displacement u)

1.	MD_2025_05_06_09_08_25.txt		u_max = 5 mm
2.	MD_2025_05_06_10_43_20.txt		u_max = 10 mm
3.	MD_2025_05_06_12_05_10.txt		u_max = 20 mm
4.	MD_2025_05_06_13_43_17.txt		u_max = 30 mm
5.	MD_2025_05_06_16_07_15.txt		u_max = 40 mm
6.	MD_2025_05_06_17_39_40.txt		u_max = 50 mm
7.	MD_2025_05_06_18_30_51.txt		u_max = 60 mm

Die Versuche wurden am 06. Mai 2025 durchgeführt. Alle Zeitangaben in UTC+00:00.

Übersicht der Messkanäle:

| Kanal | Messgröße und Einheit        | Sensortyp                                      | Ort / Kommentar                                                   |
|-------|------------------------------|------------------------------------------------|-------------------------------------------------------------------|
| 1     | Zeit [s]                     | —                                              | —                                                                 |
| 2     | Dehnung [µm/m]               | Dehnungsmessstreifen Althen, L = 120 mm        | Neben Spanngliedöffnung (Schädigung Nr. 3 an Tag 3)               |
| 3     | Zeit [s]                     | —                                              | —                                                                 |
| 4     | Kraft [kN]                   | Kraftmessdose HBK C6A, max. 500 kN             | Nordseite, Richtung Achse 30                                      |
| 5     | Kraft [kN]                   | Kraftmessdose HBK C6A, max. 500 kN             | Südseite, Richtung Achse 10                                       |
| 6     | Weg [mm]                     | Induktiver Wegaufnehmer WETA 1/10, HBK         | Horizontal, neben DMS (Kanal 2); Messpunktabstand 160 mm          |
| 7     | Temperatur [°C]              | Thermoelement Typ K                            | Steg FT 2.3 Innenseite (ca. Feldmitte)	                        |
| 8     | Umgebungstemperatur [°C]     | Thermoelement Typ K                            | 50 cm über dem Pflaster, abgehängt vom FT 2.3 (ca. Feldmitte)     |
| 9     | Zeit [s]                     | —                                              | —                                                                 |
| 10    | Weg [mm]                     | Laserdistanzsensor Baumer OM30-L0350.HV.YUN    | Stegunterkante Mitte, FT 2.1, x = 7,79 m (Mitte Feld 1)           |
| 11    | Weg [mm]                     | Laserdistanzsensor Baumer OM30-L0350.HV.YUN    | Stegunterkante Mitte, FT 2.1, x = 15,00 m (Achse 20)              |
| 12    | Weg [mm]                     | Laserdistanzsensor Baumer OM30-L0350.HV.YUN    | Stegunterkante Mitte, FT 2.1, x = 19,68 m                         |
| 13    | Zeit [s]                     | —                                              | —                                                                 |
| 14    | Weg [mm]                     | Laserdistanzsensor Baumer OM30-L0350.HV.YUN    | Stegunterkante Mitte, FT 2.1, x = 24,30 m (unter Krafteinleitung) |
| 15    | Weg [mm]                     | Laserdistanzsensor Baumer OM30-L0350.HV.YUN    | Stegunterkante Mitte, FT 2.1, x = 30,00 m (Achse 30)              |
| 16    | —                            | —                                              | —                                                                 |
| 17    | Kraft [kN]                   | Berechnete Summe aus Kanälen 4 und 5           | —                                                                 |


## Ergänzende Hinweise

- bei den ersten 5 Belastungen (bis u_max = 40 mm) wurden vor Beginn der Messung die Daten genullt (Tara)
- anschließend wurde die Zielverschiebung (ausgehend von Null) aufgebracht
- bei den Belastungen 6 und 7 wurden die Messungen nicht zuvor tariert
- die Tara-Werte der ersten Belastungsstufen sind in den Messdateien enthalten (Zeile 28+29) und ermöglichen die Ermittlung der absoluten Verformungen


## Python-Skript

Das beiliegende Skript ist wie folgt aufgebaut:

1. Datenimport
2. Post-Processing: Während der Belastungsplateaus (konstante Verschiebung, u = 5-60 mm) wurden verschiedene Messungen an der Brücke durchgeführt.
    Dabei wurde teilweise in den Bereich zwischen Laser und Messpunkt eingriffen (im Bereich der Lasteinleitung), woraus unplausible Messdaten resultierten. Die Ausreißer wurden wie folgt bereinigt:
    - Messwerte wurden entfernt, wenn 
        1. die Längenänderung zum vorherigen Messwert größer war als 1 mm und 
        2. die Gesamtverschiebung größer als 65 mm war.
    - Zudem wurde ein gleitender Mittelwert mit einer Fenstergröße von 5 vorgesehen.
3. Plots
