"""
Traffic Separation Scheme (TSS) polygon definitions for major worldwide IMO zones.

Each TSS is defined as a list of (latitude, longitude) tuples forming a closed
polygon that encompasses the full TSS area (both traffic lanes + separation zone).

Coordinates are approximate outer boundaries derived from:
- IMO Resolutions (A.957(23), A.977(24), MSC.63(67), MSC.300(87))
- COLREG.2/Circ.58 (2006), COLREG.2/Circ.61 (2010), COLREG.2/Circ.64 (2012)
- 33 CFR Part 167 (US Coast Guard)
- Turkish Maritime Traffic Regulations (2019)
- Portuguese Hydrographic Office (IH)
- AMSA (Australian Maritime Safety Authority)

Datum: WGS-84
Format: list of (lat, lon) tuples, closed polygon (first == last)
Usage: Feed into ZoneChecker as Zone(coordinates=TSS_ZONES["key"], ...)
"""

TSS_ZONES: dict[str, list[tuple[float, float]]] = {

    # =========================================================================
    # 1. STRAIT OF GIBRALTAR
    # =========================================================================
    # IMO COLREG.2/Circ.58 (2006), MSC.300(87) (2010)
    # Separation zone centerline: 35d59.01'N 005d25.68'W to 35d56.21'N 005d44.98'W
    # Eastbound lane (south), Westbound lane (north), ~2 nm each, 0.5 nm sep zone
    # Reporting: 005d15'W (east) to 005d58'W (west)
    # Reference chart: Spanish Navy Hydrographic Institute No.445, WGS-84
    "strait_of_gibraltar": [
        (36.047, -5.328),   # NE corner (north of westbound lane, east end)
        (36.020, -5.483),   # N mid
        (35.990, -5.600),   # N, approaching Tarifa
        (35.960, -5.750),   # NW, west of separation
        (35.940, -5.967),   # NW corner (reporting line 005d58'W)
        (35.900, -5.967),   # SW corner
        (35.870, -5.750),   # S, south of eastbound lane
        (35.890, -5.600),   # S mid
        (35.910, -5.483),   # S mid
        (35.940, -5.328),   # SE corner (east of eastbound lane)
        (35.948, -5.260),   # SE approach (reporting line 005d15'W area)
        (36.047, -5.260),   # NE approach
        (36.047, -5.328),   # close polygon
    ],

    # =========================================================================
    # 2. DOVER STRAIT / PAS-DE-CALAIS
    # =========================================================================
    # IMO adopted 1967 (first international TSS), amended multiple times
    # NE-bound lane (English side), SW-bound lane (French side)
    # Each lane ~2 nm wide, separation zone ~1 nm
    # Key points: F3 light-float 51d24.15'N 002d00.38'E, MPC buoy 51d06.10'N 001d38.19'E
    # North Foreland 51d23'N 001d27'E to France/Belgium border 51d05'N 002d33'E
    # Reference: MGN 364 (M+F), Admiralty Chart 5500
    "dover_strait": [
        (51.40, 1.45),      # N, north of F3 light-float
        (51.40, 2.05),      # NE corner
        (51.25, 2.25),      # E, toward Belgium
        (51.08, 2.00),      # SE, French side
        (50.90, 1.85),      # S approach, French coast side
        (50.80, 1.60),      # SW corner
        (50.80, 1.20),      # S, south of Varne
        (50.90, 1.05),      # SW approach, English side
        (51.10, 1.20),      # W, English coast side
        (51.25, 1.35),      # NW, toward North Foreland
        (51.40, 1.45),      # close polygon
    ],

    # =========================================================================
    # 3. OFF USHANT / OUESSANT (FRANCE)
    # =========================================================================
    # IMO adopted 1975, implemented 1979, amended 2003 (COLREG.2/Circ.58)
    # Two main traffic lanes + coastal two-way route
    # 2 nm lanes separated by 1.2 nm, centered ~48d25'N, 005d00'W
    # Inshore traffic zone boundary points:
    #   48d37.20'N 005d11.90'W, 48d28.00'N 005d01.40'W (Men Korn),
    #   48d25.35'N 005d08.00'W (La Jument), 48d29.39'N 005d22.05'W
    # Separation zone: 48d38.00'N 005d12.90'W to 48d29.39'N 005d22.05'W
    # Reporting area: 40 nm circle centered on Stiff radar tower, Ushant
    "off_ushant": [
        (48.75, -5.85),     # NW corner (outer boundary of northbound lane)
        (48.75, -5.55),     # N
        (48.65, -5.20),     # NE corner
        (48.47, -5.02),     # E (Men Korn area)
        (48.42, -5.13),     # SE (La Jument area)
        (48.35, -5.30),     # S
        (48.35, -5.55),     # S
        (48.45, -5.75),     # SW
        (48.55, -5.85),     # W
        (48.75, -5.85),     # close polygon
    ],

    # =========================================================================
    # 4. OFF CASQUETS (CHANNEL ISLANDS)
    # =========================================================================
    # IMO adopted November 1973
    # Two 2 nm lanes separated by 1 nm zone
    # Centered ~49d42'N, 002d25'W
    # Shown on Admiralty charts 442, 2454, 2669, 2656, 2675
    "off_casquets": [
        (49.90, -2.65),     # NW corner
        (49.90, -2.20),     # NE corner
        (49.75, -2.05),     # E
        (49.55, -2.05),     # SE corner
        (49.55, -2.50),     # S
        (49.60, -2.65),     # SW corner
        (49.75, -2.75),     # W
        (49.90, -2.65),     # close polygon
    ],

    # =========================================================================
    # 5. STRAIT OF HORMUZ
    # =========================================================================
    # IMO adopted 1973, modified 1979
    # Two 2 nm lanes, 2 nm separation zone
    # Inshore traffic zone between Musandam Peninsula and TSS
    # Center ~26d34'N, 56d15'E
    # Extends from Persian Gulf approach SE to Gulf of Oman approach
    "strait_of_hormuz": [
        (26.75, 56.00),     # NW (Persian Gulf approach, Iranian side)
        (26.75, 56.30),     # N
        (26.60, 56.60),     # NE (turning into strait)
        (26.35, 56.80),     # E (Gulf of Oman approach, north lane)
        (26.15, 56.82),     # SE (outer approach, east)
        (26.15, 56.70),     # S (outer approach, south)
        (26.30, 56.50),     # SW (Musandam side)
        (26.45, 56.25),     # W (inside strait, Oman side)
        (26.55, 56.00),     # W (Persian Gulf side, south)
        (26.75, 56.00),     # close polygon
    ],

    # =========================================================================
    # 6. STRAIT OF MALACCA AND SINGAPORE
    # =========================================================================
    # IMO adopted, STRAITREP reporting 100d40'E to 104d23'E
    # TSS extends ~250 nm from One Fathom Bank (Port Klang) to
    # Horsburgh Lighthouse (Singapore)
    # Average depth ~23 m, six choke points
    # Singapore Strait specific lanes:
    #   01d12.51'N 103d52.15'E to 01d11.59'N 103d50.21'E (westbound)
    #   01d11.13'N 103d49.08'E to 01d08.65'N 103d44.30'E (westbound)
    # Split into Malacca section and Singapore section
    "strait_of_malacca_singapore": [
        # Malacca Strait western approach to Singapore Strait eastern end
        (3.00, 100.60),     # NW (One Fathom Bank approach, north)
        (2.80, 101.20),     # N (Malaysian coast side)
        (2.50, 101.70),     # NE
        (2.20, 102.20),     # N mid-strait
        (1.85, 102.80),     # N approaching Singapore
        (1.50, 103.40),     # N Singapore approaches
        (1.30, 103.80),     # NE Singapore Strait
        (1.22, 104.10),     # E (Horsburgh Lighthouse area)
        (1.10, 104.10),     # SE
        (1.05, 103.80),     # S Singapore Strait
        (1.00, 103.40),     # S
        (1.15, 102.80),     # S
        (1.55, 102.20),     # S mid-strait
        (1.90, 101.70),     # S
        (2.30, 101.20),     # S (Indonesian side)
        (2.60, 100.60),     # SW (approach, south)
        (3.00, 100.60),     # close polygon
    ],

    # =========================================================================
    # 7. BAB EL-MANDEB
    # =========================================================================
    # IMO adopted 1973, amended 1982, updated 2003
    # Two traffic lanes separated by 1 nm zone
    # Center ~12d55.8'N, 43d20.0'E
    # Strait 14 nm wide at narrowest, two 2 nm channels
    "bab_el_mandeb": [
        (12.98, 43.15),     # NW (Red Sea approach, Djibouti side)
        (12.98, 43.42),     # NE (Red Sea approach, Yemen side)
        (12.75, 43.50),     # E (strait, Yemen coast)
        (12.55, 43.55),     # SE (Gulf of Aden approach, north)
        (12.40, 43.50),     # S (approach, south)
        (12.40, 43.30),     # SW
        (12.55, 43.15),     # W (Djibouti/Perim Island area)
        (12.75, 43.10),     # W (strait, Djibouti side)
        (12.98, 43.15),     # close polygon
    ],

    # =========================================================================
    # 8. BOSPORUS / ISTANBUL STRAIT
    # =========================================================================
    # IMO adopted 1995 (Turkish Straits TSS)
    # WGS-84 datum, Turkish Maritime Traffic Regulations 2019
    # Extends from Anadolu/Rumeli Lighthouses (north) to
    # Ahirkapi/Kadikoy Inciburnu (south, Sea of Marmara)
    # Very narrow (~0.7-1.5 nm wide), sinuous channel ~17 nm long
    "bosporus": [
        (41.23, 29.05),     # N entrance, west (Rumeli Lighthouse area)
        (41.23, 29.12),     # N entrance, east (Anadolu Lighthouse area)
        (41.18, 29.10),     # Upper strait, east
        (41.12, 29.08),     # Mid strait, Bebek area, east
        (41.07, 29.05),     # Narrowest point area, east
        (41.04, 29.02),     # South of narrows, east
        (41.00, 29.02),     # S (Kadikoy/Inciburnu area, east)
        (41.00, 28.97),     # S (Ahirkapi area, west)
        (41.04, 28.98),     # South of narrows, west
        (41.07, 29.00),     # Narrowest point, west
        (41.12, 29.02),     # Mid strait, west
        (41.18, 29.04),     # Upper strait, west
        (41.23, 29.05),     # close polygon
    ],

    # =========================================================================
    # 9. DARDANELLES / CANAKKALE STRAIT
    # =========================================================================
    # IMO adopted 1995 (part of Turkish Straits TSS)
    # WGS-84 datum
    # Extends ~37 nm from Aegean Sea entrance to Sea of Marmara
    # Pilot boarding: 40d00.45'N, 026d08.15'E (Aegean approach)
    # Northern limit: 40d42.3'N, 027d18.5'E (Houkoy)
    "dardanelles": [
        (40.23, 26.18),     # SW entrance (Aegean, south Cape Helles)
        (40.23, 26.27),     # SE entrance (Aegean, north)
        (40.15, 26.38),     # E (entering strait)
        (40.10, 26.45),     # E (Canakkale town area)
        (40.12, 26.53),     # E (The Narrows area)
        (40.20, 26.63),     # NE (widening)
        (40.30, 26.73),     # NE
        (40.38, 26.85),     # NE (approaching Marmara)
        (40.42, 26.97),     # N (Gelibolu/Gallipoli area)
        (40.42, 26.90),     # N (west bank)
        (40.35, 26.78),     # NW
        (40.25, 26.65),     # W
        (40.15, 26.48),     # W (Narrows, west bank)
        (40.08, 26.40),     # W
        (40.10, 26.30),     # W (Kum Kale area)
        (40.18, 26.18),     # W (Cape Helles, north side)
        (40.23, 26.18),     # close polygon
    ],

    # =========================================================================
    # 10. OFF CAPE FINISTERRE (SPAIN)
    # =========================================================================
    # IMO Resolution A.957(23), adopted 5 December 2003
    # Amended to 4 lanes (2 northbound, 2 southbound) after Prestige disaster
    # Separation zone (a): 42d52.90'N 009d44.00'W to 43d21.00'N 009d36.40'W
    # Separation zone (b): 42d52.90'N 009d49.40'W to 43d23.00'N 009d41.90'W
    # Outer boundary extends to ~010d00'W
    # Dangerous cargo lanes further offshore
    "off_finisterre": [
        (43.42, -10.00),    # NW corner (outer dangerous cargo lane)
        (43.42, -9.60),     # NE corner
        (43.35, -9.61),     # E (near coast lanes)
        (43.18, -9.73),     # E
        (42.88, -9.73),     # SE
        (42.73, -9.73),     # S inner boundary
        (42.73, -10.00),    # SW corner (outer boundary)
        (42.88, -10.00),    # W
        (43.13, -9.95),     # W
        (43.25, -9.92),     # NW
        (43.42, -10.00),    # close polygon
    ],

    # =========================================================================
    # 11. OFF CABO DA ROCA / BERLENGAS (PORTUGAL)
    # =========================================================================
    # IMO adopted 2004, amended COLREG.2/Circ.61 (2010)
    # Four traffic lanes, four separation zones, one ITZ
    # Berlengas area to be avoided (>300 GT / dangerous cargo)
    # Centered ~38d46.14'N, 9d48.02'W
    # COPREP Roca Control VHF Ch. 16/22
    "off_cabo_da_roca": [
        (39.10, -9.90),     # NW corner
        (39.10, -9.50),     # NE corner
        (38.90, -9.45),     # E (inshore, north of Cabo da Roca)
        (38.73, -9.50),     # E (Cabo da Roca headland, ~38d47'N 9d30'W)
        (38.55, -9.50),     # SE
        (38.40, -9.55),     # S
        (38.40, -9.90),     # SW corner
        (38.55, -9.95),     # W
        (38.73, -9.95),     # W (outer boundary, abeam Cabo da Roca)
        (38.90, -9.93),     # NW
        (39.10, -9.90),     # close polygon
    ],

    # =========================================================================
    # 12. OFF CABO DE SAO VICENTE (PORTUGAL)
    # =========================================================================
    # IMO adopted 2004, amended COLREG.2/Circ.61 (2010)
    # Four traffic lanes, four separation zones, one ITZ
    # Located at SW tip of Portugal (~37d00'N, 8d60'W)
    # N-S and E-W traffic meets here (Atlantic/Mediterranean)
    "off_cabo_sao_vicente": [
        (37.20, -9.15),     # NW corner (Atlantic approach, north)
        (37.20, -8.80),     # NE corner
        (37.05, -8.75),     # E (coast side)
        (36.90, -8.80),     # SE (south of cape)
        (36.80, -8.90),     # S (Mediterranean approach)
        (36.80, -9.15),     # SW corner (outer Atlantic boundary)
        (36.90, -9.20),     # W
        (37.05, -9.20),     # W (abeam cape)
        (37.20, -9.15),     # close polygon
    ],

    # =========================================================================
    # 13. SUEZ CANAL APPROACHES (PORT SAID)
    # =========================================================================
    # Suez Canal Authority Rules of Navigation
    # Fairway Buoy No. 8: 31d21.32'N, 32d20.81'E
    # TSS south of 31d28.7'N, between 32d00.27'E and 32d37.43'E
    # Approach channel extends ~15 nm from Fairway Buoy
    "suez_canal_approaches": [
        (31.55, 32.15),     # NW (open sea, north of approach)
        (31.55, 32.45),     # NE
        (31.40, 32.50),     # E (approach, east edge)
        (31.30, 32.40),     # SE (converging to channel)
        (31.25, 32.35),     # S (Fairway Buoy area)
        (31.20, 32.30),     # S (canal entrance area)
        (31.20, 32.25),     # SW (canal entrance)
        (31.25, 32.20),     # W (approach, west edge)
        (31.40, 32.15),     # NW
        (31.55, 32.15),     # close polygon
    ],

    # =========================================================================
    # 14. OFF CABO DE GATA (SPAIN)
    # =========================================================================
    # IMO adopted 1998, repositioned 2006 (COLREG.2/Circ.58)
    # Moved 17-21 nm south of cape for environmental protection
    # Two 3 nm lanes, 2 nm separation zone
    # E-W traffic (Gibraltar to Mediterranean)
    # Cabo de Gata Natural Park (Almeria province) protected area
    # 13-knot speed recommendation for cetacean conservation
    "off_cabo_de_gata": [
        (36.65, -2.45),     # NW corner (north of westbound lane)
        (36.65, -1.85),     # NE corner
        (36.55, -1.80),     # E
        (36.40, -1.80),     # SE corner (south of eastbound lane)
        (36.40, -2.50),     # SW corner
        (36.50, -2.55),     # W
        (36.65, -2.45),     # close polygon
    ],

    # =========================================================================
    # 15. OFF BONIFACIO STRAIT (CORSICA/SARDINIA)
    # =========================================================================
    # Strait centered ~41d19'N, 9d07'E
    # 6.8 nm wide at narrowest
    # Eastern pilot: 41d24.80'N, 009d30.00'E
    # Western pilot: 41d17.28'N, 008d58.50'E
    # VHF reporting: Pertusato (Corsica) W→E, Maddalena (Italy) E→W
    "off_bonifacio_strait": [
        (41.45, 8.90),      # NW (Corsica side, west approach)
        (41.45, 9.55),      # NE (Corsica side, east approach)
        (41.35, 9.55),      # E
        (41.25, 9.50),      # E (eastern pilot area)
        (41.15, 9.40),      # SE (Maddalena area)
        (41.10, 9.20),      # S (Sardinia side, in strait)
        (41.10, 8.95),      # SW (Sardinia side, west)
        (41.20, 8.85),      # W (western pilot area)
        (41.30, 8.85),      # W
        (41.45, 8.90),      # close polygon
    ],

    # =========================================================================
    # 16. OFF CAPE OF GOOD HOPE (SOUTH AFRICA)
    # =========================================================================
    # IMO adopted TSS off south coast (Alphard Banks)
    # Manages laden tankers and deep-draft vessels (coal, iron ore)
    # North and south of Alphard Banks + FA Platform
    # Alphard Banks ~34 nm south of Cape Infanta
    # Strong Agulhas Current (>2 knots)
    # Cape of Good Hope at 34d21'S, 18d28'E
    "off_cape_of_good_hope": [
        (-34.25, 19.80),    # NW (approaching from west, north lane)
        (-34.25, 21.00),    # NE
        (-34.50, 21.20),    # E (east of Alphard Banks)
        (-34.85, 21.20),    # SE
        (-35.15, 21.00),    # S (south of Alphard Banks)
        (-35.15, 19.80),    # SW
        (-34.85, 19.60),    # W (west of Alphard Banks)
        (-34.50, 19.60),    # NW
        (-34.25, 19.80),    # close polygon
    ],

    # =========================================================================
    # 17. GALVESTON BAY APPROACH (US GULF COAST)
    # =========================================================================
    # 33 CFR 167.350 (USCG)
    # NAD-83 datum
    # Inbound (NW heading), outbound (SE heading) lanes
    # Precautionary area at entrance to Galveston Bay
    # Bolivar Roads approach
    "galveston_bay_approach": [
        (29.35, -94.55),    # NW (inside precautionary area)
        (29.35, -94.40),    # NE
        (29.20, -94.30),    # E (approach, east)
        (29.00, -94.25),    # SE (offshore, east lane)
        (28.85, -94.25),    # S (outer approach)
        (28.85, -94.50),    # SW (outer approach)
        (29.00, -94.55),    # W (offshore, west lane)
        (29.20, -94.60),    # W (approach, west)
        (29.35, -94.55),    # close polygon
    ],

    # =========================================================================
    # 18. OFF SAN FRANCISCO (US WEST COAST)
    # =========================================================================
    # 33 CFR 167.400-167.406 (USCG)
    # NAD-83 datum
    # Six parts: Precautionary Area, Northern/Southern/Western Approaches,
    # Main Ship Channel, Area to Be Avoided
    # Convergence zone west of Golden Gate Bridge
    "off_san_francisco": [
        (37.90, -122.80),   # NW (Northern Approach, outer)
        (37.90, -122.55),   # NE
        (37.85, -122.45),   # E (approaching Golden Gate)
        (37.80, -122.50),   # E (Main Ship Channel)
        (37.70, -122.60),   # SE (Southern Approach)
        (37.60, -122.70),   # S
        (37.55, -122.85),   # SW (Southern Approach, outer)
        (37.55, -123.00),   # W (Western Approach)
        (37.65, -123.05),   # W
        (37.75, -123.00),   # NW (Western Approach, north)
        (37.85, -122.90),   # N
        (37.90, -122.80),   # close polygon
    ],

    # =========================================================================
    # 19. GREAT BARRIER REEF / TORRES STRAIT (AUSTRALIA)
    # =========================================================================
    # IMO adopted 2014 (MSC two-way route)
    # Extends from western Torres Strait through Prince of Wales Channel
    # to southern boundary of GBR Marine Park (near Cairns)
    # Booby Island: 10d36'S, 141d54'E
    # Bramble Cay: 9d09'S, 143d53'E
    # Inner Route: Cape York to near Cairns
    # Pilotage mandatory for ships >70m and tankers
    "great_barrier_reef_torres_strait": [
        (-10.55, 141.80),   # NW (Booby Island approach, west Torres Strait)
        (-9.10, 143.90),    # NE (Bramble Cay area, east Torres Strait)
        (-9.50, 144.20),    # E (entering Inner Route from north)
        (-11.00, 144.00),   # SE (Inner Route, north section)
        (-13.00, 144.30),   # S (Inner Route, mid section)
        (-14.50, 145.30),   # S (Inner Route, approaching Cairns)
        (-16.80, 146.00),   # S (southern GBR Marine Park boundary)
        (-16.80, 145.70),   # SW
        (-14.50, 145.00),   # W (Inner Route, west edge)
        (-13.00, 143.90),   # W
        (-11.00, 143.60),   # W
        (-10.55, 142.50),   # W (Torres Strait, south approach)
        (-10.55, 141.80),   # close polygon
    ],

    # =========================================================================
    # 20. OFF GOTLAND (BALTIC SEA, SWEDEN)
    # =========================================================================
    # IMO Resolution A.977(24), adopted 1 December 2005
    # Implemented 1 July 2006
    # Deep-water route (>25m depth, min draught 12m)
    # Between TSS "Off Kopu Peninsula" and "In Bornholmsgat"
    # South of Hoburgs Bank and Norra Midsjobanken
    # Reference charts: Swedish Charts Nos. 7, 8 (2001), WGS-84
    "off_gotland": [
        (57.80, 18.00),     # NW (north of Gotland, west approach)
        (57.80, 19.50),     # NE
        (57.40, 19.80),     # E (east of Gotland)
        (56.90, 19.50),     # SE (south of Hoburgs Bank)
        (56.70, 18.70),     # S (Norra Midsjobanken area)
        (56.70, 17.80),     # SW
        (56.90, 17.50),     # W
        (57.20, 17.50),     # W
        (57.50, 17.70),     # NW
        (57.80, 18.00),     # close polygon
    ],

    # =========================================================================
    # 21. OFF CAPE SPARTEL (Morocco)
    # =========================================================================
    # IMO COLREG.2/Circ.64 (2012)
    # West of Strait of Gibraltar, N-S traffic separation for Atlantic approach
    # Two lanes: northbound (east), southbound (west)
    "off_cape_spartel": [
        (35.85, -6.10),     # NE
        (35.85, -6.30),     # NW
        (35.70, -6.35),     # W
        (35.55, -6.30),     # SW
        (35.55, -6.10),     # SE
        (35.70, -6.05),     # E
        (35.85, -6.10),     # close polygon
    ],

    # =========================================================================
    # 22. IN THE STRAIT OF BONIFACIO (Banco de Hoyo area)
    # =========================================================================
    # Also known as "Al Hoceima" / western Med approach to Alboran Sea
    # IMO-adopted TSS off the Moroccan/Spanish coast in the western Mediterranean
    # E-W traffic near Banco de Hoyo (shallow area ~35.2N, -3.2W)
    "banco_de_hoyo": [
        (35.35, -3.50),     # NW
        (35.35, -2.90),     # NE
        (35.15, -2.85),     # SE
        (35.05, -3.10),     # S
        (35.05, -3.50),     # SW
        (35.15, -3.55),     # W
        (35.35, -3.50),     # close polygon
    ],
}


# ============================================================================
# CONVENIENCE: All TSS metadata for bulk loading into ZoneChecker
# ============================================================================

TSS_METADATA: dict[str, dict] = {
    # direction_deg: primary traffic flow bearing (True North = 0°, clockwise)
    # tolerance_deg: acceptable deviation from direction (default 20°)
    # bidirectional: True if the polygon covers both traffic lanes (either direction valid)
    "strait_of_gibraltar": {
        "name": "Strait of Gibraltar TSS",
        "authority": "IMO",
        "notes": "COLREG.2/Circ.58 (2006). E-W traffic, 2nm lanes, 0.5nm sep zone. GIBREP reporting.",
        "direction_deg": 90,    # Eastbound / westbound (bidirectional)
        "tolerance_deg": 25,
        "bidirectional": True,
    },
    "dover_strait": {
        "name": "Dover Strait TSS",
        "authority": "IMO",
        "notes": "First international TSS (1967). NE/SW-bound lanes. CALDOVREP reporting.",
        "direction_deg": 40,    # NE-bound / SW-bound
        "tolerance_deg": 25,
        "bidirectional": True,
    },
    "off_ushant": {
        "name": "Off Ushant (Ouessant) TSS",
        "authority": "IMO",
        "notes": "Amended 2003. 2nm lanes, 1.2nm sep zone. OUESSREP 40nm reporting area.",
        "direction_deg": 200,   # Roughly SSW / NNE
        "tolerance_deg": 30,
        "bidirectional": True,
    },
    "off_casquets": {
        "name": "Off Casquets TSS",
        "authority": "IMO",
        "notes": "Adopted 1973. 2nm lanes, 1nm sep zone. Channel Islands.",
        "direction_deg": 55,    # NE / SW
        "tolerance_deg": 25,
        "bidirectional": True,
    },
    "strait_of_hormuz": {
        "name": "Strait of Hormuz TSS",
        "authority": "IMO",
        "notes": "Adopted 1973, modified 1979. 2nm lanes, 2nm sep zone. Critical oil transit chokepoint.",
        "direction_deg": 300,   # WNW inbound / ESE outbound
        "tolerance_deg": 25,
        "bidirectional": True,
    },
    "strait_of_malacca_singapore": {
        "name": "Straits of Malacca and Singapore TSS",
        "authority": "IMO",
        "notes": "STRAITREP 100d40'E-104d23'E. 250nm extent. Six choke points.",
        "direction_deg": 135,   # SE-bound / NW-bound
        "tolerance_deg": 30,
        "bidirectional": True,
    },
    "bab_el_mandeb": {
        "name": "Bab el-Mandeb TSS",
        "authority": "IMO",
        "notes": "Adopted 1973, amended 1982/2003. 2nm lanes, 1nm sep zone. Red Sea entrance.",
        "direction_deg": 340,   # NNW into Red Sea / SSE out
        "tolerance_deg": 25,
        "bidirectional": True,
    },
    "bosporus": {
        "name": "Bosporus (Istanbul Strait) TSS",
        "authority": "IMO / Turkey",
        "notes": "Adopted 1995. Turkish Straits VTS. Very narrow sinuous channel.",
        "direction_deg": 20,    # NNE / SSW (sinuous, wide tolerance)
        "tolerance_deg": 40,
        "bidirectional": True,
    },
    "dardanelles": {
        "name": "Dardanelles (Canakkale) TSS",
        "authority": "IMO / Turkey",
        "notes": "Adopted 1995. Turkish Straits VTS. Aegean to Sea of Marmara.",
        "direction_deg": 45,    # NE / SW
        "tolerance_deg": 35,
        "bidirectional": True,
    },
    "off_finisterre": {
        "name": "Off Finisterre TSS",
        "authority": "IMO",
        "notes": "A.957(23), 2003. 4 lanes (2N, 2S) after Prestige. Dangerous cargo outer lanes.",
        "direction_deg": 180,   # N-S traffic
        "tolerance_deg": 25,
        "bidirectional": True,
    },
    "off_cabo_da_roca": {
        "name": "Off Cabo da Roca / Berlengas TSS",
        "authority": "IMO",
        "notes": "Adopted 2004, amended 2010. 4 lanes, 4 sep zones, 1 ITZ. Berlengas ATBA.",
        "direction_deg": 180,   # N-S traffic
        "tolerance_deg": 25,
        "bidirectional": True,
    },
    "off_cabo_sao_vicente": {
        "name": "Off Cabo de Sao Vicente TSS",
        "authority": "IMO",
        "notes": "Adopted 2004, amended 2010. SW tip of Portugal. Atlantic/Med traffic convergence.",
        "direction_deg": 90,    # E-W rounding
        "tolerance_deg": 30,
        "bidirectional": True,
    },
    "suez_canal_approaches": {
        "name": "Suez Canal Approaches (Port Said) TSS",
        "authority": "SCA",
        "notes": "Suez Canal Authority. Fairway Buoy at 31d21'N 32d21'E.",
        "direction_deg": 180,   # N-S approach
        "tolerance_deg": 25,
        "bidirectional": True,
    },
    "off_cabo_de_gata": {
        "name": "Off Cabo de Gata TSS",
        "authority": "IMO",
        "notes": "Adopted 1998, repositioned 2006. 3nm lanes, 2nm sep. 13kn cetacean speed rec.",
        "direction_deg": 65,    # ENE / WSW
        "tolerance_deg": 25,
        "bidirectional": True,
    },
    "off_bonifacio_strait": {
        "name": "Off Bonifacio Strait TSS",
        "authority": "IMO / France / Italy",
        "notes": "Corsica-Sardinia. Pilotage: Pertusato (W-E), Maddalena (E-W).",
        "direction_deg": 90,    # E-W
        "tolerance_deg": 25,
        "bidirectional": True,
    },
    "off_cape_of_good_hope": {
        "name": "Off Cape of Good Hope TSS",
        "authority": "IMO / SAMSA",
        "notes": "Off Alphard Banks, south coast. Deep-draft tanker/bulk routes. Agulhas Current.",
        "direction_deg": 90,    # E-W rounding
        "tolerance_deg": 30,
        "bidirectional": True,
    },
    "galveston_bay_approach": {
        "name": "Galveston Bay Approach TSS",
        "authority": "USCG",
        "notes": "33 CFR 167.350. NAD-83. Bolivar Roads approach to Houston Ship Channel.",
        "direction_deg": 330,   # NNW approach
        "tolerance_deg": 20,
        "bidirectional": True,
    },
    "off_san_francisco": {
        "name": "Off San Francisco TSS",
        "authority": "USCG",
        "notes": "33 CFR 167.400-406. NAD-83. Precautionary area, 3 approaches, main channel, ATBA.",
        "direction_deg": 90,    # E approach (multiple approaches, wide tolerance)
        "tolerance_deg": 40,
        "bidirectional": True,
    },
    "great_barrier_reef_torres_strait": {
        "name": "Great Barrier Reef / Torres Strait Two-Way Route",
        "authority": "IMO / AMSA",
        "notes": "Adopted 2014. Inner Route. Pilotage mandatory >70m/tankers. REEFVTS.",
        "direction_deg": 330,   # NW-SE route
        "tolerance_deg": 30,
        "bidirectional": True,
    },
    "off_gotland": {
        "name": "Off Gotland TSS / Deep-Water Route",
        "authority": "IMO",
        "notes": "A.977(24), 2005. Min depth 25m, draught limit 12m. Baltic NE-bound traffic.",
        "direction_deg": 30,    # NE / SW
        "tolerance_deg": 25,
        "bidirectional": True,
    },
    "off_cape_spartel": {
        "name": "Off Cape Spartel TSS",
        "authority": "IMO",
        "notes": "COLREG.2/Circ.64 (2012). N-S traffic west of Gibraltar. Atlantic approach.",
        "direction_deg": 180,   # N-S traffic
        "tolerance_deg": 25,
        "bidirectional": True,
    },
    "banco_de_hoyo": {
        "name": "Banco de Hoyo TSS",
        "authority": "IMO",
        "notes": "Western Mediterranean, Alboran Sea. E-W traffic near shallow Banco de Hoyo (~35.2N).",
        "direction_deg": 75,    # ENE / WSW
        "tolerance_deg": 25,
        "bidirectional": True,
    },
}
