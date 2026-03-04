"""
Excel noon report parser.

Parses noon reports from Excel files containing:
- Position (lat/lon)
- Speed, course
- Fuel consumption
- Weather conditions
- Vessel condition (laden/ballast)
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ExcelParser:
    """
    Parse noon reports from Excel files.

    Supports common noon report formats used in maritime industry.
    """

    # Common column name mappings
    COLUMN_MAPPINGS = {
        "date": ["date", "report_date", "noon_date", "dt"],
        "latitude": ["latitude", "lat", "position_lat"],
        "longitude": ["longitude", "lon", "lng", "position_lon"],
        "speed": ["speed", "speed_kts", "sog", "speed_og"],
        "course": ["course", "heading", "cog"],
        "distance": ["distance", "distance_nm", "dist_steamed"],
        "fuel_consumption": [
            "fuel",
            "fuel_consumption",
            "fo_consumption",
            "me_fo",
            "daily_fuel",
        ],
        "wind_speed": ["wind_speed", "wind_bf", "wind"],
        "wind_direction": ["wind_dir", "wind_direction"],
        "wave_height": ["wave_height", "sea_state", "waves"],
        "draft_fwd": ["draft_fwd", "draft_forward", "fwd_draft"],
        "draft_aft": ["draft_aft", "aft_draft"],
        "cargo": ["cargo", "cargo_mt", "cargo_qty"],
        "condition": ["condition", "laden_ballast", "load_condition"],
    }

    def __init__(self, excel_file: Path):
        """
        Initialize Excel parser.

        Args:
            excel_file: Path to Excel file

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not excel_file.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_file}")

        self.excel_file = excel_file
        self.df: Optional[pd.DataFrame] = None
        self.column_map: Dict[str, str] = {}

    def parse(self, sheet_name: str = 0) -> List[Dict]:
        """
        Parse noon reports from Excel file.

        Args:
            sheet_name: Sheet name or index (default: first sheet)

        Returns:
            List of noon report dictionaries

        Raises:
            ValueError: If required columns not found
        """
        logger.info(f"Parsing Excel file: {self.excel_file}")

        # Read Excel file
        try:
            self.df = pd.read_excel(self.excel_file, sheet_name=sheet_name)
        except Exception as e:
            logger.error(f"Failed to read Excel file: {e}")
            raise

        # Normalize column names
        self.df.columns = [str(col).lower().strip() for col in self.df.columns]

        # Map columns
        self._map_columns()

        # Validate required columns
        required = ["date", "latitude", "longitude", "fuel_consumption"]
        missing = [col for col in required if col not in self.column_map]
        if missing:
            raise ValueError(
                f"Required columns not found: {missing}. "
                f"Available columns: {list(self.df.columns)}"
            )

        # Parse each row
        noon_reports = []
        for idx, row in self.df.iterrows():
            try:
                report = self._parse_row(row)
                if report:
                    noon_reports.append(report)
            except Exception as e:
                logger.warning(f"Failed to parse row {idx}: {e}")
                continue

        logger.info(f"Parsed {len(noon_reports)} noon reports")
        return noon_reports

    def _map_columns(self) -> None:
        """Map Excel columns to standard field names."""
        for standard_name, possible_names in self.COLUMN_MAPPINGS.items():
            for col in self.df.columns:
                if col in possible_names:
                    self.column_map[standard_name] = col
                    break

        logger.debug(f"Column mapping: {self.column_map}")

    def _parse_row(self, row: pd.Series) -> Optional[Dict]:
        """
        Parse a single row into a noon report dictionary.

        Args:
            row: DataFrame row

        Returns:
            Noon report dictionary or None if invalid
        """
        # Extract date
        date_col = self.column_map["date"]
        report_date = pd.to_datetime(row[date_col])

        # Extract position
        lat = float(row[self.column_map["latitude"]])
        lon = float(row[self.column_map["longitude"]])

        # Validate position
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            logger.warning(f"Invalid position: lat={lat}, lon={lon}")
            return None

        # Extract fuel consumption
        fuel_col = self.column_map["fuel_consumption"]
        fuel_mt = float(row[fuel_col])

        if fuel_mt <= 0 or fuel_mt > 200:  # Sanity check
            logger.warning(f"Invalid fuel consumption: {fuel_mt} MT")
            return None

        # Build report dictionary
        report = {
            "date": report_date,
            "latitude": lat,
            "longitude": lon,
            "fuel_consumption_mt": fuel_mt,
        }

        # Optional fields
        optional_mappings = {
            "speed": "speed_kts",
            "course": "course_deg",
            "distance": "distance_nm",
            "wind_speed": "wind_speed_bf",
            "wind_direction": "wind_direction_deg",
            "wave_height": "wave_height_m",
            "draft_fwd": "draft_fwd_m",
            "draft_aft": "draft_aft_m",
            "cargo": "cargo_mt",
            "condition": "condition",
        }

        for map_key, report_key in optional_mappings.items():
            if map_key in self.column_map:
                try:
                    value = row[self.column_map[map_key]]
                    if pd.notna(value):
                        report[report_key] = self._convert_value(map_key, value)
                except Exception as e:
                    logger.debug(f"Could not parse {map_key}: {e}")

        # Determine loading condition if not specified
        if "condition" not in report:
            report["is_laden"] = self._infer_loading_condition(report)
        else:
            condition_str = str(report["condition"]).lower()
            report["is_laden"] = "laden" in condition_str or "load" in condition_str

        return report

    def _convert_value(self, field: str, value) -> float:
        """Convert field value to appropriate type."""
        if field == "wind_speed":
            # Convert Beaufort to m/s if needed
            val = float(value)
            if val <= 12:  # Likely Beaufort scale
                # Beaufort to m/s approximation
                return 0.836 * (val**1.5)
            else:
                return val

        elif field == "wind_direction":
            # Convert wind direction to degrees
            val = str(value).upper().strip()
            direction_map = {
                "N": 0,
                "NNE": 22.5,
                "NE": 45,
                "ENE": 67.5,
                "E": 90,
                "ESE": 112.5,
                "SE": 135,
                "SSE": 157.5,
                "S": 180,
                "SSW": 202.5,
                "SW": 225,
                "WSW": 247.5,
                "W": 270,
                "WNW": 292.5,
                "NW": 315,
                "NNW": 337.5,
            }
            if val in direction_map:
                return direction_map[val]
            return float(value)

        elif field == "condition":
            # Return condition string as-is (not numeric)
            return str(value)

        elif field == "wave_height":
            # Ensure in meters
            val = float(value)
            if val > 20:  # Likely in feet
                return val * 0.3048
            return val

        else:
            return float(value)

    def _infer_loading_condition(self, report: Dict) -> bool:
        """
        Infer if vessel is laden based on draft.

        Args:
            report: Noon report dictionary

        Returns:
            True if laden, False if ballast
        """
        # Use draft if available
        if "draft_fwd_m" in report and "draft_aft_m" in report:
            avg_draft = (report["draft_fwd_m"] + report["draft_aft_m"]) / 2
            # Typical laden draft > 10m, ballast < 8m for MR tanker
            return avg_draft > 9.0

        # Use cargo quantity if available
        if "cargo_mt" in report:
            return report["cargo_mt"] > 10000

        # Default to laden (conservative for fuel estimation)
        return True

    def export_to_csv(self, output_file: Path) -> None:
        """
        Export parsed data to CSV.

        Args:
            output_file: Output CSV file path
        """
        if self.df is None:
            raise RuntimeError("No data to export. Call parse() first.")

        self.df.to_csv(output_file, index=False)
        logger.info(f"Exported data to {output_file}")

    def get_statistics(self) -> Dict:
        """
        Get statistics from parsed noon reports.

        Returns:
            Dictionary with statistics
        """
        if self.df is None:
            raise RuntimeError("No data available. Call parse() first.")

        stats = {
            "total_reports": len(self.df),
            "date_range": (
                self.df[self.column_map["date"]].min(),
                self.df[self.column_map["date"]].max(),
            ),
        }

        if "fuel_consumption" in self.column_map:
            fuel_col = self.column_map["fuel_consumption"]
            stats["total_fuel_mt"] = self.df[fuel_col].sum()
            stats["avg_daily_fuel_mt"] = self.df[fuel_col].mean()

        if "distance" in self.column_map:
            dist_col = self.column_map["distance"]
            stats["total_distance_nm"] = self.df[dist_col].sum()

        if "speed" in self.column_map:
            speed_col = self.column_map["speed"]
            stats["avg_speed_kts"] = self.df[speed_col].mean()

        return stats
