"""
Vessel calibration module for WINDMAR.

Calibrates the theoretical Holtrop-Mennen model against actual
vessel performance data from noon reports.

Key calibration factors:
- Hull fouling (increases calm water resistance over time)
- Wind coefficient adjustment
- Wave response adjustment
- Engine degradation (increases SFOC)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from .vessel_model import VesselModel, VesselSpecs


logger = logging.getLogger(__name__)


@dataclass
class NoonReport:
    """Single noon report data point."""

    # Timestamp
    timestamp: datetime

    # Position
    latitude: float
    longitude: float

    # Vessel performance
    speed_over_ground_kts: float
    speed_through_water_kts: Optional[float] = None
    heading_deg: float = 0.0

    # Fuel consumption (MT for period, typically 24h)
    fuel_consumption_mt: float = 0.0
    period_hours: float = 24.0

    # Loading condition
    is_laden: bool = True
    draft_fwd_m: Optional[float] = None
    draft_aft_m: Optional[float] = None

    # Engine data
    engine_power_kw: Optional[float] = None
    engine_rpm: Optional[float] = None

    # Weather observations
    wind_speed_kts: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    wave_height_m: Optional[float] = None
    wave_direction_deg: Optional[float] = None
    swell_height_m: Optional[float] = None
    swell_direction_deg: Optional[float] = None

    # Current
    current_speed_kts: Optional[float] = None
    current_direction_deg: Optional[float] = None

    # Derived (calculated if not provided)
    distance_nm: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'NoonReport':
        """Create from dictionary."""
        d = d.copy()
        d['timestamp'] = datetime.fromisoformat(d['timestamp'])
        return cls(**d)


@dataclass
class CalibrationFactors:
    """Calibration factors for vessel model adjustment."""

    # Resistance calibration
    calm_water: float = 1.0  # Hull fouling factor (>1 = more resistance)
    wind: float = 1.0  # Wind coefficient adjustment
    waves: float = 1.0  # Wave response adjustment

    # Engine calibration
    sfoc_factor: float = 1.0  # SFOC multiplier (>1 = more consumption)

    # Metadata
    calibrated_at: Optional[datetime] = None
    num_reports_used: int = 0
    calibration_error: float = 0.0  # Mean error after calibration

    # Time since last dry dock (affects hull fouling)
    days_since_drydock: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        if self.calibrated_at:
            d['calibrated_at'] = self.calibrated_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'CalibrationFactors':
        """Create from dictionary."""
        d = d.copy()
        if d.get('calibrated_at'):
            d['calibrated_at'] = datetime.fromisoformat(d['calibrated_at'])
        return cls(**d)


@dataclass
class CalibrationResult:
    """Result of calibration optimization."""

    factors: CalibrationFactors
    reports_used: int
    reports_skipped: int
    mean_error_before: float  # Mean absolute error before calibration (MT/day)
    mean_error_after: float  # Mean absolute error after calibration (MT/day)
    improvement_pct: float  # Percentage improvement
    residuals: List[Dict]  # Per-report residuals for analysis


class VesselCalibrator:
    """
    Calibrates vessel model against actual performance data.

    Uses noon reports to derive calibration factors that adjust
    the theoretical Holtrop-Mennen predictions to match actual
    vessel performance.
    """

    # Minimum reports needed for calibration
    MIN_REPORTS = 5

    # Maximum calibration factor deviation from 1.0
    MAX_FACTOR_DEVIATION = 0.5  # ±50%

    def __init__(
        self,
        vessel_specs: Optional[VesselSpecs] = None,
        storage_path: Optional[Path] = None,
    ):
        """
        Initialize calibrator.

        Args:
            vessel_specs: Vessel specifications
            storage_path: Path for storing calibration data
        """
        self.vessel_specs = vessel_specs or VesselSpecs()
        self.storage_path = storage_path or Path("data/calibration")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Base model without calibration
        self.base_model = VesselModel(specs=self.vessel_specs)

        # Loaded noon reports
        self.noon_reports: List[NoonReport] = []

    def add_noon_report(self, report: NoonReport) -> None:
        """Add a noon report to the dataset."""
        self.noon_reports.append(report)

    def add_noon_reports_from_csv(self, csv_path: Path) -> int:
        """
        Import noon reports from CSV file.

        Expected columns:
        - timestamp (ISO format or common date formats)
        - latitude, longitude
        - speed_over_ground_kts
        - fuel_consumption_mt
        - period_hours (optional, default 24)
        - is_laden (optional, default True)
        - wind_speed_kts, wind_direction_deg (optional)
        - wave_height_m, wave_direction_deg (optional)
        - heading_deg (optional)

        Returns:
            Number of reports imported
        """
        import csv
        from dateutil import parser as date_parser

        count = 0
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Parse timestamp
                    timestamp = date_parser.parse(row['timestamp'])

                    # Create report
                    report = NoonReport(
                        timestamp=timestamp,
                        latitude=float(row['latitude']),
                        longitude=float(row['longitude']),
                        speed_over_ground_kts=float(row['speed_over_ground_kts']),
                        fuel_consumption_mt=float(row['fuel_consumption_mt']),
                        period_hours=float(row.get('period_hours', 24)),
                        is_laden=row.get('is_laden', 'true').lower() == 'true',
                        heading_deg=float(row.get('heading_deg', 0)),
                    )

                    # Optional weather data
                    if row.get('wind_speed_kts'):
                        report.wind_speed_kts = float(row['wind_speed_kts'])
                    if row.get('wind_direction_deg'):
                        report.wind_direction_deg = float(row['wind_direction_deg'])
                    if row.get('wave_height_m'):
                        report.wave_height_m = float(row['wave_height_m'])
                    if row.get('wave_direction_deg'):
                        report.wave_direction_deg = float(row['wave_direction_deg'])
                    if row.get('speed_through_water_kts'):
                        report.speed_through_water_kts = float(row['speed_through_water_kts'])
                    if row.get('engine_power_kw'):
                        report.engine_power_kw = float(row['engine_power_kw'])

                    self.noon_reports.append(report)
                    count += 1

                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse row: {e}")
                    continue

        logger.info(f"Imported {count} noon reports from {csv_path}")
        return count

    def add_noon_reports_from_excel(self, excel_path: Path) -> int:
        """
        Import noon reports from Excel file using ExcelParser.

        Returns:
            Number of reports imported
        """
        from src.database.excel_parser import ExcelParser

        parser = ExcelParser(excel_path)
        parsed = parser.parse()

        count = 0
        for row in parsed:
            try:
                report = NoonReport(
                    timestamp=row['date'],
                    latitude=row['latitude'],
                    longitude=row['longitude'],
                    speed_over_ground_kts=row.get('speed_kts', 0.0),
                    fuel_consumption_mt=row['fuel_consumption_mt'],
                    period_hours=24.0,
                    is_laden=row.get('is_laden', True),
                    heading_deg=row.get('course_deg', 0.0),
                )

                # Optional weather data — ExcelParser wind_speed is m/s, convert to kts
                if 'wind_speed_bf' in row:
                    report.wind_speed_kts = row['wind_speed_bf'] * 1.94384
                if 'wind_direction_deg' in row:
                    report.wind_direction_deg = row['wind_direction_deg']
                if 'wave_height_m' in row:
                    report.wave_height_m = row['wave_height_m']

                self.noon_reports.append(report)
                count += 1
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Failed to convert parsed row: {e}")
                continue

        logger.info(f"Imported {count} noon reports from Excel {excel_path}")
        return count

    def _prepare_report_for_model(self, report: NoonReport) -> Tuple[float, Dict]:
        """
        Prepare noon report data for model comparison.

        Returns:
            Tuple of (speed_kts, weather_dict)
        """
        # Use speed through water if available, otherwise SOG
        speed_kts = report.speed_through_water_kts or report.speed_over_ground_kts

        # Build weather dict
        weather = {'heading_deg': report.heading_deg}

        if report.wind_speed_kts is not None:
            # Convert knots to m/s
            weather['wind_speed_ms'] = report.wind_speed_kts * 0.51444
            weather['wind_dir_deg'] = report.wind_direction_deg or 0

        if report.wave_height_m is not None:
            weather['sig_wave_height_m'] = report.wave_height_m
            weather['wave_dir_deg'] = report.wave_direction_deg or 0

            # Combine swell if present
            if report.swell_height_m:
                # Use RMS combination
                weather['sig_wave_height_m'] = np.sqrt(
                    report.wave_height_m**2 + report.swell_height_m**2
                )

        return speed_kts, weather

    def _predict_fuel(
        self,
        report: NoonReport,
        calibration: CalibrationFactors,
    ) -> float:
        """
        Predict fuel consumption for a report using given calibration.

        Returns:
            Predicted fuel consumption (MT)
        """
        speed_kts, weather = self._prepare_report_for_model(report)

        # Calculate distance
        distance_nm = speed_kts * report.period_hours

        # Create model with calibration factors
        model = VesselModel(
            specs=self.vessel_specs,
            calibration_factors={
                'calm_water': calibration.calm_water,
                'wind': calibration.wind,
                'waves': calibration.waves,
            }
        )

        # Get prediction
        result = model.calculate_fuel_consumption(
            speed_kts=speed_kts,
            is_laden=report.is_laden,
            weather=weather if len(weather) > 1 else None,
            distance_nm=distance_nm,
        )

        # Apply SFOC factor
        fuel_mt = result['fuel_mt'] * calibration.sfoc_factor

        return fuel_mt

    def _filter_valid_reports(self) -> List[NoonReport]:
        """Filter reports suitable for calibration."""
        valid = []

        for report in self.noon_reports:
            # Skip reports with missing critical data
            if report.fuel_consumption_mt <= 0:
                continue
            if report.speed_over_ground_kts <= 0:
                continue

            # Skip port stays (very low speed)
            if report.speed_over_ground_kts < 5.0:
                continue

            # Skip maneuvering periods (short periods)
            if report.period_hours < 12:
                continue

            valid.append(report)

        return valid

    def calibrate(
        self,
        days_since_drydock: int = 0,
    ) -> CalibrationResult:
        """
        Perform calibration against noon reports.

        Uses optimization to find calibration factors that minimize
        the prediction error.

        Args:
            days_since_drydock: Days since last dry dock (affects expected fouling)

        Returns:
            CalibrationResult with optimized factors
        """
        valid_reports = self._filter_valid_reports()
        skipped = len(self.noon_reports) - len(valid_reports)

        if len(valid_reports) < self.MIN_REPORTS:
            raise ValueError(
                f"Need at least {self.MIN_REPORTS} valid reports for calibration, "
                f"got {len(valid_reports)}"
            )

        logger.info(f"Calibrating with {len(valid_reports)} reports ({skipped} skipped)")

        # Calculate initial error (no calibration)
        initial_factors = CalibrationFactors()
        initial_errors = []
        for report in valid_reports:
            predicted = self._predict_fuel(report, initial_factors)
            error = predicted - report.fuel_consumption_mt
            initial_errors.append(abs(error))

        mean_error_before = np.mean(initial_errors)
        logger.info(f"Initial mean error: {mean_error_before:.2f} MT")

        # Define objective function
        def objective(x):
            factors = CalibrationFactors(
                calm_water=x[0],
                wind=x[1],
                waves=x[2],
                sfoc_factor=x[3],
            )

            total_squared_error = 0
            for report in valid_reports:
                predicted = self._predict_fuel(report, factors)
                error = predicted - report.fuel_consumption_mt
                total_squared_error += error**2

            return total_squared_error

        # Initial guess
        # Estimate hull fouling from days since drydock
        # Typical: 1% per month up to 15%
        estimated_fouling = min(1.0 + (days_since_drydock / 30) * 0.01, 1.15)

        x0 = [estimated_fouling, 1.0, 1.0, 1.0]

        # Bounds
        min_f = 1.0 - self.MAX_FACTOR_DEVIATION
        max_f = 1.0 + self.MAX_FACTOR_DEVIATION
        bounds = [
            (0.6, 1.5),   # calm_water: wider range — Holtrop-Mennen can overpredict
            (min_f, max_f),  # wind
            (min_f, max_f),  # waves
            (0.85, 1.2),   # sfoc_factor
        ]

        # Optimize
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
        )

        # Build final calibration
        optimal_factors = CalibrationFactors(
            calm_water=result.x[0],
            wind=result.x[1],
            waves=result.x[2],
            sfoc_factor=result.x[3],
            calibrated_at=datetime.utcnow(),
            num_reports_used=len(valid_reports),
            days_since_drydock=days_since_drydock,
        )

        # Calculate final error and residuals
        final_errors = []
        residuals = []
        for report in valid_reports:
            predicted = self._predict_fuel(report, optimal_factors)
            actual = report.fuel_consumption_mt
            error = predicted - actual
            pct_error = (error / actual) * 100 if actual > 0 else 0

            final_errors.append(abs(error))
            residuals.append({
                'timestamp': report.timestamp.isoformat(),
                'actual_mt': actual,
                'predicted_mt': predicted,
                'error_mt': error,
                'error_pct': pct_error,
                'speed_kts': report.speed_over_ground_kts,
                'is_laden': report.is_laden,
            })

        mean_error_after = np.mean(final_errors)
        optimal_factors.calibration_error = mean_error_after

        improvement = ((mean_error_before - mean_error_after) / mean_error_before) * 100

        logger.info(f"Optimized factors: {optimal_factors}")
        logger.info(f"Final mean error: {mean_error_after:.2f} MT ({improvement:.1f}% improvement)")

        return CalibrationResult(
            factors=optimal_factors,
            reports_used=len(valid_reports),
            reports_skipped=skipped,
            mean_error_before=mean_error_before,
            mean_error_after=mean_error_after,
            improvement_pct=improvement,
            residuals=residuals,
        )

    def estimate_hull_fouling(
        self,
        days_since_drydock: int,
        operating_regions: List[str] = None,
    ) -> float:
        """
        Estimate hull fouling factor without calibration data.

        Based on industry data for typical fouling rates.

        Args:
            days_since_drydock: Days since last dry dock
            operating_regions: List of operating regions (affects fouling rate)

        Returns:
            Estimated calm_water calibration factor
        """
        # Base fouling rate: ~1% per month
        base_rate = 0.01 / 30  # per day

        # Adjust for operating regions
        region_multipliers = {
            'tropical': 1.5,
            'warm_temperate': 1.2,
            'cold': 0.8,
            'polar': 0.5,
        }

        multiplier = 1.0
        if operating_regions:
            for region in operating_regions:
                if region.lower() in region_multipliers:
                    multiplier = max(multiplier, region_multipliers[region.lower()])

        # Calculate fouling
        fouling_rate = base_rate * multiplier
        fouling_factor = 1.0 + (days_since_drydock * fouling_rate)

        # Cap at 20% increase
        fouling_factor = min(fouling_factor, 1.20)

        return fouling_factor

    def save_calibration(
        self,
        vessel_id: str,
        factors: CalibrationFactors,
    ) -> Path:
        """Save calibration factors to file."""
        filepath = self.storage_path / f"{vessel_id}_calibration.json"

        data = {
            'vessel_id': vessel_id,
            'factors': factors.to_dict(),
            'vessel_specs': asdict(self.vessel_specs),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved calibration to {filepath}")
        return filepath

    def load_calibration(self, vessel_id: str) -> Optional[CalibrationFactors]:
        """Load calibration factors from file."""
        filepath = self.storage_path / f"{vessel_id}_calibration.json"

        if not filepath.exists():
            logger.warning(f"No calibration file found for {vessel_id}")
            return None

        with open(filepath, 'r') as f:
            data = json.load(f)

        factors = CalibrationFactors.from_dict(data['factors'])
        logger.info(f"Loaded calibration for {vessel_id}: {factors}")
        return factors

    def save_noon_reports(self, vessel_id: str) -> Path:
        """Save noon reports to file."""
        filepath = self.storage_path / f"{vessel_id}_noon_reports.json"

        data = {
            'vessel_id': vessel_id,
            'reports': [r.to_dict() for r in self.noon_reports],
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath

    def load_noon_reports(self, vessel_id: str) -> int:
        """Load noon reports from file."""
        filepath = self.storage_path / f"{vessel_id}_noon_reports.json"

        if not filepath.exists():
            return 0

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.noon_reports = [
            NoonReport.from_dict(r) for r in data['reports']
        ]

        return len(self.noon_reports)


def create_calibrated_model(
    vessel_specs: Optional[VesselSpecs] = None,
    calibration_factors: Optional[CalibrationFactors] = None,
    days_since_drydock: int = 0,
) -> VesselModel:
    """
    Factory function to create a calibrated vessel model.

    Args:
        vessel_specs: Vessel specifications
        calibration_factors: Pre-computed calibration factors
        days_since_drydock: Days since last dry dock (for estimated fouling)

    Returns:
        Calibrated VesselModel
    """
    specs = vessel_specs or VesselSpecs()

    if calibration_factors:
        factors = {
            'calm_water': calibration_factors.calm_water,
            'wind': calibration_factors.wind,
            'waves': calibration_factors.waves,
        }
    else:
        # Estimate fouling based on days since drydock
        calibrator = VesselCalibrator(specs)
        fouling = calibrator.estimate_hull_fouling(days_since_drydock)
        factors = {
            'calm_water': fouling,
            'wind': 1.0,
            'waves': 1.0,
        }

    return VesselModel(specs=specs, calibration_factors=factors)
