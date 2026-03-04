"""
Model calibration from historical noon report data.

Calibrates vessel performance model using:
- Observed fuel consumption vs predicted
- Statistical optimization
- Loading condition separation
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from ..optimization.vessel_model import VesselModel, VesselSpecs

_VS = VesselSpecs  # shorthand for default parameter references


logger = logging.getLogger(__name__)


class ModelCalibrator:
    """
    Calibrate vessel performance model from noon reports.

    Uses optimization to find calibration factors that minimize
    difference between predicted and observed fuel consumption.
    """

    def __init__(self, vessel_specs: Optional[VesselSpecs] = None):
        """
        Initialize calibrator.

        Args:
            vessel_specs: Vessel specifications (defaults to MR tanker)
        """
        self.vessel_specs = vessel_specs or VesselSpecs()
        self.calibrated_factors: Optional[Dict[str, float]] = None
        self.calibration_quality: Optional[Dict] = None

    def calibrate(
        self,
        noon_reports: List[Dict],
        initial_factors: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Calibrate model from noon reports.

        Args:
            noon_reports: List of noon report dictionaries
            initial_factors: Initial calibration factors (defaults to 1.0)

        Returns:
            Dictionary of calibration factors:
                - calm_water: Calm water resistance factor
                - wind: Wind resistance factor
                - waves: Wave resistance factor

        Raises:
            ValueError: If insufficient data for calibration
        """
        if len(noon_reports) < 10:
            raise ValueError(
                f"Insufficient data for calibration: {len(noon_reports)} reports. "
                "Need at least 10."
            )

        logger.info(f"Calibrating model with {len(noon_reports)} noon reports")

        # Filter valid reports
        valid_reports = self._filter_valid_reports(noon_reports)
        logger.info(f"Using {len(valid_reports)} valid reports for calibration")

        if len(valid_reports) < 5:
            raise ValueError("Too few valid reports after filtering")

        # Separate by loading condition
        laden_reports = [r for r in valid_reports if r["is_laden"]]
        ballast_reports = [r for r in valid_reports if not r["is_laden"]]

        logger.info(f"Laden: {len(laden_reports)}, Ballast: {len(ballast_reports)}")

        # Set initial factors
        if initial_factors is None:
            initial_factors = {
                "calm_water": 1.0,
                "wind": 1.0,
                "waves": 1.0,
            }

        # Perform optimization
        x0 = [
            initial_factors["calm_water"],
            initial_factors["wind"],
            initial_factors["waves"],
        ]

        result = minimize(
            self._objective_function,
            x0,
            args=(valid_reports,),
            method="Nelder-Mead",
            bounds=[(0.5, 2.0), (0.5, 3.0), (0.5, 3.0)],
            options={"maxiter": 1000},
        )

        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")

        # Extract calibrated factors
        self.calibrated_factors = {
            "calm_water": result.x[0],
            "wind": result.x[1],
            "waves": result.x[2],
        }

        logger.info(f"Calibrated factors: {self.calibrated_factors}")

        # Calculate calibration quality metrics
        self.calibration_quality = self._calculate_quality_metrics(
            valid_reports, self.calibrated_factors
        )

        logger.info(f"Calibration RMSE: {self.calibration_quality['rmse']:.2f} MT/day")
        logger.info(f"Calibration R²: {self.calibration_quality['r_squared']:.3f}")

        return self.calibrated_factors

    def _filter_valid_reports(self, reports: List[Dict]) -> List[Dict]:
        """
        Filter valid noon reports for calibration.

        Removes outliers and reports with missing data.

        Args:
            reports: List of noon reports

        Returns:
            Filtered list of valid reports
        """
        valid = []

        for report in reports:
            # Check required fields
            required = ["date", "latitude", "longitude", "fuel_consumption_mt"]
            if not all(field in report for field in required):
                continue

            # Check for reasonable values
            fuel = report["fuel_consumption_mt"]
            if fuel <= 0 or fuel > 150:  # Unrealistic daily consumption
                continue

            # Get speed (or estimate from distance)
            if "speed_kts" in report:
                speed = report["speed_kts"]
            elif "distance_nm" in report:
                speed = report["distance_nm"] / 24.0  # Assume 24-hour report
            else:
                # Can't calibrate without speed
                continue

            # Check speed is reasonable
            if speed < 5 or speed > 20:
                continue

            # Add speed to report if not present
            if "speed_kts" not in report:
                report["speed_kts"] = speed

            valid.append(report)

        return valid

    def _objective_function(self, factors: np.ndarray, reports: List[Dict]) -> float:
        """
        Objective function for optimization.

        Calculates sum of squared errors between predicted and observed fuel.

        Args:
            factors: Array of [calm_water, wind, waves] factors
            reports: List of noon reports

        Returns:
            Sum of squared errors
        """
        calibration_factors = {
            "calm_water": factors[0],
            "wind": factors[1],
            "waves": factors[2],
        }

        # Create model with calibration factors
        model = VesselModel(
            specs=self.vessel_specs,
            calibration_factors=calibration_factors,
        )

        errors = []
        for report in reports:
            # Prepare weather data if available
            weather = None
            if "wind_speed_bf" in report:
                weather = {
                    "wind_speed_ms": report["wind_speed_bf"],
                    "wind_dir_deg": report.get("wind_direction_deg", 0),
                    "heading_deg": report.get("course_deg", 0),
                }

                if "wave_height_m" in report:
                    weather["sig_wave_height_m"] = report["wave_height_m"]
                    weather["wave_dir_deg"] = report.get("wind_direction_deg", 0)

            # Calculate predicted fuel
            distance = report.get("distance_nm", report["speed_kts"] * 24)

            predicted = model.calculate_fuel_consumption(
                speed_kts=report["speed_kts"],
                is_laden=report["is_laden"],
                weather=weather,
                distance_nm=distance,
            )

            # Calculate error
            observed = report["fuel_consumption_mt"]
            error = predicted["fuel_mt"] - observed
            errors.append(error**2)

        # Return mean squared error
        mse = np.mean(errors)
        return mse

    def _calculate_quality_metrics(
        self, reports: List[Dict], factors: Dict[str, float]
    ) -> Dict:
        """
        Calculate calibration quality metrics.

        Args:
            reports: List of noon reports
            factors: Calibration factors

        Returns:
            Dictionary with quality metrics
        """
        model = VesselModel(
            specs=self.vessel_specs,
            calibration_factors=factors,
        )

        observed = []
        predicted = []
        errors = []

        for report in reports:
            # Prepare weather
            weather = None
            if "wind_speed_bf" in report:
                weather = {
                    "wind_speed_ms": report["wind_speed_bf"],
                    "wind_dir_deg": report.get("wind_direction_deg", 0),
                    "heading_deg": report.get("course_deg", 0),
                }
                if "wave_height_m" in report:
                    weather["sig_wave_height_m"] = report["wave_height_m"]
                    weather["wave_dir_deg"] = report.get("wind_direction_deg", 0)

            distance = report.get("distance_nm", report["speed_kts"] * 24)

            pred = model.calculate_fuel_consumption(
                speed_kts=report["speed_kts"],
                is_laden=report["is_laden"],
                weather=weather,
                distance_nm=distance,
            )

            obs = report["fuel_consumption_mt"]

            observed.append(obs)
            predicted.append(pred["fuel_mt"])
            errors.append(obs - pred["fuel_mt"])

        observed = np.array(observed)
        predicted = np.array(predicted)
        errors = np.array(errors)

        # Calculate metrics
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / observed)) * 100

        # R-squared
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        ss_res = np.sum(errors**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r_squared": r_squared,
            "n_samples": len(reports),
            "mean_observed": np.mean(observed),
            "mean_predicted": np.mean(predicted),
            "std_error": np.std(errors),
        }

    def predict_consumption(
        self,
        speed_kts: float,
        is_laden: bool,
        weather: Optional[Dict] = None,
        distance_nm: float = 24.0 * _VS.service_speed_laden,  # One day at service speed
    ) -> Dict:
        """
        Predict fuel consumption using calibrated model.

        Args:
            speed_kts: Vessel speed (knots)
            is_laden: Loading condition
            weather: Weather conditions
            distance_nm: Distance (nautical miles)

        Returns:
            Prediction dictionary

        Raises:
            RuntimeError: If model not calibrated yet
        """
        if self.calibrated_factors is None:
            raise RuntimeError("Model not calibrated. Call calibrate() first.")

        model = VesselModel(
            specs=self.vessel_specs,
            calibration_factors=self.calibrated_factors,
        )

        return model.calculate_fuel_consumption(
            speed_kts, is_laden, weather, distance_nm
        )

    def get_calibration_report(self) -> str:
        """
        Generate calibration report text.

        Returns:
            Formatted calibration report

        Raises:
            RuntimeError: If model not calibrated yet
        """
        if self.calibrated_factors is None or self.calibration_quality is None:
            raise RuntimeError("Model not calibrated. Call calibrate() first.")

        report = []
        report.append("=" * 60)
        report.append("VESSEL PERFORMANCE MODEL CALIBRATION REPORT")
        report.append("=" * 60)
        report.append("")

        report.append("Calibration Factors:")
        report.append(
            f"  Calm Water Resistance: {self.calibrated_factors['calm_water']:.3f}"
        )
        report.append(f"  Wind Resistance:       {self.calibrated_factors['wind']:.3f}")
        report.append(
            f"  Wave Resistance:       {self.calibrated_factors['waves']:.3f}"
        )
        report.append("")

        q = self.calibration_quality
        report.append("Calibration Quality:")
        report.append(f"  Number of Samples:     {q['n_samples']}")
        report.append(f"  RMSE:                  {q['rmse']:.2f} MT/day")
        report.append(f"  MAE:                   {q['mae']:.2f} MT/day")
        report.append(f"  MAPE:                  {q['mape']:.1f}%")
        report.append(f"  R²:                    {q['r_squared']:.3f}")
        report.append(f"  Mean Observed:         {q['mean_observed']:.2f} MT/day")
        report.append(f"  Mean Predicted:        {q['mean_predicted']:.2f} MT/day")
        report.append(f"  Std Error:             {q['std_error']:.2f} MT/day")
        report.append("")

        # Interpretation
        report.append("Interpretation:")
        if q["r_squared"] > 0.8:
            report.append("  ✓ Excellent calibration quality (R² > 0.8)")
        elif q["r_squared"] > 0.6:
            report.append("  ✓ Good calibration quality (R² > 0.6)")
        elif q["r_squared"] > 0.4:
            report.append("  ⚠ Moderate calibration quality (R² > 0.4)")
        else:
            report.append("  ✗ Poor calibration quality (R² < 0.4)")
            report.append("    Consider collecting more data or checking data quality")

        if q["mape"] < 10:
            report.append("  ✓ Low prediction error (MAPE < 10%)")
        elif q["mape"] < 20:
            report.append("  ⚠ Moderate prediction error (MAPE < 20%)")
        else:
            report.append("  ✗ High prediction error (MAPE > 20%)")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    def plot_calibration(self, output_file: Optional[str] = None) -> None:
        """
        Plot calibration results.

        Args:
            output_file: Output file path (if None, display plot)

        Raises:
            RuntimeError: If model not calibrated yet
        """
        if self.calibrated_factors is None:
            raise RuntimeError("Model not calibrated. Call calibrate() first.")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed")
            return

        # This would require storing the reports, which we can add if needed
        logger.info("Calibration plotting requires stored report data")
