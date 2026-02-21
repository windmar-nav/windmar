"""Voyage persistence tables.

Revision ID: 003_voyage
Revises: 002_engine_log
Create Date: 2026-02-21

Adds voyages and voyage_legs tables for voyage history and reporting.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "003_voyage"
down_revision = "002_engine_log"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "voyages",
        sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("name", sa.String(200), nullable=True),
        sa.Column("departure_port", sa.String(200), nullable=True),
        sa.Column("arrival_port", sa.String(200), nullable=True),
        sa.Column("departure_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("arrival_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("total_distance_nm", sa.Float(), nullable=False),
        sa.Column("total_time_hours", sa.Float(), nullable=False),
        sa.Column("total_fuel_mt", sa.Float(), nullable=False),
        sa.Column("avg_sog_kts", sa.Float(), nullable=True),
        sa.Column("avg_stw_kts", sa.Float(), nullable=True),
        sa.Column("calm_speed_kts", sa.Float(), nullable=False),
        sa.Column("is_laden", sa.Boolean(), nullable=False, server_default=sa.text("TRUE")),
        sa.Column("vessel_specs_snapshot", postgresql.JSONB(), nullable=True),
        sa.Column("cii_estimate", postgresql.JSONB(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    )

    op.create_index("ix_voyages_departure", "voyages", ["departure_time"], postgresql_using="btree")
    op.create_index("ix_voyages_name", "voyages", ["name"])

    op.create_table(
        "voyage_legs",
        sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("voyage_id", postgresql.UUID(), sa.ForeignKey("voyages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("leg_index", sa.Integer(), nullable=False),
        sa.Column("from_name", sa.String(200), nullable=True),
        sa.Column("from_lat", sa.Float(), nullable=False),
        sa.Column("from_lon", sa.Float(), nullable=False),
        sa.Column("to_name", sa.String(200), nullable=True),
        sa.Column("to_lat", sa.Float(), nullable=False),
        sa.Column("to_lon", sa.Float(), nullable=False),
        sa.Column("distance_nm", sa.Float(), nullable=False),
        sa.Column("bearing_deg", sa.Float(), nullable=True),
        sa.Column("wind_speed_kts", sa.Float(), nullable=True),
        sa.Column("wind_dir_deg", sa.Float(), nullable=True),
        sa.Column("wave_height_m", sa.Float(), nullable=True),
        sa.Column("wave_dir_deg", sa.Float(), nullable=True),
        sa.Column("current_speed_ms", sa.Float(), nullable=True),
        sa.Column("current_dir_deg", sa.Float(), nullable=True),
        sa.Column("calm_speed_kts", sa.Float(), nullable=True),
        sa.Column("stw_kts", sa.Float(), nullable=True),
        sa.Column("sog_kts", sa.Float(), nullable=True),
        sa.Column("speed_loss_pct", sa.Float(), nullable=True),
        sa.Column("time_hours", sa.Float(), nullable=False),
        sa.Column("departure_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("arrival_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fuel_mt", sa.Float(), nullable=False),
        sa.Column("power_kw", sa.Float(), nullable=True),
        sa.Column("data_source", sa.String(50), nullable=True),
        sa.UniqueConstraint("voyage_id", "leg_index", name="uq_voyage_legs_voyage_leg"),
    )

    op.create_index("ix_voyage_legs_voyage", "voyage_legs", ["voyage_id"])


def downgrade() -> None:
    op.drop_index("ix_voyage_legs_voyage", table_name="voyage_legs")
    op.drop_table("voyage_legs")
    op.drop_index("ix_voyages_name", table_name="voyages")
    op.drop_index("ix_voyages_departure", table_name="voyages")
    op.drop_table("voyages")
