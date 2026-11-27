"""Model registry: the active risk model and its ModelRun record.

The model is not trained here: its coefficients come from riskModel/nutrisense_risk_model.json and its evaluation from
riskModel/model_card.json (both written by ml/export_app_model.py). On first use the run is recorded once per version
(ifls5-lr-2) and any older run (ifls5-lr-1, the demo models) is deactivated; earlier assessments keep their own model_run_id.
"""
import logging
import threading

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..ai import risk_model
from ..models import ModelRun

log = logging.getLogger(__name__)

_lock = threading.Lock()
_active: risk_model.RiskModel | None = None
NAME = "stunting_risk"
MAIN_SETTING = "Random 80/20 split"  # the headline evaluation setting


def run_metrics() -> dict:
    """The model card, plus the main setting's headline figures at the top level for the governance list."""
    card = risk_model.model_card()
    main = card["evaluation"][MAIN_SETTING]
    return {**card, "setting": MAIN_SETTING, "accuracy": main["accuracy"], "recall_high": main["recall_high"],
            "precision_high": main["precision_high"], "auc": main["auc"],
            "false_negative_rate_high": round(1 - main["recall_high"], 4), "threshold_high": risk_model.THRESHOLD_HIGH,
            "threshold_medium": risk_model.THRESHOLD_MEDIUM}


def n_samples() -> int:
    data = risk_model.model_card()["data"]  # "IFLS5 2014-15, 5,202 children aged 0-59 months, ..."
    return int(data.split(",", 1)[1].split("children")[0].replace(",", "").strip())


def register(db: Session) -> ModelRun:
    """Record the shipped model as the active run (once per version)."""
    run = db.scalar(select(ModelRun).where(ModelRun.name == NAME, ModelRun.version == risk_model.MODEL_VERSION).order_by(ModelRun.id.desc()))
    if run is None:
        card = risk_model.model_card()
        run = ModelRun(name=NAME, algorithm=risk_model.ALGORITHM, version=risk_model.MODEL_VERSION, n_samples=n_samples(),
                       metrics=run_metrics(), feature_importances={}, is_active=True,
                       notes=f"{card['model']}. {card['data']}. Excludes {card['excludes']}.\n" + "\n".join(card["caveats"]))
        db.add(run)
        db.flush()
    db.execute(update(ModelRun).where(ModelRun.name == NAME, ModelRun.id != run.id).values(is_active=False))
    run.is_active = True
    db.commit()
    log.info("Active risk model: %s (run %s)", run.version, run.id)
    return run


def get_active(db: Session) -> risk_model.RiskModel:
    global _active
    with _lock:
        if _active is not None:
            return _active
    run = db.scalar(select(ModelRun).where(ModelRun.name == NAME, ModelRun.is_active.is_(True)).order_by(ModelRun.id.desc()))
    if run is None or run.version != risk_model.MODEL_VERSION:
        run = register(db)
    model = risk_model.RiskModel(run.id)
    with _lock:
        _active = model
    return model


def reset_cache() -> None:
    global _active
    with _lock:
        _active = None
