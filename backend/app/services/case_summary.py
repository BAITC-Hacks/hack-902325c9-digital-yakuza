import json
from functools import lru_cache

import pandas as pd

from app.core.beeline import agent_directory, agent_version, RUN_TIMEOUT_SECONDS


@lru_cache
def get_case_summary() -> dict:
    directory = agent_directory()
    import agent
    from environment import MAX_PILOTS, MIN_PILOT_CUSTOMERS, MAX_PILOT_CUSTOMERS
    from scoring_core import CHANNELS, TOTAL_BUDGET, MAX_TOTAL_CONTACTS
    from scoring_core import MAX_CAMPAIGNS, MAX_CUSTOMERS_PER_CAMPAIGN

    profile = pd.read_csv(directory / "customer_profile.csv")
    tariffs = pd.read_csv(directory / "tariff_dictionary.csv")
    segments = {}
    for column in ("arpu_segment", "data_segment", "call_segment", "current_tariff"):
        groups = profile.groupby(column, dropna=False).agg(
            customers=("ID_NUMBER", "size"),
            mean_predicted_arpu=("predicted_arpu", "mean"),
            total_predicted_arpu=("predicted_arpu", "sum"),
        ).reset_index()
        segments[column] = json.loads(groups.to_json(orient="records"))
    return {
        "case": "Beeline tariff marketing campaigns",
        "environment": "official_mock",
        "synthetic_data": True,
        **agent_version(),
        "strategy": {
            **{
                field: getattr(agent, constant)
                for field, constant in {
                    "max_pilot_budget_fraction": "EXPLORE_BUDGET_SHARE",
                    "max_pilot_contacts_fraction": "EXPLORE_CONTACT_SHARE",
                    "exploration_timeout_seconds": "TIME_BUDGET_S",
                    "max_pilots_per_candidate": "MAX_PILOTS_PER_CANDIDATE",
                    "pilot_channel": "PILOT_CHANNEL",
                    "fallback_channel": "FALLBACK_CHANNEL",
                    "prior_format": "PRIOR_FORMAT",
                    "prior_mode": "PRIOR_MODE",
                    "channel_economics": "CHANNEL_ECONOMICS",
                    "risk_k": "RISK_K",
                    "prior_shrink": "PRIOR_SHRINK",
                    "transfer_sd": "TRANSFER_SD",
                    "pilot_n_large": "PILOT_N_LARGE",
                    "pilot_n_small": "PILOT_N_SMALL",
                    "small_segment": "SMALL_SEGMENT",
                    "fallback_risk_k": "FALLBACK_RISK_K",
                    "max_campaigns": "MAX_CAMPAIGNS",
                    "max_per_campaign": "MAX_PER_CAMPAIGN",
                    "min_candidate_size": "MIN_CANDIDATE_SIZE",
                    "pilot_repeat_only_if_unclear": "PILOT_REPEAT_ONLY_IF_UNCLEAR",
                    "explore_unseen": "EXPLORE_UNSEEN",
                }.items() if hasattr(agent, constant)
            },
            "max_pilots": MAX_PILOTS,
        },
        "customers": len(profile),
        "baseline_total_arpu": float(profile["predicted_arpu"].sum()),
        "mean_predicted_arpu": float(profile["predicted_arpu"].mean()),
        "tariffs": json.loads(tariffs.to_json(orient="records")),
        "segments": segments,
        "channels": CHANNELS,
        "data_quality": {
            "missing_current_tariff": int(profile["current_tariff"].isna().sum()),
            "missing_arpu_segment": int(profile["arpu_segment"].isna().sum()),
            "zero_predicted_arpu": int((profile["predicted_arpu"].fillna(0) <= 0).sum()),
        },
        "constraints": {
            "max_campaigns": MAX_CAMPAIGNS,
            "max_campaign_size": MAX_CUSTOMERS_PER_CAMPAIGN,
            "max_contacts": MAX_TOTAL_CONTACTS,
            "budget": TOTAL_BUDGET,
            "max_pilots": MAX_PILOTS,
            "min_pilot_size": MIN_PILOT_CUSTOMERS,
            "max_pilot_size": MAX_PILOT_CUSTOMERS,
            "max_runtime_seconds": RUN_TIMEOUT_SECONDS,
            "pilots_consume_budget_and_contacts": True,
        },
    }
