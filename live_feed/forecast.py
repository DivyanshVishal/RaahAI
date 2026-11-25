"""Run the saved models with the station observations available at each report."""

import numpy as np
import pandas as pd

from predict import raw_q
from train_multistop import apply_blend, make_pairs, prepare


class JourneyForecast:
    def __init__(self, segments, settings, models):
        self.segments = segments.sort_values("seg_idx").reset_index(drop=True)
        self.settings = settings
        self.models = models
        self.pre = raw_q(models, "pre", prepare(self.segments, settings["pre_departure"], settings["categories"]))
        self.departures = {}
        self.arrivals = {}

    def update(self, event):
        stop = event["stop_index"]
        if event["delay_min"] is not None:
            observations = self.departures if event["type"] == "departure" else self.arrivals
            observations[stop] = event

        # A late report from a previous station must not move the train backwards.
        current = max(self.arrivals.keys() | self.departures.keys(), default=-1)
        if current < 0:
            return []
        departure = self.departures.get(current)
        arrival = self.arrivals.get(current)
        latest = departure or arrival
        current_rows = self.segments.index[self.segments.seg_idx.eq(current)]
        if current_rows.empty:
            return []
        i = current_rows[0]
        ahead = np.arange(i, len(self.segments))
        journey = self.segments.copy()

        def delay(observations, index):
            return observations[index]["delay_min"] if index in observations else np.nan

        max_departure = max((obs["delay_min"] for index, obs in self.departures.items() if index <= current),
                            default=np.nan)
        dep = delay(self.departures, current)
        arr = delay(self.arrivals, current)
        row = journey.iloc[i]
        dwell = np.nan
        if departure and arrival and pd.notna(row.sch_dwell_A_min):
            dwell = (pd.Timestamp(departure["actual_at"]) - pd.Timestamp(arrival["actual_at"])).total_seconds() / 60
            dwell -= row.sch_dwell_A_min
        source = "dep" if departure else "arr"
        known = latest["delay_min"]
        values = {
            "known_delay_A": known,
            "known_delay_A_src": source,
            "dep_delay_A": dep,
            "arr_delay_A": arr,
            "dwell_overrun_A": dwell,
            "gain_prev1": arr - delay(self.departures, current - 1),
            "gain_prev2": delay(self.arrivals, current - 1) - delay(self.departures, current - 2),
            "trend_2stops": dep - delay(self.departures, current - 2),
            "origin_dep_delay": delay(self.departures, 0),
            "max_dep_delay_so_far": max_departure if departure else np.nan,
            "arr_imputed_A": False,
        }
        for column, value in values.items():
            journey.loc[i, column] = value
        size = len(journey)
        pairs = make_pairs(journey, np.full(len(ahead), i), ahead, np.zeros(len(ahead)), np.full(len(ahead), size))
        # Use the physical stop numbers for the horizon, even if a segment was excluded during cleaning.
        horizons = journey.iloc[ahead].seg_idx.to_numpy() - current + 1
        pairs["stops_ahead"] = horizons
        live = known + raw_q(self.models, "live", prepare(pairs, self.settings["live"], self.settings["categories"]))
        blend = self.settings["blend"]
        center, low, high = apply_blend(live, self.pre[ahead], horizons, np.full(len(ahead), known),
                                        blend["weights"], blend["scale"])
        forecasts = []
        earliest = pd.Timestamp(latest["actual_at"]).tz_localize(None)
        for k, segment in enumerate(journey.iloc[ahead].itertuples()):
            if pd.isna(segment.sch_arr_B):
                continue
            # A downstream arrival cannot happen before the train was observed at this stop.
            floor = (earliest - segment.sch_arr_B).total_seconds() / 60
            forecasts.append({
                "stop_index": int(segment.seg_idx) + 1, "station": segment.to_stn,
                "scheduled_at": segment.sch_arr_B.isoformat(), "from_station": latest["station"],
                "from_type": latest["type"], "available_at": event["available_at"],
                "observed_delay": known, "stops_ahead": int(horizons[k]),
                "pre": float(self.pre[ahead[k], 1]), "raw_delay": float(center[k]),
                "delay": max(float(center[k]), floor), "low": max(float(low[k]), floor), "high": max(float(high[k]), floor),
            })
        return forecasts
