import { Target } from "lucide-react";
import { checkAccuracy, type ReplayState } from "@/lib/replay-state";
import { minutes } from "@/lib/display";

export function Accuracy({ state }: { state: ReplayState }) {
  const { stats } = state;
  const derived = checkAccuracy(state);
  const signedBias = derived ? Number(derived.bias.toFixed(2)) : null;
  return (
    <section className="panel accuracy" aria-labelledby="accuracy-heading">
      <div className="panel-title">
        <div className="title-icon"><Target size={19} aria-hidden="true" /></div>
        <div><h2 id="accuracy-heading">Forecast accuracy</h2><p>Running results for scored arrivals</p></div>
      </div>
      <div className="accuracy-metrics">
        <div><span>Mean absolute error</span><strong>{stats?.mean_abs_error_min == null ? "—" : <>{stats.mean_abs_error_min.toFixed(2)} <small>min</small></>}</strong></div>
        <div><span>Median absolute error</span><strong>{derived ? <>{derived.medianAbsoluteError.toFixed(2)} <small>min</small></> : "—"}</strong></div>
        <div><span>Bias</span><strong>{signedBias == null ? "—" : <>{signedBias > 0 ? "+" : ""}{signedBias.toFixed(2)} <small>min</small></>}</strong></div>
        <div><span>Inside forecast range</span><strong>{stats?.within_range_pct == null ? "—" : `${stats.within_range_pct.toFixed(1)}%`}</strong></div>
        <div><span>Arrival checks</span><strong>{stats?.checks.toLocaleString("en-IN") ?? "—"}</strong></div>
      </div>
      <p className="accuracy-note">{stats?.checks && !derived
        ? "Median and bias need a recorded error for every scored arrival."
        : "Median and bias use rounded arrival errors. Positive bias means forecasts skew late."}</p>
      {state.recentChecks.length > 0 && (
        <div className="checks">
          <h3>Recent checks</h3>
          {state.recentChecks.slice(-4).reverse().map(({ train, check }, index) => (
            <div className="check-row" key={`${train.journey_id}-${check.stop_no}-${index}`}>
              <span><strong>{check.station}</strong> <small>{train.train_no}</small></span>
              <span title="Forecast minus actual">{minutes(check.error_min)}</span>
              <span className={check.was_in_range ? "check-in" : "check-out"}>{check.was_in_range ? "Inside range" : "Outside range"}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
