import argparse
from datetime import datetime
from src.model import VisaSurvivalModel
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Visa Processing Prediction CLI")
    parser.add_argument("--consulate", type=str, default="GuangZhou", help="Consulate (e.g., GuangZhou, BeiJing, ShangHai)")
    parser.add_argument("--visa_type", type=str, default="H1", help="Visa Type (H1, F1, etc.)")
    parser.add_argument("--check_date", type=str, required=True, help="Check Date (YYYY-MM-DD)")
    parser.add_argument("--major", type=str, default="CS", help="Major/Field of Study")
    parser.add_argument("--as_of", type=str, help="Reference 'current' date for historical checks (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Calculate t0
    check_date = pd.to_datetime(args.check_date)
    reference_now = pd.to_datetime(args.as_of) if args.as_of else datetime.now()
    t0 = (reference_now - check_date).days
    
    print(f"\n--- Visa Prediction Report ---")
    print(f"Profile: {args.consulate} | {args.visa_type} | {args.major}")
    print(f"Check Date: {args.check_date} ({t0} days elapsed as of {reference_now.date()})")
    
    if t0 < 0:
        print("Error: Check date is in the future.")
        return

    model = VisaSurvivalModel()
    
    # Major bucketing - strict regex with word boundaries (matching model.py)
    import re
    def bucket_major(m):
        m = str(m).lower()
        
        # CS & AI (Strict boundaries to avoid "physics"->cs)
        cs_pat = r"\b(cs|cse|eecs|computer\s+science|soft|ai|machine\s+learning|ml|nlp|data|algorithm|vision)\b"
        if re.search(cs_pat, m): return 'CS'
            
        # ECE & Robotics
        ece_pat = r"\b(ee|ece|elect|robot|circuit|micro|nano|semiconductor)\b"
        if re.search(ece_pat, m): return 'ECE'
            
        # Other STEM
        stem_pat = r"\b(bio|chem|phys|mater|math|stat|mech|civil|aero|nuclear|health|med)\b"
        if re.search(stem_pat, m): return 'STEM'
            
        return 'Other'
    
    major_b = bucket_major(args.major)
    
    print("Fitting Mixture Model to cohort (Optimized Params)...")
    # Using winning hyperparameters from evaluate.py: tau=45, ghost_decay=90
    params = model.fit_aft(consulate=args.consulate, visa_type=args.visa_type, major_bucket=major_b,
                           tau=45, ghost_decay=90)
    
    forecast = model.predict_conditional(params, t0, calibration_factor=1.5)
    
    # Check for extreme tail warning
    if 'Warning' in forecast:
        print(f"\n⚠️  WARNING: {forecast['Warning']}")
        print("   This case is far beyond the model's training distribution.")
        print("   The model predicts this case should have ALREADY CLEARED.")
        return
    
    print("\nPROBABILISTIC FORECAST:")
    print(f"  Expected wait: {forecast['ExpectedValue']:.1f} total days")
    print(f"  Estimated Completion: {(check_date + pd.Timedelta(days=forecast['ExpectedValue'])).date()}")
    
    print("\nCONFIDENCE BANDS (Total Wait Time):")
    print(f"  P25 (Early): {forecast['P25']:.1f} days")
    print(f"  P50 (Median): {forecast['P50']:.1f} days")
    print(f"  P75 (Late): {forecast['P75']:.1f} days")
    print(f"  P90 (Tail): {forecast['P90']:.1f} days")
    
    print("\nCLEARANCE PROBABILITY:")
    print(f"  Chance of clearing in next 7 days: {forecast.get('ProbNext7', 0)*100:.1f}%")
    print(f"  Chance of clearing in next 30 days: {forecast.get('ProbNext30', 0)*100:.1f}%")
    
    # Transparency: Nearest Neighbors (AP-regime only: T >= 45 days)
    print("\nTRANSPARENCY: SIMILAR AP-REGIME CLEARED CASES (waited 45+ days)")
    neighbors = model.get_similar_cases(args.consulate, args.visa_type, major_b, k=5, min_days=45)
    if not neighbors.empty:
        for _, row in neighbors.iterrows():
            handle = row['user_handle'] if pd.notna(row['user_handle']) else "User"
            check_d = row['check_date'].date() if pd.notna(row['check_date']) else "N/A"
            wait = int(row['waiting_days']) if pd.notna(row['waiting_days']) else "?"
            print(f"  - {handle:<15} | Checked: {check_d} | Wait: {wait} days")
    else:
        print("  No AP-regime matches found in recent history.")
    
    # Kaplan-Meier Baseline (dynamic based on user's inputs)
    print(f"\n=== KAPLAN-MEIER BASELINE ({args.consulate} {args.visa_type} {major_b}) ===")
    km_result = model.kaplan_meier(consulate_list=[args.consulate], visa_type=args.visa_type, major_bucket=major_b)
    
    # Fallback to broader cohort if not enough data
    if km_result is None or km_result['n_events'] < 5:
        print(f"  (Insufficient {args.consulate} data, falling back to All Consulates)")
        km_result = model.kaplan_meier(consulate_list=None, visa_type=args.visa_type, major_bucket=major_b)
    if km_result:
        print(f"  Cohort Size: {km_result['n_total']} cases ({km_result['n_events']} completed)")
        print(f"  KM Quantiles:")
        for k, v in km_result['quantiles'].items():
            if v:
                print(f"    {k}: {v:.0f} days")
            else:
                print(f"    {k}: Not reached in data")
        
        # Find survival at t0 with CI
        for i, t in enumerate(km_result['times']):
            if t >= t0:
                s_at_t0 = km_result['survival'][i]
                ci_lo = km_result['ci_lower'][i]
                ci_hi = km_result['ci_upper'][i]
                print(f"  Survival at Day {t0}: {s_at_t0*100:.1f}% [{ci_lo*100:.0f}%-{ci_hi*100:.0f}%] still waiting")
                
                # P(extreme | T > t0) via KM with stale-adjustment
                # extreme = T > 130 days
                s_130 = None
                for j, tj in enumerate(km_result['times']):
                    if tj >= 130:
                        s_130 = km_result['survival'][j]
                        break
                
                if s_130 is not None and s_at_t0 > 0:
                    p_extreme_raw = s_130 / s_at_t0
                    
                    # Stale-adjustment: assume X% of pending >120 days are actually cleared
                    # This reduces effective survival at t=130
                    # Stale rates: pessimistic=10%, neutral=40%, optimistic=70%
                    stale_rates = {'Pessimistic': 0.10, 'Neutral': 0.40, 'Optimistic': 0.70}
                    
                    print(f"\n  EXTREME TAIL RISK (T > 130 days | T > {t0}):")
                    print(f"    Raw KM (no stale adjustment):     {p_extreme_raw*100:.0f}%")
                    
                    for name, stale_pct in stale_rates.items():
                        # Adjusted S(130) = S(130) * (1 - stale_pct for long-pending)
                        # Simplified: reduce extreme prob proportionally
                        adj_p_extreme = p_extreme_raw * (1 - stale_pct)
                        print(f"    {name:12s} ({stale_pct*100:.0f}% stale): {adj_p_extreme*100:.0f}%")
                    
                    print(f"    → Realistic range: {p_extreme_raw*(1-0.7)*100:.0f}%-{p_extreme_raw*(1-0.1)*100:.0f}%")
                break
    
    # Mixture Model Transparency
    print("\n=== MIXTURE MODEL TRANSPARENCY ===")
    report = model.get_transparency_report(params, t0)
    print(f"  Regime 1 (Fast): Median={report['median_regime1']:.1f} days, σ={report['sigma1']:.2f}")
    print(f"  Regime 2 (AP-Heavy): Median={report['median_regime2']:.1f} days, σ={report['sigma2']:.2f}")
    print(f"  Mixing Weight π (AP-heavy prior): {report['pi']*100:.1f}%")
    print(f"  Posterior P(AP-heavy | T > {t0}): {report['posterior_ap_heavy']*100:.1f}%")
    
    # Sensitivity Analysis: How forecast changes at future days
    print("\n=== SENSITIVITY ANALYSIS ===")
    print("  How your forecast changes if still waiting at...")
    future_days = [70, 90, 110, 130]
    for future_t0 in future_days:
        if future_t0 <= t0:
            continue
        future_forecast = model.predict_conditional(params, future_t0, calibration_factor=1.5)
        if 'Warning' in future_forecast:
            print(f"    Day {future_t0}: ⚠️ Model expects clearance by then")
        else:
            print(f"    Day {future_t0}: P50={future_forecast['P50']:.0f}d, P90={future_forecast['P90']:.0f}d")
    
    # Agreement Check: Conditional KM P50 vs Mixture P50
    print("\n=== CALIBRATION CHECK ===")
    if km_result:
        cond_km = model.km_conditional_quantiles(km_result, t0)
        if cond_km and cond_km.get('P50'):
            km_p50_cond = cond_km['P50']
            mix_p50 = forecast['P50']
            diff = abs(km_p50_cond - mix_p50)
            status = "✓ Good" if diff < 15 else "⚠️ Divergence"
            print(f"  KM P50 (conditional): {km_p50_cond:.0f} days | Mixture P50: {mix_p50:.0f} days | Δ={diff:.0f}d {status}")
        else:
            print(f"  Mixture P50: {forecast['P50']:.0f} days (Conditional KM P50 not reached - limited tail data)")
    else:
        print(f"  Mixture P50: {forecast['P50']:.0f} days (No KM data)")

if __name__ == "__main__":
    main()
