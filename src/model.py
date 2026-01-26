import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from datetime import datetime
import os

class VisaSurvivalModel:
    def __init__(self, raw_data_path="data/raw_data.csv"):
        self.df = pd.read_csv(raw_data_path)
        self.preprocess()

    def preprocess(self):
        # Convert dates
        self.df['check_date'] = pd.to_datetime(self.df['check_date'], format="%Y-%m-%d", errors='coerce')
        self.df['complete_date'] = pd.to_datetime(self.df['complete_date'], format="%Y-%m-%d", errors='coerce')
        
        # Calculate t (observed time)
        # For Clear/Reject, t is waiting_days. For Pending, t is days since check_date to "now"
        now = datetime.now()  # Use actual current time
        
        def calc_t(row):
            if row['status'] in ['Clear', 'Reject'] and row['complete_date'] is not pd.NaT:
                days = (row['complete_date'] - row['check_date']).days
                return max(days, 1)
            else:
                days = (now - row['check_date']).days
                return max(days, 1)

        self.df['t'] = self.df.apply(calc_t, axis=1)
        self.df['event'] = self.df['status'].apply(lambda x: 1 if x in ['Clear', 'Reject'] else 0)
        
        # Filter out extreme outliers or impossible dates
        self.df = self.df[self.df['t'] < 500].copy()
        
        # Major bucketing - strict regex with word boundaries
        import re
        def bucket_major(m):
            m = str(m).lower()
            
            # CS & AI (Strict boundaries to avoid "physics"->cs)
            # Matches: "cs", "computer science", "ml", "ai" etc.
            cs_pat = r"\b(cs|cse|eecs|computer\s+science|soft|ai|machine\s+learning|ml|nlp|data|algorithm|vision)\b"
            if re.search(cs_pat, m):
                return 'CS'
                
            # ECE & Robotics
            ece_pat = r"\b(ee|ece|elect|robot|circuit|micro|nano|semiconductor)\b"
            if re.search(ece_pat, m):
                return 'ECE'
                
            # Other STEM
            stem_pat = r"\b(bio|chem|phys|mater|math|stat|mech|civil|aero|nuclear|health|med)\b"
            if re.search(stem_pat, m):
                return 'STEM'
                
            return 'Other'
            
        self.df['major_bucket'] = self.df['major'].apply(bucket_major)

    def kaplan_meier(self, consulate_list=None, visa_type=None, major_bucket=None, max_days=365):
        """
        Compute Kaplan-Meier survival curve for a cohort.
        Returns: times, survival probabilities, and quantile estimates.
        """
        # Filter cohort
        mask = pd.Series(True, index=self.df.index)
        if consulate_list:
            mask &= self.df['consulate'].isin(consulate_list)
        if visa_type:
            mask &= (self.df['visa_type'] == visa_type)
        if major_bucket:
            mask &= (self.df['major_bucket'] == major_bucket)
        
        cohort = self.df[mask].copy()
        
        if len(cohort) == 0:
            return None
        
        # Sort by observed time
        cohort = cohort.sort_values('t')
        
        times = []
        survival = []
        variance_sum = 0  # For Greenwood formula
        n_at_risk = len(cohort)
        s = 1.0
        
        unique_t = sorted(cohort['t'].unique())
        
        for t in unique_t:
            if t > max_days:
                break
            events_at_t = len(cohort[(cohort['t'] == t) & (cohort['event'] == 1)])
            censored_at_t = len(cohort[(cohort['t'] == t) & (cohort['event'] == 0)])
            
            if n_at_risk > 0 and events_at_t > 0:
                s = s * (1 - events_at_t / n_at_risk)
                # Greenwood variance increment: d / (n * (n - d))
                if n_at_risk > events_at_t:
                    variance_sum += events_at_t / (n_at_risk * (n_at_risk - events_at_t))
            
            times.append(t)
            survival.append(s)
            n_at_risk -= (events_at_t + censored_at_t)
        
        # Compute Greenwood 95% CI
        se = [s * np.sqrt(variance_sum) for s in survival]  # Simplified; proper is cumulative
        ci_lower = [max(0, s - 1.96 * se_i) for s, se_i in zip(survival, se)]
        ci_upper = [min(1, s + 1.96 * se_i) for s, se_i in zip(survival, se)]
        
        # Compute quantiles
        quantiles = {}
        for q in [0.25, 0.5, 0.75, 0.9]:
            target_s = 1 - q
            for i, s_val in enumerate(survival):
                if s_val <= target_s:
                    quantiles[f"P{int(q*100)}"] = times[i]
                    break
            else:
                quantiles[f"P{int(q*100)}"] = None  # Not reached
        
        return {
            'times': times,
            'survival': survival,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'quantiles': quantiles,
            'n_total': len(cohort),
            'n_events': len(cohort[cohort['event'] == 1])
        }
    
    def km_conditional_quantiles(self, km_result, t0):
        """
        Compute conditional quantiles from KM: P(T | T > t0).
        Returns quantiles of remaining time distribution.
        """
        if km_result is None:
            return None
        
        times = km_result['times']
        survival = km_result['survival']
        
        # Find S(t0)
        s_t0 = 1.0
        for i, t in enumerate(times):
            if t >= t0:
                s_t0 = survival[i]
                break
        
        if s_t0 < 1e-6:
            return None
        
        # Conditional survival: S(t | T > t0) = S(t) / S(t0)
        cond_quantiles = {}
        for q in [0.25, 0.5, 0.75, 0.9]:
            target_cond_s = 1 - q  # We want P(T > t | T > t0) = 1 - q
            target_s = s_t0 * target_cond_s
            
            for i, s_val in enumerate(survival):
                if times[i] >= t0 and s_val <= target_s:
                    cond_quantiles[f"P{int(q*100)}"] = times[i]
                    break
            else:
                cond_quantiles[f"P{int(q*100)}"] = None
        
        return cond_quantiles

    def get_transparency_report(self, params, t0):
        """
        Return fitted mixture parameters and posterior AP-heavy probability.
        """
        mu1, sigma1, mu2, sigma2, pi_logit = params
        pi = 1 / (1 + np.exp(-pi_logit))
        
        # Survival at t0 for each component
        s1_t0 = norm.sf((np.log(t0) - mu1) / sigma1)
        s2_t0 = norm.sf((np.log(t0) - mu2) / sigma2)
        
        # Posterior P(AP-heavy | T > t0)
        # P(Regime2 | T > t0) = pi * S2(t0) / (pi * S2(t0) + (1-pi) * S1(t0))
        denom = pi * s2_t0 + (1 - pi) * s1_t0
        if denom < 1e-10:
            posterior_ap_heavy = 1.0
        else:
            posterior_ap_heavy = (pi * s2_t0) / denom
        
        return {
            'mu1': mu1,
            'sigma1': sigma1,
            'mu2': mu2,
            'sigma2': sigma2,
            'pi': pi,
            'posterior_ap_heavy': min(posterior_ap_heavy, 0.999),  # Cap at 99.9% for honest display
            'median_regime1': np.exp(mu1),
            'median_regime2': np.exp(mu2)
        }


    def log_likelihood(self, params, t, event, weights):
        """
        Mixture model likelihood:
        - Component 1 (Fast/Routine): Log-normal
        - Component 2 (AP-Heavy): Log-Student-t (heavier tails)
        """
        mu1, sigma1, mu2, sigma2, pi_logit = params
        pi = 1 / (1 + np.exp(-pi_logit))  # Sigmoid to keep pi in [0, 1]
        
        if sigma1 <= 0.01 or sigma2 <= 0.01: return 1e10
        
        # Component 1 (Routine) - Log-normal
        z1 = (np.log(t) - mu1) / sigma1
        f1 = (1 / (t * sigma1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * z1**2)
        s1 = norm.sf(z1)
        
        # Component 2 (AP-Heavy) - Log-Student-t with df=5 (heavier tails)
        # Using Student-t for the log-transformed variable
        from scipy.stats import t as student_t
        df = 5  # Degrees of freedom - lower = heavier tails
        z2 = (np.log(t) - mu2) / sigma2
        
        # PDF of log-t: (1/t) * pdf_t(z2) / sigma2
        f2 = student_t.pdf(z2, df) / (t * sigma2)
        # Survival of log-t: 1 - cdf_t(z2)
        s2 = student_t.sf(z2, df)
        
        # Mixture Likelihood
        prob = event * (pi * f2 + (1 - pi) * f1) + (1 - event) * (pi * s2 + (1 - pi) * s1)
        ll = np.log(np.maximum(prob, 1e-10))
        
        return -np.sum(ll * weights)

    def fit_aft(self, consulate=None, visa_type=None, major_bucket=None, 
                tau=30, ghost_threshold=90, ghost_decay=30):
        # 1. Base Cohort
        mask = pd.Series(True, index=self.df.index)
        if consulate: mask &= (self.df['consulate'] == consulate)
        if visa_type: mask &= (self.df['visa_type'] == visa_type)
        if major_bucket: mask &= (self.df['major_bucket'] == major_bucket)
        
        cohort = self.df[mask].copy()
        
        # Dynamic pooling: Trust local data more when we have it
        local_completed = len(cohort[cohort['event'] == 1])
        
        if local_completed >= 5:
            # Enough local data - use it with full weight, minimal pooling
            cohort['pool_weight'] = 1.0
        elif local_completed >= 2:
            # Some local data - light pooling
            broader_mask = (self.df['visa_type'] == visa_type) & (self.df['major_bucket'] == major_bucket)
            global_cohort = self.df[broader_mask].copy()
            sample_size = min(len(global_cohort), 20)  # Reduced from 30
            if sample_size > 0:
                cohort = pd.concat([cohort, global_cohort.sample(sample_size, random_state=42)])
            cohort['pool_weight'] = 0.7  # Higher weight on local
            cohort.iloc[:len(self.df[mask]), cohort.columns.get_loc('pool_weight')] = 1.0
        else:
            # Very sparse - heavy pooling
            broader_mask = (self.df['visa_type'] == visa_type)
            global_cohort = self.df[broader_mask].copy()
            cohort = pd.concat([cohort, global_cohort.sample(min(len(global_cohort), 30), random_state=42)])
            cohort['pool_weight'] = 0.5
            cohort.iloc[:len(self.df[mask]), cohort.columns.get_loc('pool_weight')] = 1.0

        # 2. Skeptical Ghost Weighting
        def get_ghost_weight(row):
            if row['event'] == 1: return 1.0
            if row['t'] <= ghost_threshold: return 1.0
            return max(0.01, np.exp(-(row['t'] - ghost_threshold) / ghost_decay))

        cohort['ghost_weight'] = cohort.apply(get_ghost_weight, axis=1)

        # 3. Wave-Aware Recency (based on CLEARANCE date for completed, CHECK date for pending)
        # This captures "what just cleared" rather than "what was checked recently"
        def get_recency_date(row):
            if row['event'] == 1 and pd.notna(row['complete_date']):
                return row['complete_date']
            return row['check_date']
        
        cohort['recency_date'] = cohort.apply(get_recency_date, axis=1)
        max_date = cohort['recency_date'].max()
        cohort['recency_weight'] = np.exp(-(max_date - cohort['recency_date']).dt.days / tau)
        
        cohort['final_weight'] = cohort['pool_weight'] * cohort['ghost_weight'] * cohort['recency_weight']

        # Initial guess
        comp = cohort[cohort['event'] == 1]
        t_median = np.median(comp['t']) if not comp.empty else 75
        mu_start = np.log(max(t_median, 1))
        
        x0 = [np.log(14), 0.4, mu_start, 0.5, 0.0]
        
        res = minimize(self.log_likelihood, x0=x0, 
                       args=(cohort['t'].values, cohort['event'].values, cohort['final_weight'].values),
                       bounds=[(1, 5), (0.25, 1.5), (3, 7), (0.25, 1.5), (-5, 5)])  # Wider sigma bounds
        
        return res.x

        return results
        
    def get_similar_cases(self, consulate, visa_type, major_bucket, k=10, min_days=0):
        # Find cases that match the criteria
        mask = (self.df['consulate'] == consulate) & \
               (self.df['visa_type'] == visa_type) & \
               (self.df['major_bucket'] == major_bucket) & \
               (self.df['event'] == 1) & \
               (self.df['t'] >= min_days)  # Filter by minimum wait days
               
        similar = self.df[mask].copy()
        if similar.empty:
            return pd.DataFrame()
            
        # Sort by recency (check_date)
        similar = similar.sort_values('check_date', ascending=False)
        
        return similar.head(k)[['check_date', 'waiting_days', 'major', 'user_handle']]

    def predict_conditional(self, params, t0, calibration_factor=1.0, quantile_steps=8000, expectation_steps=5000):
        """
        Conditional prediction given T > t0.
        calibration_factor: multiplier for interval width (>1 widens intervals)
        """
        mu1, sigma1, mu2, sigma2, pi_logit = params
        pi = 1 / (1 + np.exp(-pi_logit))
        
        from scipy.stats import t as student_t
        df = 5  # Same as in log_likelihood
        
        def S(t):
            # Component 1 (Log-normal)
            s1 = norm.sf((np.log(t) - mu1) / sigma1)
            # Component 2 (Log-Student-t with df=5)
            s2 = student_t.sf((np.log(t) - mu2) / sigma2, df)
            return pi * s2 + (1 - pi) * s1

        def S_vec(t_arr):
            t_arr = np.maximum(np.asarray(t_arr, dtype=float), 1e-9)
            z1 = (np.log(t_arr) - mu1) / sigma1
            z2 = (np.log(t_arr) - mu2) / sigma2
            s1 = norm.sf(z1)
            s2 = student_t.sf(z2, df)
            return pi * s2 + (1 - pi) * s1
            
        s_t0 = S(t0)
        
        # Edge case: If survival probability is extremely low
        if s_t0 < 1e-6:
            return {
                "P10": t0, "P25": t0, "P50": t0, "P75": t0, "P90": t0,
                "ExpectedValue": t0,
                "ProbNext7": 1.0,
                "ProbNext30": 1.0,
                "Warning": f"Extreme tail case (S({t0})={s_t0:.2e}). Model expects this case to have already cleared."
            }
        
        # Calibrated quantiles (inflate intervals by calibration_factor)
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        results = {}
        
        max_search = max(t0 + 400, 600)
        t_range = np.linspace(t0, max_search, quantile_steps)
        s_vals = S_vec(t_range)
        
        # Find P50 first (anchor)
        p50_target = s_t0 * 0.5
        p50_idx = np.abs(s_vals - p50_target).argmin()
        p50 = t_range[p50_idx]
        
        for q in quantiles:
            target = s_t0 * (1 - q)
            idx = np.abs(s_vals - target).argmin()
            raw_quantile = t_range[idx]
            
            # Apply calibration: widen distance from P50
            if calibration_factor != 1.0:
                delta = raw_quantile - p50
                calibrated_quantile = p50 + delta * calibration_factor
                results[f"P{int(q*100)}"] = max(t0, calibrated_quantile)
            else:
                results[f"P{int(q*100)}"] = raw_quantile
            
        # Expected value
        t_int = np.linspace(t0, 2000, expectation_steps)
        dt = t_int[1] - t_int[0]
        s_int = S_vec(t_int)
        results['ExpectedValue'] = t0 + np.sum(s_int / s_t0) * dt
        
        results['ProbNext7'] = (S(t0) - S(t0 + 7)) / s_t0
        results['ProbNext30'] = (S(t0) - S(t0 + 30)) / s_t0
        
        return results

    def validate_model(self, params, test_cases):
        mu1, sigma1, mu2, sigma2, pi_logit = params
        pi = 1 / (1 + np.exp(-pi_logit))
        
        coverage = 0
        log_loss = 0
        completed_count = 0
        
        for _, case in test_cases.iterrows():
            t_actual = case['t']
            event = case['event']
            
            # Mixture components
            # f(t)
            f1 = (1 / (t_actual * sigma1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(t_actual) - mu1) / sigma1)**2)
            f2 = (1 / (t_actual * sigma2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(t_actual) - mu2) / sigma2)**2)
            f = pi * f2 + (1 - pi) * f1
            
            # S(t)
            s1 = norm.sf((np.log(t_actual) - mu1) / sigma1)
            s2 = norm.sf((np.log(t_actual) - mu2) / sigma2)
            S_t = pi * s2 + (1 - pi) * s1
            
            # Log Likelihood contribution
            # If event=1, use density f(t). If event=0 (censored), use survival S(t).
            if event == 1:
                prob = f
            else:
                prob = S_t
                
            log_loss -= np.log(max(prob, 1e-10))
            
            # Coverage check (Only valid for COMPLETED cases)
            if event == 1:
                completed_count += 1
                # Find t10, t90 s.t. S(t)=0.9 and S(t)=0.1
                # This numerical search is expensive inside a loop, optimized slightly?
                # For validation, we can accept slow speed.
                t_eval = np.linspace(1, 400, 4000)
                s_eval = np.array([pi * norm.sf((np.log(t)-mu2)/sigma2) + (1-pi)*norm.sf((np.log(t)-mu1)/sigma1) for t in t_eval])
                
                t10 = t_eval[np.abs(s_eval - 0.9).argmin()]
                t90 = t_eval[np.abs(s_eval - 0.1).argmin()]
                
                if t10 <= t_actual <= t90:
                    coverage += 1
                
        return {
            "coverage_p10_p90": coverage / completed_count if completed_count > 0 else 0,
            "avg_log_loss": log_loss / len(test_cases) if len(test_cases) > 0 else 0
        }

if __name__ == "__main__":
    model = VisaSurvivalModel()
    mu, sigma = model.fit_aft(consulate="GuangZhou", visa_type="H1", major_bucket="CS")
    print(f"Model Parameters: mu={mu:.4f}, sigma={sigma:.4f}")
    
    # User case: Check Date 12/3, t0 = 53 days
    t0 = 53
    forecast = model.predict_conditional(mu, sigma, t0)
    print(f"\nForecast for User (GZ, H1B, CS, 53 days waiting):")
    for k, v in forecast.items():
        if 'Prob' in k:
            print(f"{k}: {v*100:.1f}%")
        else:
            print(f"{k}: {v:.1f} days (Approx {datetime(2025, 12, 3) + pd.Timedelta(days=v)})")
